# floodmark ML

Two-stage classifier for 511GA (Georgia DOT) traffic camera frames around
Atlanta: is the road surface flooded?

- **Stage A:** dry vs. wet-or-flooded road surface.
- **Stage B:** for wet frames, flooded vs. not.
- Both stages export to ONNX with a Grad-CAM-style heatmap output built
  in-graph, so serving needs only `onnxruntime` + `numpy` + `pillow` (no
  TensorFlow, no gradients at inference time).
- Status logic: `dry` if `pA < tA`; else `flooded` if `pB >= tB`; else `wet`.
  See `docs/INFERENCE_API.md` for the full contract.

The demo separates **flood-score attribution** from the optional **experimental
predicted-water overlay**. Weak classifier evidence leaves the frame uncolored;
raw attribution is opt-in for debugging. Heatmap colors are not pixel probabilities,
water depth, or a road-safety assessment. The shipped classifier (v4) sees the whole
frame (letterbox to 320 px) and adds daytime Atlanta frames; v3 is kept in `models/v3/`
and the older center-crop model in `models/v1/`.

`docs/` (plan, progress log, dataset notes, inference API, backend handoff,
phase reports) is kept local and not in the repo; ask the ML side for it. Honest
model limits: `reports/EVALUATION.md`.

## Setup

Always use the shared venv's python directly, never bare `pip`:

```
/path/to/my_env/bin/python -m pip install -r flood-ml/requirements.txt        # backend/inference only
/path/to/my_env/bin/python -m pip install -r flood-ml/requirements-train.txt  # + data prep, training, eval, demo
```

`requirements.txt` is the light runtime CI installs (numpy, pillow,
onnxruntime, onnx, all pinned to versions with Python 3.11 wheels).
`requirements-train.txt` adds TensorFlow, tf2onnx, pandas, CLIP/torch, the
dataset-download clients, and Gradio; it is not needed to just run inference.

All commands below assume `cd flood-ml` and run with
`PYTHONPATH=src ../my_env/bin/python -m <module>`.

## Data collection (511GA collector)

`src/ga511/` polls the 511GA camera API and snapshot images, drops dead
frames (placeholders, frozen repeats, HTTP errors), and weak-labels frames
by precipitation. Runs as a detached daemon (see
`docs/phase_reports/phase1_ga511.md` for full detail):

```
# start (detached; also see phase1_ga511.md for the exact Popen invocation)
PYTHONPATH=src ../my_env/bin/python -m ga511.daemon run --interval-min 60 \
    --event-poll-min 5 --weather-fill-min 15 &

# status: running, pid, frames_ok, frames_dead_by_reason, last_frame_timestamp_utc, ...
PYTHONPATH=src ../my_env/bin/python -m ga511.daemon status

# stop
PYTHONPATH=src ../my_env/bin/python -m ga511.daemon stop
```

## Labeling tool

A local page for hand-labeling 511GA frames (keys: 1=dry, 2=wet, 3=flooded,
0=unusable, arrows to navigate):

```
PYTHONPATH=src ../my_env/bin/python -m prep.label_tool --port 8000 --ga511-root data/ga511
```

## Data prep

`src/prep/build_manifest.py` builds `data/processed/manifest.csv` end to
end: resizes/copies images, applies mask-based or heuristic flood labeling,
runs the CLIP road-scene filter, dedups by perceptual hash, and writes
group-aware train/val/test splits.

```
PYTHONPATH=src ../my_env/bin/python -m prep.build_manifest [--smoke N] [--skip-clip]
```

Reports: `reports/class_counts.md`, `reports/figures/class_counts.png`,
`data/processed/splits_report.json`.

## Training

```
PYTHONPATH=src ../my_env/bin/python -m train.train --stage a --backbone mobilenetv3small \
    --epochs-head 8 --epochs-ft 6 --ft-layers 30 --mode letterbox --img-size 224
PYTHONPATH=src ../my_env/bin/python -m train.train --stage b --variant mixed \
    --backbone mobilenetv3small --epochs-head 8 --epochs-ft 6 --mode letterbox --img-size 224
```

`--mode` picks the input geometry: `crop` (short-side resize + crop, the
original behavior), `squash` (whole frame -> SxS, aspect ignored), or
`letterbox` (long side -> S, pad to SxS with mid-gray; available as an experiment, see `docs/phase_reports/improve_v2.md`, since a crop-only
pipeline only sees the middle of a wide 511GA frame). `--img-size` (S) must
match between Stage A and Stage B training, `pipeline_eval`, `export_onnx`,
and inference (`models/config.json`'s `preprocess` block).

Each run appends a row to `reports/runs.csv` and writes TensorBoard logs to
`logs/<run_id>/`:

```
../my_env/bin/tensorboard --logdir flood-ml/logs
```

Pick thresholds and the Stage B variant with `train.pipeline_eval`, then
export both stages to ONNX (also verifies parity vs. Keras and benchmarks
CPU latency):

```
PYTHONPATH=src ../my_env/bin/python -m train.pipeline_eval --stage-a-run <run_id> \
    --stage-b-mixed-run <run_id> --out-json reports/pipeline_selection.json
PYTHONPATH=src ../my_env/bin/python -m train.export_onnx --run-id <run_id> --stage a --out models/stage_a.onnx
PYTHONPATH=src ../my_env/bin/python -m train.export_onnx --run-id <run_id> --stage b --out models/stage_b.onnx
```

GPU option: `notebooks/train_colab.ipynb` clones the `ml` branch and runs
`train.py` on Colab. Restricted data (Flood Master) must only go to
team-controlled storage if you use it there.

## Evaluation

`src/eval/` is an independent evaluation harness (Phase 4) against the
shipped ONNX models on `split == test` only: metrics, PR curves, shortcut
and leakage checks, Grad-CAM energy, failure slices, and an error gallery.
See the `Reproduce` section of `reports/EVALUATION.md` for every command.

## Inference

`src/inference/` is the package the backend imports. It needs only
`onnxruntime`, `numpy`, and `pillow` -- no TensorFlow.

```python
from inference import load_models, predict, TemporalSmoother, CameraMoveDetector

models = load_models()  # reads models/config.json, caches onnxruntime sessions
smoother = TemporalSmoother(n=3, blocklist=["11372", "17397", "13750"])

pred = predict(frame, models=models, camera_id="11372")  # PIL/ndarray/bytes/path
status = smoother.update("11372", pred).status
print(pred.status, pred.confidence, pred.stage_probabilities, pred.note)
# pred.heatmap_png is PNG bytes (or None if heatmap=False), <=640px on the long side
```

**"wet" means water was detected but the flood score is below the alert
threshold (possible flooding), not "wet pavement" -- see Known limits.**
`pred.note` explains this on every `wet` prediction. Full contract, field
meanings, and latency numbers: `docs/INFERENCE_API.md`.

CLI over a folder of frames (sorted by name, treated as one camera's
sequence when `--camera-id` is given):

```
PYTHONPATH=src ../my_env/bin/python -m inference.cli data/ga511/frames/<view_id> \
    --camera-id <camera_id> --smooth-n 3 --json --save-heatmaps /tmp/heatmaps \
    --blocklist 11372,17397,13750
```

Gradio demo (local only, `share=False`):

```
PYTHONPATH=src ../my_env/bin/python src/demo_app.py   # -> http://127.0.0.1:7860
```

## Datasets and licenses

No dataset images live in this repo (`data/` is gitignored). Full detail:
`docs/DATASETS.md`, `data/raw/SOURCES.md`, `docs/FLOOD_MASTER.md`.

| Dataset | License | Role |
|---|---|---|
| Roadway Flooding Image Dataset | CC BY 4.0 | flooded/wet road positives |
| Flood Area Segmentation | CC0 1.0 | excluded (almost all aerial) |
| FRED (Flooded Road Environments Dataset) | CC BY-NC-SA 4.0 | dry/flooded, vehicle-mounted |
| NYSDOT Road Surface Conditions | CC BY 4.0 | wet-but-not-flooded gap fill |
| TinyCamML | MIT | reference only (no labeled roadway images used) |
| Flood Master Database | Non-commercial research, no redistribution (AIIA Lab, AUTH) | cleaned masks + external test videos |
| 511GA (Georgia DOT) camera feed | Public feed; internal research use | Atlanta-area train/val/test frames |
| Iowa DOT RWIS webcams (Iowa Environmental Mesonet) | Public domain | real wet/dry DOT camera frames (user-labeled) |
| European Flood 2013 (cvjena, Wikimedia Commons) | per-image CC BY / CC BY-SA / CC0 | flooded + not-flooded street photos |
| AlleyFloodNet (Lee & Joo 2025) | CC BY 4.0 (dataset-level; some watermarked photos) | flooded + not-flooded alleys |

**Excluded:** Water Segmentation Dataset (V-FloodNet lineage) -- its license
is unverified (Kaggle says "unknown"; the upstream V-FloodNet repo says all
rights reserved), and two of its sub-collections carry their own
non-commercial or no-redistribution terms. Never used for training, and
deleted locally in the final cleanup (`acquire.public_datasets` can re-fetch it). FloodNet (drone imagery) and RSCD were never
downloaded -- out of scope (aerial view) or unverified license,
respectively. See `docs/DATASETS.md` for the full "not downloaded" table and
the gap-search candidates.

Any weights trained on FRED or the Flood Master Database inherit
non-commercial terms, so trained model files are **not** published in this
public repo (see "Where the models live" below).

## Known limits

Full detail: `reports/EVALUATION.md` (v4 on top, then v3, then v1). In short, for v4 on held-out test data:

- Flood recall 0.85 (463/542), precision 0.91 (463/507). v3 caught more floods (483/542) but also more false ones. On the original test set: recall 0.68 (115/168, v3 0.75), precision 1.00.
- Elevated fixed-camera flood video: 0.34 recall (21/62, one video; v3 0.55). A real 511GA title bar pasted on flood photos hides many floods.
- **Wet pavement is not detected** (6/31 true wet frames get status "wet"). Rain-wet streets can read as **flooded**: 38% of test "not flooded" street photos (43/112). Expect false alerts in the first real rain.
- Live Atlanta cameras, held-out, one dry day: 0/1,099 daytime and 2/779 night frames called flooded (v3: 21/1,099 daytime). Not yet tested on a real Atlanta flood or a rainy day.
- Night infrared views with no road (cameras 13539, 14188) can be called flooded. Use `TemporalSmoother` (3 in a row) and the blocklist (11372, 17397, 13750).
- `CameraMoveDetector` builds its reference from each camera's first frames (night here), so it flags ~27% of daytime frames as moved.
- `status == "wet"` means water detected below the flood alert threshold (possible flooding), not wet pavement.

## Where the models live

`models/stage_a.onnx`, `models/stage_b.onnx`, and `models/config.json` are
**not committed** (`models/` is gitignored) -- both stages are fine-tuned in
part on FRED (CC BY-NC-SA 4.0) and the Flood Master Database (non-commercial,
no-redistribution), so the weights can't be published in this public repo.
Get them from the team's private share (`models/handoff/floodmark_models_v4.zip`; see
`docs/BACKEND_HANDOFF.md`). Some training images are CC BY-SA, which is another reason
the weights stay private.

`FLOODML_MODEL_DIR` (or `load_models(model_dir=...)`) points `src/inference`
at wherever you put them; it defaults to `flood-ml/models`.

## Credit

This project uses the **Flood Master Database**, created by the AIIA Lab,
Aristotle University of Thessaloniki (AUTH):
<https://aiia.csd.auth.gr/flood-master-database/>. Used under a signed,
non-commercial research license; the raw data is not redistributed and does
not appear anywhere in this repository.
