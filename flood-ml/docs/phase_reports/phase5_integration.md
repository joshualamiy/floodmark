# Phase 5: integration package

Scope: `src/inference/` for the backend, temporal smoothing, camera-move
detection, a CLI, a Gradio demo, tests, README, and `docs/INFERENCE_API.md`.
No packages needed installing -- everything uses what's already in
`requirements.txt` (`numpy`, `pillow`, `onnxruntime`, `onnx` -- the last only
in tests, to build tiny ONNX models with `onnx.helper`) or
`requirements-train.txt` (`gradio`, imported only inside `src/demo_app.py`).

## What I built

- `src/inference/session.py`: `load_models(model_dir=None)`, cached by
  resolved `model_dir` (module-level dict); default
  `flood-ml/models`, overridable by `FLOODML_MODEL_DIR` then by the
  argument.
- `src/inference/preprocess.py`: input normalization (`to_pil` accepts PIL,
  RGB/RGBA/grayscale ndarray, bytes, path) and the resize/JPEG-roundtrip/crop
  pipeline (see below).
- `src/inference/heatmap.py`: 7x7 Stage-B CAM -> upsample -> normalize ->
  map onto the original frame -> jet colormap -> alpha-blend -> cap at
  640px -> PNG bytes. Pure numpy/PIL, no cv2/matplotlib.
- `src/inference/predict.py`: `Prediction` dataclass + `.to_dict()`,
  `predict()`, `predict_batch()`.
- `src/inference/smoothing.py`: `TemporalSmoother` (N-consecutive,
  skip semantics, blocklist, `to_dict`/`from_dict`).
- `src/inference/camera_move.py`: `CameraMoveDetector` (Sobel edges,
  shift-tolerant NCC, median reference over first K frames).
- `src/inference/cli.py`: folder runner, `--camera-id/--smooth-n/--json/
  --save-heatmaps/--blocklist`, wires the smoother and move detector
  together (moved frames are `smoother.skip()`ped, not `update()`d).
- `src/demo_app.py`: Gradio upload -> status/confidence/probs/heatmap +
  a limits blurb. `gradio` imported only in this file.
- `tests/_fake_onnx.py` + 7 new test files (`tests/test_inference_*.py`,
  56 tests) -- see "Tests" below.
- `README.md` (new, at `flood-ml/README.md`), `docs/INFERENCE_API.md` (new).
- `models/config.json`: added a `preprocess` block (kept everything else).

## Preprocessing: what it does, and the agreement numbers

Pipeline: resize so the short side is 256px (`Image.LANCZOS`, **no-op if
already <=256** -- copied from `src/prep/common.py::resize_short_side`;
real 511GA frames are 240-253px short side and are therefore never
upscaled) -> optional JPEG q95 round-trip -> center crop to 224
(zero-padding first if a side is still <224, matching
`tf.image.resize_with_crop_or_pad`).

Measured on 100 random val rows of `data/processed/manifest.csv`: ran the
real ONNX models on (a) `orig_path` through this pipeline and (b) the
stored processed `path` JPEG through a plain center crop (the "gold
standard" of what training actually saw, modulo the PIL-vs-TF decoder
difference already documented in `reports/EVALUATION.md`):

| | max\|ΔpA\| | mean\|ΔpA\| | max\|ΔpB\| | mean\|ΔpB\| | status flips |
|---|---|---|---|---|---|
| without JPEG round-trip | 0.114 | 0.008 | 0.131 | 0.006 | 0/100 |
| **with JPEG round-trip (shipped default)** | **0.038** | **0.0005** | **0.005** | **0.0001** | **0/100** |

The round-trip measurably tightens agreement (about 3x lower max error on
both stages), so it's on by default
(`models/config.json.preprocess.jpeg_roundtrip = true`). Both configs
scored 0 status flips on this sample. Full numbers are recorded in
`models/config.json`'s new `preprocess` block, including the exact
agreement-check summary string.

## Camera-move detector: calibration numbers

Structure-only (grayscale -> 64x64 downscale -> Sobel edge magnitude ->
shift-tolerant NCC over +-2px), so day/night/rain/headlights don't trigger
it. Calibrated on real `data/ga511/frames.csv` (good frames only): 1,263
same-camera frame pairs (first good frame as reference, later frames as
"not moved" queries, spanning whatever day/night variety each camera had)
vs. 2,000 random cross-camera pairs (proxy "moved"):

| Threshold | False-trigger rate (same cam) | Detection rate (diff cam) | Youden's J |
|---|---|---|---|
| 0.30 | 2.9% (36/1263) | 81.9% (1638/2000) | 0.79 |
| **0.35 (shipped)** | **4.4% (55/1263)** | **90.0% (1800/2000)** | 0.86 |
| 0.40 (J-optimal) | 7.2% (91/1263) | 95.2% (1904/2000) | **0.88** |
| 0.50 | 17.7% | 99.3% | 0.82 |

Shipped 0.35 rather than the J-optimal 0.40: a false trigger here just
skips one frame (no misclassification risk), so it's worth trading a little
detection rate for fewer skipped good frames. Also sanity-checked on
synthetic scenes (`tests/test_inference_camera_move.py`): brightness (x1.4),
gamma (0.5), and heavy darkening (x0.25) on the same textured scene all
score 0.86-0.99 (correctly not moved); a heavy center-crop+zoom or a
different scene scores -0.09 to 0.22 (correctly moved).

## CLI on a real 511GA camera

Ran `PYTHONPATH=src ../my_env/bin/python -m inference.cli
data/ga511/frames/18782 --camera-id 11372 --smooth-n 3 --json
--save-heatmaps <dir>` -- view 18782 is 511GA camera **11372**, the one
`reports/EVALUATION.md` flags for repeat false alarms (fired "flooded" on
2 of its 3 captures; defocused grass/guardrail, no road visible). Output
over its 4 real captured frames:

| frame | raw status | pA | pB | smoothed (n=3) |
|---|---|---|---|---|
| 1790384813 | flooded | 0.980 | 0.991 | wet (1/3) |
| 1790385859 | flooded | 0.827 | 0.924 | wet (2/3) |
| 1790388014 | dry | 0.170 | 0.267 | dry (streak reset) |
| 1790393233 | flooded | 0.974 | 0.989 | wet (1/3) |

This exactly reproduces the evaluation's finding, and shows the smoother
doing its job: with N=3 confirmation, this camera never actually reaches a
smoothed `"flooded"` report over this sequence. Re-ran with `--blocklist
11372`: all four frames report `wet` with
`note="camera blocklisted; flood alerts suppressed"`, never `flooded`.
Heatmaps saved correctly (viewed one by eye: a red/orange hotspot on the
defocused grass patch at bottom-left of the frame, matching the
false-alarm region described in the evaluation).

## Gradio demo

Launched detached: `PYTHONPATH=src ../my_env/bin/python src/demo_app.py`.
`curl -o /dev/null -w "%{http_code}" http://127.0.0.1:7860` -> `200`.
Called `demo_app.run()` in-process on a real 511GA frame: returned
`status="flooded"`, `conf="0.971"`, a `stage_probabilities` dict, and a
decoded `PIL.Image` heatmap of the correct frame size. **Left running** at
**http://127.0.0.1:7860** (PID 89700 at report time), `share=False`.

## Tests

56 new tests across 7 files (`tests/test_inference_{predict,heatmap,
preprocess,smoothing,camera_move,cli,integration}.py`), plus
`tests/_fake_onnx.py` (not collected as a test module; builds tiny ONNX
graphs with `onnx.helper` matching the real I/O: input `"image"`
`[N,224,224,3]`, outputs `cam [N,7,7]` then `prob [N,1]` in that declared
order). `prob = sigmoid(prob_slope*mean(image)/255 + prob_bias)` with
`prob_slope=0` by default, so tests hit exact probability targets
regardless of image content via `prob_bias = logit(target)`; `cam` is
either a broadcast constant or `ReLU(cam_scale*pooled_grayscale +
cam_bias)`, a real function of image content, for heatmap tests. Covers:
status/threshold boundaries (including `pA == tA` and `pB == tB` exactly),
`stage_probabilities` summing to 1 and matching `confidence`, every input
type (PIL/ndarray RGB+RGBA+gray/bytes/path), heatmap PNG decoding + size +
640px cap + aspect + the blank-CAM branch, `predict_batch == predict`,
`to_dict()` JSON-safety with/without the heatmap, the smoother (N,
skip, blocklist, independent cameras, serialization), the camera-move
detector (brightness/gamma/darkening invariance, crop/zoom and
different-scene detection, reference building, serialization), the CLI on
a temp folder (text and JSON output, heatmap saving, smoothing reaching
`flooded`, blocklist suppression), and an integration test on the real
shipped ONNX models (skipped via `pytest.mark.skipif` when
`models/stage_a.onnx` is absent -- it's present here, so it ran).

Full suite from the repo root, exactly as CI runs it:
`../my_env/bin/python -m pytest -q flood-ml` -> **226 passed** (170 before
this phase + 56 new). `../my_env/bin/python -m ruff check flood-ml` ->
clean. Nothing in the new tests imports gradio or tensorflow.

## Decisions the orchestrator may want to weigh in on

1. **`camera_id` on `predict()`** is accepted but currently unused inside
   `predict()` itself (kept for forward compatibility / logging; all
   per-camera logic lives in `TemporalSmoother`/`CameraMoveDetector`, which
   both take `camera_id` explicitly). If the backend wants `predict()` to
   auto-apply a camera's blocklist status or similar, that would need a
   small signature/behavior change -- flagging it rather than guessing.
2. **`TemporalSmoother.update()` returns `SmoothedStatus(status, note)`**
   (a `NamedTuple`), not a bare string -- the brief's "reports 'flooded'
   only after N consecutive... reports 'wet'... plus a note" needed
   somewhere to carry the note, and a bare string can't. It unpacks as a
   2-tuple or via `.status`/`.note`. Flagging this as a design choice in
   case the backend team expected a plain string.
3. **CameraMoveDetector threshold (0.35)** trades some detection rate for
   fewer false skips (see numbers above); 0.40 is the empirical
   Youden's-J optimum on this same-vs-different-camera proxy data if the
   team would rather bias the other way.
4. Did **not** touch `docs/API.md` (root, read-only) or `docs/PLAN.md`/
   `docs/PROGRESS.md` per the rules; suggested changes for the team are in
   `docs/INFERENCE_API.md`'s last section.

## Nothing failed / unverified

Everything above ran successfully against the real shipped models
(`models/stage_a.onnx`, `models/stage_b.onnx`, `data_version v1-c7dea35e`).
No new pip packages were installed.
