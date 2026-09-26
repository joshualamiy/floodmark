# Water-only segmentation baseline

Status: frozen for independent review. No held-out test pixels were opened or evaluated.
Run: `water_20260926_mnv3s320_v1`. Work stayed on branch `ml`; no commit was made.

## Result and limits

A small MobileNetV3Small encoder with a shallow skip decoder was actually trained
at 320x320, using only existing locally cached ImageNet weights. The encoder is
frozen; there are 901,257 parameters total and 19,737 trainable decoder parameters.
Eight epochs completed, and epoch 7 was selected using validation source-macro
pixel IoU at threshold 0.5. A fixed five-point validation sweep
`{0.3, 0.4, 0.5, 0.6, 0.7}` then selected threshold **0.3**. There was one run,
with no further model experiments. Loading, training, export and parity took
37.6 seconds on CPU (original-frame evaluation followed).

These are mask metrics from the frozen ONNX artifact, after removing letterbox
padding and restoring probabilities to each original frame, then thresholding:

| Validation source | Frames | Pixel IoU | Pixel Dice | Mean image IoU | Mean image Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| FRED | 32 | 0.317710 | 0.482216 | 0.238333 | 0.307268 |
| Roadway Flooding | 154 | 0.809527 | 0.894739 | 0.793225 | 0.875148 |

Pixel IoU/Dice pool TP/FP/FN within each source. Mean image scores give each image
equal weight; both empty truth and empty prediction score 1. Classification
confidence, class labels and classifier CAM are not inputs to these metrics.

FRED generalization is weak: validation covers only Pullenvale, with 14
water-positive masks and 18 genuinely annotated empty masks. Fifteen of those
18 empty masks receive some false-positive water; their pooled predicted water
area is 1.356% of pixels. Roadway validation has no empty masks. The model has no
validated GA511/Iowa camera-domain segmentation performance, and shallow 320px
features can miss narrow water boundaries and overpaint objects or dry pavement.
These validation scores were used for selection, so they are not unbiased
held-out performance estimates.

**This is water segmentation, not roadway segmentation or road-flood extent.**
River and roadside water can be highlighted. Keep it in a separate optional
panel, retain `road=false` wording, and do not change classifier status from this
output. Main/reviewer should decide whether this baseline is suitable to expose.

## Data, geometry and leakage

- Training: 308 annotated frames, comprising 125/633 eligible FRED frames and
  all 183 eligible Roadway frames. Fixed seed 42; stable SHA256 order with group
  round-robin; source cap 192 and group cap 64. FRED counts are Mount Cotton 64,
  Holmview 42, Dairy Creek 19. All 186 supported validation masks were used.
- Existing manifest group assignments were preserved. The guard checks group
  identity within source, plus duplicate cluster, exact original path, mask path
  and perceptual hash across sources/splits whenever an identity involves either
  supported source. No conflicts involve the water sources. Four existing
  cross-split duplicate clusters (one exact perceptual hash) involve only
  unannotated GA511 classification rows; they are outside this training sample
  and were not modified. This is a metadata guard, not a new near-duplicate or
  content-level audit.
- Only rows with both `orig_path` and a real `mask_path` are eligible. Unannotated
  dry sequences, GA511, NYSDOT and unsupported sources are excluded. No zero masks
  are synthesized from classification labels.
- FRED: label 2 is water, label 1 is road, label 0 is other. All 665 train/val
  masks inspected were 1728x1080 for 1920x1200 original frames, with equal aspect
  ratio. Masks are resized to the original frame using nearest neighbor.
- Roadway: all 337 train/val masks inspected contain binary labels 0/1,
  with 1 marking water. Masks and originals are both 512x384. Visual inspection
  of four training pairs per source confirmed that water is the positive region.
  The existing prep source reader independently documents these semantics.
- Unknown label values and mask/frame aspect-ratio mismatches fail explicitly.
  Equal-aspect alignment was visually spot-checked, not exhaustively reviewed.
- Training uses full original frames, bilinear image resize, nearest-neighbor
  target resize, centered gray letterboxing and a valid-pixel mask. Padding is
  excluded from BCE, soft Dice and canvas selection metrics. Seeded horizontal
  flips are the only augmentation. There is no crop or JPEG round trip.
- Original-frame evaluation uses nearest-neighbor ground-truth geometry and
  bilinear restored probabilities. Each source is reported separately because
  source sizes, annotation styles and scene distributions differ.

## Frozen artifact and parity

Default private artifact: `models/water/water.onnx` (3,643,191 bytes).
Configuration: `models/water/config.json`. Selected Keras checkpoint and exact
ONNX/config copies: `models/water/water_20260926_mnv3s320_v1/`.

ONNX SHA256:

```text
a68e2bfe23cc335e79d8f359d53dfd3f45284bf3c051fd79c9cdaf522ec4c27b
```

Manifest SHA256:

```text
703f0040ee222528e86c7b0d8a11b6f7093a43bc1b9f71c3ac9100816de418d6
```

Pretrained weights SHA256:

```text
002803a73df2ab65425267ee2082d1ec213b6aa9716f19ac1a1ed5361239c5f6
```

ONNX opset 17, NHWC float32 RGB input `image` in [0,255], output
`water_prob` shaped [N,320,320,1]. TensorFlow/ONNX parity on all 186 validation
frames: max absolute probability error **3.439188e-5**, mean **5.664178e-7**,
six valid canvas pixels differing after thresholding. Maximum-error gate was
1e-4 and passed. Mean batched ONNX time was 3.70 ms per image with two CPU
threads; this excludes image loading, resizing and PNG encoding and is not
end-to-end latency.

## Integration contract

```python
from inference.water import load_water_model, predict_water

model = load_water_model()
if model is not None:
    water = predict_water(image, model=model)  # dict, not a dataclass
```

- `load_water_model(model_dir=None)` is cached by artifact identity. Missing
  ONNX/config returns `None`; missing state is not cached. It supports
  `FLOODML_WATER_MODEL_DIR`. Corrupt artifacts raise.
- `predict_water(image, model=None)` accepts PIL, RGB ndarray, image bytes or
  path. If no model is available it raises `FileNotFoundError`, never silently
  producing a dry/empty prediction.
- Returned **dict** keys: `water_overlay_png` (bytes), `water_mask_png` (bytes),
  `water_fraction` (float), `threshold` (float), `model_version` (str),
  `note` (str).
- Overlay: original-size RGBA PNG, cyan (0,190,255), alpha 112 on predicted water
  and alpha 0 elsewhere. It is a transparent overlay layer, not a composited
  photograph. To show context, alpha-composite it onto the original image.
- Mask: original-size grayscale PNG, values 0 or 255.
  `water_fraction` is the fraction of all original-frame pixels predicted water,
  not a fraction of road pixels and not a flood severity estimate.
- Exact note: “Water-only segmentation; road=false. Not flooded-roadway extent
  or classifier CAM. River water may be highlighted; this model does not identify roads.”
- Runtime dependencies are only standard library, Pillow, NumPy, onnxruntime.
  TensorFlow/Keras/tf2onnx are confined to training. Real-artifact smoke checks
  verified dict output, caching and original geometry on one val image/source.

## Validation and independent Phase4 commands

Run from `flood-ml/` using the existing environment:

```sh
PYTHONPATH=src ../my_env/bin/python -B -m train.water_eval --split val --out reports/water_20260926_mnv3s320_v1/review_val.json
```

Reserved for the independent reviewer **after freeze**; not run by this worker:

```sh
PYTHONPATH=src ../my_env/bin/python -B -m train.water_eval --split test --allow-test --out reports/water_20260926_mnv3s320_v1/phase4_test.json
```

The evaluator checks frozen status, ONNX hash and manifest hash, preserves the
fixed threshold and reports original-frame pixel IoU/Dice by source. It only
supports FRED and Roadway mask semantics; other held-out sources remain excluded.
Test access requires the explicit reviewer opt-in; no tuning should follow
test evaluation.

Command used for the single training run (the trainer refuses to overwrite an
already published artifact):

```sh
PYTHONPATH=src ../my_env/bin/python -B -m train.water_train --run-id water_20260926_mnv3s320_v1 --epochs 8 --max-seconds 420
```

TensorBoard is written per run:

```sh
../my_env/bin/python -m tensorboard.main --logdir logs/water_20260926_mnv3s320_v1/tensorboard
```

## Files and verification

Created `src/inference/water.py`; `src/train/water_data.py`,
`water_model.py`, `water_metrics.py`, `water_train.py`, `water_eval.py`;
three `tests/test_water_*.py` files; this report; and the separate
`reports/water_runs.csv`. No shared run log, classifier, demo, README,
requirements or conftest was edited by this worker.

Private run evidence is under `reports/water_20260926_mnv3s320_v1/`:
`selected_rows.json`, `history.json`, `training_report.json`,
`validation.json`, `runtime_smoke.json`, and source overlay previews.
Training mask inspection grid: `reports/water_audit/mask_semantics.jpg`.
TensorBoard and console/training logs are under the matching `logs/water_*/`.
Weights, raw data and image reports remain ignored; no data/model/env files were
staged or committed.

**25 tests passed** in 2.72 seconds. Coverage includes original portrait/wide/
thin-frame geometry, padding-only predictions, transparent overlays, dict
contract, optional missing artifact/caching, nonfinite output rejection,
runtime import without training dependencies, FRED/roadway label semantics,
nearest-neighbor targets, geometry mismatch rejection, missing-mask exclusion,
deterministic sampling, cross-source leakage, pixel metrics, held-out opt-in,
loss gradients ignoring padding, and a frozen encoder. Training tests call
`pytest.importorskip("tensorflow")` before training imports, without modifying
conftest.

No final tests, export or training steps failed. The initial broad manifest
audit flagged unrelated GA511 duplicates; the guard was narrowed to identities
involving supported water sources while preserving cross-source collision
checks. No extra experiments are needed before main's frozen validation review.
