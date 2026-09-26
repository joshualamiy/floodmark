# Inference API (`src/inference/`)

The contract the backend calls. Needs only `onnxruntime`, `numpy`, `pillow`
(no TensorFlow, no network, no `models/` to import the package -- `models/`
is only needed to actually run `load_models()`/`predict()`).

## `load_models(model_dir=None)`

Reads `<model_dir>/config.json`, creates two `onnxruntime` CPU
`InferenceSession`s, and caches the result keyed by the resolved
`model_dir`. `model_dir` defaults to `flood-ml/models`, overridable by the
`FLOODML_MODEL_DIR` env var, then by the `model_dir` argument (which wins).

## `predict(image, models=None, heatmap=True, camera_id=None) -> Prediction`

`image`: a `PIL.Image`, an RGB uint8 `ndarray` (HxWx3; RGBA and grayscale
are also accepted and converted), raw JPEG/PNG bytes, or a path (`str` or
`Path`). `models=None` calls `load_models()` internally. `camera_id` is
accepted for forward compatibility / logging -- smoothing and blocklisting
are `TemporalSmoother`'s job, not `predict()`'s.

`predict_batch(images, ...) -> list[Prediction]` is the batch form; results
are identical to calling `predict()` per image.

### `Prediction` fields

| Field | Type | Meaning |
|---|---|---|
| `status` | `"dry" \| "wet" \| "flooded"` | see status logic below |
| `confidence` | float | `stage_probabilities[status]` |
| `stage_a_probs` | `{dry, wet}` | Stage A's own two-class output |
| `stage_b_probs` | `{not_flooded, flooded}` | Stage B's own two-class output (always computed) |
| `stage_probabilities` | `{dry, wet, flooded}` | sums to 1; matches root `docs/API.md` |
| `heatmap_png` | bytes or `None` | PNG overlay, `None` if `heatmap=False` |
| `thresholds` | `{tA, tB}` | thresholds used for this call |
| `model_version` | `{data_version, stage_a_run_id, stage_b_run_id}` | for logging/debugging |
| `note` | str or `None` | set on `wet` predictions (see below), or by a blocklisted camera via `TemporalSmoother` |

`Prediction.to_dict(include_heatmap=False)` returns a JSON-safe dict.
`heatmap_png` is `None` unless `include_heatmap=True`, in which case it's a
base64 string (raw bytes aren't JSON-serializable). The backend should
generally keep the `Prediction` object itself (for `.heatmap_png` raw
bytes, e.g. to save/serve as a file) and use `to_dict()` only for
logging/JSON transport of the rest.

### Status logic (`PLAN.md` section 3, unchanged by Phase 5)

```
dry     if pA < tA
flooded if pA >= tA and pB >= tB
wet     otherwise (pA >= tA and pB < tB)

stage_probabilities = {dry: 1-pA, wet: pA*(1-pB), flooded: pA*pB}
confidence = stage_probabilities[status]
```

Stage B always runs (it's cheap, ~1ms), regardless of Stage A's output, so
the heatmap is always available even on frames Stage A calls dry.

### What "wet" actually means

**Read `reports/EVALUATION.md` before wiring this up.** Stage A never
learned to detect plain wet pavement -- on independent test data it called
every true "wet" (rain sheen, no flooding) frame "dry." In practice,
`status == "wet"` means *"water was detected, but the flood score is below
the alert threshold"* -- i.e. a probable flood that isn't confident enough
to alert on, not standing water on an otherwise-dry-looking road. 33 of 36
test frames given `wet` were actually floods.

Every `wet` prediction from `predict()` carries
`note = "water detected below flood alert threshold"`. Surface this text
(or your own phrasing of the same idea) anywhere the frontend shows `wet`.

## Preprocessing (`src/inference/preprocess.py`)

Matches the training path as closely as a PIL-only pipeline can. `mode` (from
`models/config.json`'s `preprocess.mode`; **old configs without "mode" keep
working as "crop"**) picks the input geometry -- see improve_v2
(`docs/phase_reports/improve_v2.md`) for why a crop-only pipeline misses
off-center floods on wide frames:

- **`"crop"`** (original behavior): resize so the short side is
  `preprocess.resize_short_side` (default 256px, `Image.LANCZOS`, no-op if
  already smaller), then center crop/pad to `size` x `size`.
- **`"squash"`**: resize the whole frame to `size` x `size`, aspect ignored.
- **`"letterbox"`**: resize the long side to `size`, then pad to `size` x
  `size` with flat mid-gray (128,128,128), centered.

All three optionally JPEG q95 round-trip first (`preprocess.jpeg_roundtrip`,
on by default). `size` comes from `preprocess.size` (falls back to
`input.size`, then 224).

**Preprocessing agreement** (crop mode), measured on 100 random val rows
(`orig_path` run through this pipeline vs. the stored processed `path`
JPEG with a plain center crop, both through the real ONNX models):

| | max\|ΔpA\| | max\|ΔpB\| | status flips |
|---|---|---|---|
| without JPEG round-trip | 0.114 | 0.131 | 0/100 |
| **with JPEG round-trip (shipped)** | **0.038** | **0.005** | 0/100 |

The round-trip measurably tightens agreement, so it's on by default. Numbers
are also recorded in `models/config.json`'s `preprocess` block.

**Heatmap mapping is mode-specific** (`src/inference/heatmap.py`): for
`"crop"` the CAM is placed back inside the crop box on the original frame;
for `"squash"` the CAM covers the *whole* original frame (resized back,
aspect ignored, same as the forward mapping); for `"letterbox"` the CAM's
padded borders are cropped out first, then the remaining content box is
resized back onto the whole original frame.

## Heatmap (`src/inference/heatmap.py`)

Always built from **Stage B's** `cam` (`ReLU(sum_k w_k*A_k)`, 7x7, raw --
"where the model sees water"), regardless of `status`:

1. Bilinear-upsample 7x7 -> 224x224 (the crop box).
2. Per-image min-max normalize; if the CAM is ~all-zero, the heat is blank
   (still returns a valid PNG -- just the plain frame, no color).
3. Map the 224x224 crop box back onto the *original* input frame's
   coordinates (inverting the resize scale factor); pixels outside the crop
   get no heat.
4. Colorize with a numpy jet-style colormap, alpha-blend (alpha 0.45) over
   the original frame.
5. Downscale (if needed) so the long side is <= 640px, aspect preserved.
6. Encode as PNG bytes.

## `TemporalSmoother(n=3, blocklist=())`

Per-camera consecutive-frame confirmation. `update(camera_id,
prediction_or_status) -> SmoothedStatus(status, note)` (a `NamedTuple`;
unpacks as `status, note = smoother.update(...)`, or use `.status`):

- Reports `"flooded"` only after `n` consecutive raw `"flooded"` frames for
  that camera; before that, a raw `"flooded"` frame is reported as `"wet"`.
- A non-`"flooded"` raw frame resets that camera's streak to 0.
- `skip(camera_id)` (e.g. after `CameraMoveDetector` flags a moved frame)
  registers the camera but **does not** touch its streak -- the skipped
  frame neither counts toward `n` nor resets it.
- Cameras in `blocklist` never report `"flooded"`: a raw `"flooded"` becomes
  `("wet", "camera blocklisted; flood alerts suppressed")`. The demo ships
  camera `11372` as a suggested blocklist entry (see below).
- `to_dict()`/`from_dict()` round-trip `n`, `blocklist`, and per-camera
  streak counts, so the backend can persist state across restarts.

## `CameraMoveDetector(k_ref=3, threshold=0.35)`

Flags a re-aimed camera so the backend can skip the frame instead of
misclassifying it. **Structure only** (grayscale -> 64x64 downscale ->
Sobel edge magnitude), never color or brightness, so day/night, rain, and
headlights don't trigger it.

`update(camera_id, frame) -> (moved: bool, score: float)`. The first
`k_ref` frames for a camera build its reference (the per-pixel median of
their edge maps) and always return `(False, 1.0)`. After that, `score` is
the best normalized cross-correlation between the reference and the new
frame's edge map over small (+-2px) shifts, so a few pixels of shake/jitter
don't false-trigger. `moved = score < threshold`. `set_reference(camera_id,
frame)` sets the reference explicitly instead of averaging the first
`k_ref` frames. `to_dict()`/`from_dict()` persist the reference maps.

**Calibration**, on real 511GA data (`data/ga511/frames.csv`, good frames
only): 1,263 same-camera frame pairs (proxy "not moved") vs. 2,000
cross-camera pairs (proxy "moved"):

| Threshold | False-trigger rate (same camera) | Detection rate (different camera) |
|---|---|---|
| 0.30 | 2.9% (36/1263) | 81.9% |
| **0.35 (shipped)** | **4.4% (55/1263)** | **90.0%** |
| 0.40 (best Youden's J = 0.88) | 7.2% | 95.2% |
| 0.50 | 17.7% | 99.3% |

0.35 was chosen over the Youden's-J-optimal 0.40 because a false trigger
here only costs a skipped frame (no misclassification), so it's worth
biasing toward fewer false skips. Also checked on synthetic scenes:
brightness/gamma/darkening changes on the same scene score 0.86-0.99 (not
moved); a heavy crop/zoom or a different scene scores -0.09-0.22 (moved) --
see `tests/test_inference_camera_move.py`.

## Latency

From `reports/onnx_parity.md` (Apple M5 Max, `onnxruntime` CPU, batch 1,
median/p95 over 100 runs): Stage A ~1.0ms, Stage B ~1.1ms, full pipeline
~2.6ms at default thread count. 511GA samples about once per view per hour
by default; even a live demo polling every few seconds has orders of
magnitude of headroom.

## Suggested changes to root `docs/API.md`

1. **Input format**: `predict(image)` — resolve the "PIL Image or RGB numpy
   array; *TBD*" question: `predict()` here accepts PIL, RGB/RGBA/grayscale
   ndarray, raw bytes, or a path, and does all resizing/cropping internally.
   The backend can hand it whatever it already has in hand.
2. **Heatmap resolution**: not fixed and not the same as the input frame's
   pixel dimensions in the sense of being computed at that resolution --
   it's a 7x7 CAM upsampled and mapped back onto the input frame's
   coordinates, then capped at 640px on the long side (so a 4K frame's
   heatmap is downscaled, not native-res). Suggest `docs/API.md` state the
   640px cap explicitly, since the frontend will need to scale the overlay
   back up if it wants to match a larger displayed frame.
3. **Minimum confidence before reporting `flooded`**: this is effectively
   answered by `threshold_tB` (calibrated for >=90% precision on `flooded`)
   plus `TemporalSmoother`'s N-consecutive-frame confirmation (default
   N=3) and the per-camera blocklist, rather than a single confidence
   cutoff applied at request time. Suggest the backend hold smoother state
   per camera (see `to_dict`/`from_dict`) rather than trusting a single
   frame's `status`.
4. **What "wet" means**: `docs/API.md`'s example shows `wet` at confidence
   0.72 without qualification. Suggest documenting it as "water detected,
   below the flood alert threshold (possible flooding)" rather than "wet
   pavement" -- see the section above; this model cannot currently
   distinguish the two, and test data shows most `wet` calls are actually
   under-confident floods, not rain-slicked dry roads.
