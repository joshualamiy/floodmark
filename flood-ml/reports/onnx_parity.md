# ONNX export parity and latency

Both models export a `tf.function` (fixed `input_signature`, NHWC float32,
batch `None`) built from the loaded Keras checkpoint, converted with
`tf2onnx.convert.from_function` at opset 17. Each graph has two outputs:
`prob` (N,1) and `cam` (N,h,w) = `ReLU(sum_k w_k * A_k)`, computed in-graph
directly from the last conv feature map and the Dense(1, sigmoid) head's
kernel -- no gradients, no TensorFlow, at inference time. See
`src/train/export_onnx.py`'s module docstring for the one real gotcha found
while building this (a `tf.constant` closed over from outside the traced
function gets exposed as an extra graph input by tf2onnx; building it
*inside* the function body fixes that).

`tf2onnx`'s transpose-optimizer pass throws (and recovers from) a handful of
internal `ValueError`s on every conversion in this environment
(`bias_size = max(numpy_val.shape)` on an empty shape) -- harmless; the
optimizer catches it and falls back, and the resulting graph still passes
`onnx.checker.check_model` and the numeric checks below. Logged to
`logs/jobs/tf2onnx_stage_{a,b}.log` rather than printed.

## Parity (>= 100 val images, never test)

| Stage | Run ID | n images | max \|prob_onnx - prob_keras\| | CAM Pearson r (upsampled, per-image normalized) |
|---|---|---|---|---|
| A (dry vs wet) | `20260926-031525_a_mobilenetv3small` | 150 | 2.56e-06 | 0.999999999902 |
| B (mixed: flooded vs not) | `20260926-031638_bmixed_mobilenetv3small` | 150 | 2.74e-06 | 0.999999999941 |

Both comfortably clear the required bars (max \|Δprob\| <= 1e-4, CAM r >=
0.99). The near-exact CAM match is expected, not a coincidence: the
`cam` op is *defined* as `ReLU(sum_k w_k * A_k)` from the Dense kernel in
both the Keras GradientTape path and the ONNX graph, and per-image
min-max normalization cancels the one remaining degree of freedom (Grad-CAM's
channel weights equal the Dense kernel times a single positive
per-image scalar -- see `src/train/model.py`'s docstring for the derivation).

## CPU latency (batch 1, single image, median / p95 over 100 runs)

| | median (ms) | p95 (ms) |
|---|---|---|
| Stage A, default threads | 1.04 | 1.12 |
| Stage A, 1 thread | 2.86 | 2.94 |
| Stage B, default threads | 1.10 | 1.24 |
| Stage B, 1 thread | 2.86 | 2.95 |
| Full pipeline (A then B sequentially), default threads | 2.57 | 2.81 |
| Full pipeline (A then B sequentially), 1 thread | 5.76 | 5.87 |

Measured on the same Apple M5 Max used for training (`onnxruntime` CPU
execution provider only). Well under any real-time budget for a per-frame
traffic-camera classifier (511GA samples about once per view per hour by
default, and even a live demo polling every few seconds has orders of
magnitude of headroom here).

## What's shipped

- `models/stage_a.onnx`, `models/stage_b.onnx`
- `models/config.json`: thresholds (`tA`, `tB`), input spec, backbone,
  variant, run IDs, class names, and the status logic from PLAN.md section 3.

See `docs/phase_reports/phase3_modeling.md` for the full runs table,
threshold selection, and the Stage B variant decision.
