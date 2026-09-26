"""Grad-CAM for the Stage A/B classifiers built in `model.py`.

Two implementations that should agree up to a positive per-image scale
(see `model.py`'s docstring for why):

- `gradcam_batch`: real GradientTape Grad-CAM. Backprops the sigmoid output
  through to the last conv feature map, averages gradients spatially per
  channel, and applies ReLU. This is the reference / ground truth used in
  tests and to verify the ONNX export.
- `cam_from_dense_weights`: no gradients at all -- just `ReLU(sum_k w_k *
  A_k)` using the Dense(1, sigmoid) head's kernel directly. This is what
  `export_onnx.py` bakes into the ONNX graph, since ONNX Runtime has no
  autodiff at inference time.

Both take/return numpy arrays so callers (train.py, export_onnx.py, tests)
don't need to juggle tf tensors directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def gradcam_batch(grad_model, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`images`: float32 array (N,H,W,3), pixel values in [0,255] (raw,
    un-preprocessed -- preprocessing is built into the model). Returns
    `(cam, prob)`: `cam` is (N,h,w) float32, ReLU'd but NOT normalized;
    `prob` is (N,1) float32, the sigmoid output.
    """
    import tensorflow as tf

    images_t = tf.convert_to_tensor(images, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(images_t)
        conv_out, prob = grad_model(images_t, training=False)
        target = prob[:, 0]
    grads = tape.gradient(target, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # (N, C)
    cam = tf.einsum("nhwc,nc->nhw", conv_out, weights)
    cam = tf.nn.relu(cam)
    return cam.numpy(), prob.numpy()


def cam_from_dense_weights(conv_out: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """`conv_out`: (N,h,w,C) feature map (numpy or tf tensor). `kernel`:
    (C, 1) Dense kernel (as returned by `model.dense_kernel`). Returns (N,h,w)
    float32, ReLU'd, not normalized -- the same convention as
    `gradcam_batch`'s `cam` output.
    """
    import tensorflow as tf

    conv_out = tf.convert_to_tensor(conv_out, dtype=tf.float32)
    w = tf.constant(np.asarray(kernel)[:, 0], dtype=tf.float32)
    cam = tf.einsum("nhwc,c->nhw", conv_out, w)
    return tf.nn.relu(cam).numpy()


def normalize_cam(cam: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-image min-max normalize (min is usually 0 after ReLU) to [0,1].
    Cancels the positive scalar difference between `gradcam_batch` and
    `cam_from_dense_weights`, so normalized CAMs from the two methods should
    be (near-)identical.
    """
    cam = np.asarray(cam, dtype=np.float32)
    flat = cam.reshape(cam.shape[0], -1)
    lo = flat.min(axis=1, keepdims=True)
    hi = flat.max(axis=1, keepdims=True)
    denom = np.maximum(hi - lo, eps)
    norm = (flat - lo) / denom
    return norm.reshape(cam.shape)


def overlay_heatmap(image_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """`image_uint8`: (H,W,3) RGB uint8. `cam`: (h,w) float32 in [0,1] (already
    normalized). Returns an (H,W,3) RGB uint8 overlay: `cam` is resized to the
    image's resolution with bilinear interpolation, colored with OpenCV's
    JET colormap, and alpha-blended over the image.
    """
    import cv2

    h, w = image_uint8.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    cam_u8 = np.clip(cam_resized * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = (image_uint8.astype(np.float32) * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_gradcam_gallery(
    built,
    rows,
    out_dir: Path | str,
    *,
    n_per_label: int = 8,
    seed: int = 0,
    repo_root: str = "",
) -> list[str]:
    """Picks a seeded sample of up to `n_per_label` val rows per label,
    runs GradientTape Grad-CAM, and writes overlay PNGs to `out_dir`
    (reports/gradcam_val/, local-only per PLAN.md -- never committed).
    Returns the list of written file paths (as strings).
    """
    import cv2
    from PIL import Image

    from train.model import make_grad_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grad_model = make_grad_model(built)

    rng = np.random.default_rng(seed)
    written = []
    for label, group in rows.groupby("label"):
        idx = rng.permutation(len(group))[:n_per_label]
        picked = group.iloc[idx]
        for _, row in picked.iterrows():
            path = row["path"]
            full_path = str(Path(repo_root) / path) if repo_root else path
            img = Image.open(full_path).convert("RGB")
            img = img.resize((built.input_size, built.input_size), Image.BILINEAR)
            img_arr = np.array(img, dtype=np.uint8)
            batch = img_arr[np.newaxis].astype(np.float32)
            cam, prob = gradcam_batch(grad_model, batch)
            cam_norm = normalize_cam(cam)[0]
            overlay = overlay_heatmap(img_arr, cam_norm)
            stem = Path(path).stem
            out_path = out_dir / f"{label}_{stem}_p{prob[0, 0]:.2f}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            written.append(str(out_path))
    return written
