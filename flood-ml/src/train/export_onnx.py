# keras -> onnx with prob + in-graph cam outputs
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

INPUT_NAME = "image"
OUTPUT_PROB = "prob"
OUTPUT_CAM = "cam"


def build_export_fn(keras_model, img_size: int):
    import tensorflow as tf

    from train.model import DENSE_LAYER_NAME, FEATURE_LAYER_NAME

    backbone = keras_model.get_layer(FEATURE_LAYER_NAME)
    dense = keras_model.get_layer(DENSE_LAYER_NAME)
    # cam = relu(sum_k w_k * A_k) with the dense weights, so no gradients at serve time
    kernel, _bias = dense.get_weights()
    kernel_np = kernel[:, 0].astype("float32")

    input_signature = [tf.TensorSpec([None, img_size, img_size, 3], tf.float32, name=INPUT_NAME)]

    @tf.function(input_signature=input_signature)
    def export_fn(image):
        prob = keras_model(image, training=False)
        conv = backbone(image, training=False)
        w = tf.constant(kernel_np)
        cam = tf.einsum("nhwc,c->nhw", conv, w)
        cam = tf.nn.relu(cam)
        return {OUTPUT_PROB: prob, OUTPUT_CAM: cam}

    return export_fn, input_signature


def export_stage_to_onnx(
    run_id: str,
    out_path: str | Path,
    *,
    img_size: int = 224,
    models_root: str = "models",
    opset: int = 17,
    convert_log: str | None = None,
):
    import keras
    import tf2onnx

    ckpt_path = Path(models_root) / run_id / "model.keras"
    keras_model = keras.models.load_model(ckpt_path)
    export_fn, input_signature = build_export_fn(keras_model, img_size)

    tf2onnx_logger = logging.getLogger("tf2onnx")
    handlers_added = []
    if convert_log:
        Path(convert_log).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(convert_log)
        tf2onnx_logger.addHandler(fh)
        handlers_added.append(fh)
    try:
        model_proto, _ = tf2onnx.convert.from_function(
            export_fn, input_signature=input_signature, opset=opset, output_path=None,
        )
    finally:
        for fh in handlers_added:
            tf2onnx_logger.removeHandler(fh)
            fh.close()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(model_proto.SerializeToString())
    logger.info("wrote %s (%d bytes)", out_path, out_path.stat().st_size)
    return keras_model, out_path


def _load_val_images(rows, img_size: int, mode: str = "crop") -> np.ndarray:
    from inference.preprocess import preprocess as infer_preprocess

    imgs = np.zeros((len(rows), img_size, img_size, 3), dtype=np.float32)
    for i, path in enumerate(rows["path"].to_numpy()):
        arr, _geom = infer_preprocess(path, mode=mode, size=img_size, do_jpeg_roundtrip=False)
        imgs[i] = arr
    return imgs


def verify_export(
    keras_model,
    onnx_path: str | Path,
    val_rows,
    *,
    img_size: int = 224,
    mode: str = "crop",
    n_images: int = 100,
    seed: int = 0,
) -> dict:
    import onnxruntime as ort

    from train.gradcam import gradcam_batch, normalize_cam
    from train.model import FEATURE_LAYER_NAME, BuiltModel, make_grad_model

    rng = np.random.default_rng(seed)
    n = min(n_images, len(val_rows))
    idx = rng.choice(len(val_rows), size=n, replace=False)
    sample = val_rows.iloc[idx].reset_index(drop=True)
    images = _load_val_images(sample, img_size, mode=mode)

    keras_prob = keras_model.predict(images, verbose=0).reshape(-1)

    backbone = keras_model.get_layer(FEATURE_LAYER_NAME)
    built = BuiltModel(model=keras_model, backbone=backbone, features=None,
                        backbone_name="loaded", input_size=img_size)
    grad_model = make_grad_model(built)
    keras_cam_raw, _ = gradcam_batch(grad_model, images)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    onnx_out = sess.run(None, {INPUT_NAME: images})
    onnx_prob = onnx_out[out_names.index(OUTPUT_PROB)].reshape(-1)
    onnx_cam_raw = onnx_out[out_names.index(OUTPUT_CAM)]

    max_prob_diff = float(np.max(np.abs(onnx_prob - keras_prob)))

    import cv2

    def _upsample(cam_raw):
        return np.stack([
            cv2.resize(cam_raw[i], (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            for i in range(n)
        ])

    onnx_cam_norm = normalize_cam(_upsample(onnx_cam_raw))
    keras_cam_norm = normalize_cam(_upsample(keras_cam_raw))

    a = onnx_cam_norm.reshape(-1)
    b = keras_cam_norm.reshape(-1)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(a, b)[0, 1])

    return {
        "n_images": n,
        "max_prob_diff": max_prob_diff,
        "cam_pearson_r": pearson_r,
    }


def benchmark_latency(onnx_path: str | Path, img_size: int, n_runs: int = 100, intra_op_threads: int | None = None) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    if intra_op_threads is not None:
        so.intra_op_num_threads = intra_op_threads
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    x = np.random.uniform(0, 255, size=(1, img_size, img_size, 3)).astype("float32")

    for _ in range(5):
        sess.run(None, {INPUT_NAME: x})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {INPUT_NAME: x})
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return {
        "median_ms": float(np.median(times) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "threads": intra_op_threads if intra_op_threads is not None else "default",
        "n_runs": n_runs,
    }


if __name__ == "__main__":
    import argparse

    from train.data import load_manifest, select_all_rows

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--mode", default="crop", help="input geometry: crop/squash/letterbox")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--n-verify", type=int, default=100)
    parser.add_argument("--convert-log", default="logs/jobs/tf2onnx_convert.log")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    keras_model, out_path = export_stage_to_onnx(
        args.run_id, args.out, img_size=args.img_size, models_root=args.models_root,
        convert_log=args.convert_log,
    )

    df = load_manifest(args.manifest)
    val_rows = select_all_rows(df, split="val")
    verification = verify_export(
        keras_model, out_path, val_rows, img_size=args.img_size, mode=args.mode, n_images=args.n_verify,
    )
    logger.info("verification: %s", json.dumps(verification, indent=2))

    latency_default = benchmark_latency(out_path, args.img_size, intra_op_threads=None)
    latency_1thread = benchmark_latency(out_path, args.img_size, intra_op_threads=1)
    logger.info("latency default threads: %s", json.dumps(latency_default, indent=2))
    logger.info("latency 1 thread: %s", json.dumps(latency_1thread, indent=2))

    report = {
        "run_id": args.run_id,
        "stage": args.stage,
        "onnx_path": str(out_path),
        "verification": verification,
        "latency_default_threads": latency_default,
        "latency_1_thread": latency_1thread,
    }
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

