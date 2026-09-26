"""Bounded water-only training. Run from flood-ml with the existing Python."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from train.water_data import ROOT, annotated_rows, capped_rows, counts, load_arrays, read_manifest
from train.water_metrics import canvas_metrics

LOG = logging.getLogger(__name__)


def log_shared_run(report, evaluation, date=None):
    from train.train import append_run_row

    args = report["args"]
    metrics = evaluation["metrics"]
    notes = {source: {key: metrics[source][key] for key in ("iou", "dice")}
             for source in report["val_counts"]}
    append_run_row({
        "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "run_id": report["run_id"], "stage": "water_segmentation",
        "variant": "full_frame_letterbox", "model": "mobilenetv3small_skip_decoder",
        "data_version": "v1-" + report["config"]["manifest_sha256"][:8],
        "img_size": args["size"], "epochs_head": len(report["history"]), "epochs_ft": 0,
        "n_train": sum(report["train_counts"].values()),
        "n_val": sum(report["val_counts"].values()),
        "threshold": report["config"]["threshold"],
        "train_time_s": round(report["elapsed_seconds"], 1),
        "logdir": f"logs/{report['run_id']}/tensorboard",
        "notes": "validation pixel metrics (selected on val); " + json.dumps(notes, sort_keys=True),
    }, ROOT / "reports/runs.csv")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict_batches(model, images, batch_size):
    return np.concatenate([model(images[i:i + batch_size].astype(np.float32), training=False).numpy()
                           for i in range(0, len(images), batch_size)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--per-source", type=int, default=192)
    parser.add_argument("--per-group", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=int, default=420)
    parser.add_argument("--weights", default=str(
        Path.home() / ".keras/models/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5"))
    args = parser.parse_args()
    if not args.run_id.startswith("water_") or Path(args.run_id).name != args.run_id:
        parser.error("run-id must be a single water_* directory name")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "ml":
        raise RuntimeError("Water work is restricted to branch ml")
    model_dir = ROOT / "models/water"
    log_dir = ROOT / "logs" / args.run_id
    report_dir = ROOT / "reports" / args.run_id
    checkpoint_dir = model_dir / args.run_id
    if (model_dir / "config.json").exists():
        raise RuntimeError("A water artifact already exists; preserve frozen models for review")
    for directory in (log_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / "training.log", level=logging.INFO,
                        format="%(asctime)s %(message)s", force=True)
    started = time.monotonic()
    rows = read_manifest(ROOT / args.manifest)
    train = capped_rows(annotated_rows(rows, "train"), args.per_source, args.per_group, args.seed)
    val = annotated_rows(rows, "val")
    if set(counts(train)) != set(counts(val)):
        raise ValueError("Each training source must have annotated validation data")
    (report_dir / "selected_rows.json").write_text(json.dumps(
        {"train": train, "val": val}, indent=2))
    LOG.info("Loading train=%s val=%s", counts(train), counts(val))
    x, y = load_arrays(train, args.size)
    vx, vy = load_arrays(val, args.size)
    import keras
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    from train.water_model import build_water_model, masked_loss

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    model = build_water_model(args.size, args.weights)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    writer = tf.summary.create_file_writer(str(log_dir / "tensorboard"))
    rng = np.random.default_rng(args.seed)

    @tf.function(reduce_retracing=True)
    def step(pixels, target):
        with tf.GradientTape() as tape:
            probability = model(pixels, training=True)
            loss = masked_loss(target, probability)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    history, best_score, best_epoch, stale = [], -1.0, 0, 0
    training_start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        losses = []
        for offset in range(0, len(x), args.batch_size):
            if offset == 0:
                order = rng.permutation(len(x))
            idx = order[offset:offset + args.batch_size]
            bx, by = x[idx].astype(np.float32), y[idx].copy()
            flip = rng.random(len(idx)) < 0.5
            bx[flip], by[flip] = bx[flip, :, ::-1], by[flip, :, ::-1]
            losses.append(float(step(bx, by)))
        probabilities = predict_batches(model, vx, args.batch_size)
        metrics = canvas_metrics(probabilities, vy, val, 0.5)
        score = float(np.mean([metrics[source]["iou"] for source in counts(val)]))
        record = {"epoch": epoch, "loss": float(np.mean(losses)),
                  "source_macro_iou": score, "metrics": metrics,
                  "seconds": time.monotonic() - epoch_start}
        history.append(record)
        with writer.as_default():
            tf.summary.scalar("loss/train", record["loss"], step=epoch)
            tf.summary.scalar("val/source_macro_iou", score, step=epoch)
            for source, metric in metrics.items():
                tf.summary.scalar(f"val/{source}/mask_iou", metric["iou"], step=epoch)
                tf.summary.scalar(f"val/{source}/mask_dice", metric["dice"], step=epoch)
        writer.flush()
        LOG.info("Epoch %s loss=%.4f macro_iou=%.4f seconds=%.1f",
                     epoch, record["loss"], score, record["seconds"])
        (report_dir / "history.json").write_text(json.dumps(history, indent=2))
        if score > best_score + 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            model.save(checkpoint_dir / "best.keras")
        else:
            stale += 1
        if stale >= 3 or time.monotonic() - training_start >= args.max_seconds:
            break
    writer.close()
    model = keras.models.load_model(checkpoint_dir / "best.keras", compile=False)
    probabilities = predict_batches(model, vx, args.batch_size)
    threshold_results = {}
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        metrics = canvas_metrics(probabilities, vy, val, threshold)
        threshold_results[str(threshold)] = {
            "source_macro_iou": float(np.mean([metrics[s]["iou"] for s in counts(val)])),
            "metrics": metrics,
        }
    threshold = float(max(threshold_results,
                          key=lambda t: (threshold_results[t]["source_macro_iou"],
                                         -abs(float(t) - 0.5))))
    signature = [tf.TensorSpec([None, args.size, args.size, 3], tf.float32, name="image")]

    @tf.function(input_signature=signature)
    def export(image):
        return {"water_prob": model(image, training=False)}

    onnx_path = checkpoint_dir / "water.onnx"
    tf2onnx.convert.from_function(export, input_signature=signature, opset=17,
                                  output_path=str(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    max_error, errors_sum, n_pixels, disagreements = 0., 0., 0, 0
    timings = []
    for i in range(0, len(vx), args.batch_size):
        bx = vx[i:i + args.batch_size].astype(np.float32)
        tick = time.monotonic()
        actual = session.run(["water_prob"], {"image": bx})[0]
        timings.append((time.monotonic() - tick) / len(bx))
        expected = probabilities[i:i + len(bx)]
        error = np.abs(actual - expected)
        max_error = max(max_error, float(error.max()))
        errors_sum += float(error.sum())
        n_pixels += error.size
        valid = vy[i:i + len(bx), ..., 1:] > 0
        disagreements += int(np.count_nonzero(((actual >= threshold) != (expected >= threshold)) & valid))
    parity = {"split": "val", "n_images": len(vx), "max_abs_error": max_error,
              "mean_abs_error": errors_sum / n_pixels,
              "threshold_disagreement_valid_pixels": disagreements,
              "mean_batch_ms_per_image": float(np.mean(timings) * 1000)}
    if max_error > 1e-4:
        raise RuntimeError(f"ONNX parity failed: {parity}")
    config = {
        "task": "water_segmentation", "model_version": args.run_id,
        "input_size": args.size, "threshold": threshold, "frozen": True,
        "road": False, "geometry": "full_frame_letterbox",
        "output": "water_prob", "onnx_sha256": digest(onnx_path),
        "manifest_sha256": digest(ROOT / args.manifest),
        "pretrained_sha256": digest(args.weights), "best_epoch": best_epoch,
        "test_evaluated": False,
        "note": "Water segmentation only; road=false. Not flooded-roadway extent or classifier CAM.",
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(config, indent=2))
    # Publish only after the selected weights/threshold have passed validation parity.
    (model_dir / "water.onnx").write_bytes(onnx_path.read_bytes())
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))
    report = {
        "run_id": args.run_id, "args": vars(args), "train_counts": counts(train),
        "val_counts": counts(val), "eligible_train_counts": counts(annotated_rows(rows, "train")),
        "parameters": model.count_params(),
        "trainable_parameters": int(sum(np.prod(v.shape) for v in model.trainable_variables)),
        "best_epoch": best_epoch, "history": history, "threshold_selection": threshold_results,
        "config": config, "parity": parity, "elapsed_seconds": time.monotonic() - started,
        "test_evaluated": False,
    }
    (report_dir / "training_report.json").write_text(json.dumps(report, indent=2))
    from train.water_eval import evaluate
    evaluation = evaluate(model_dir, ROOT / args.manifest, "val")
    (report_dir / "validation.json").write_text(json.dumps(evaluation, indent=2))
    row = {
        "run_id": args.run_id, "task": "water_segmentation",
        "train_n": len(train), "val_n": len(val), "size": args.size,
        "epochs": len(history), "best_epoch": best_epoch, "threshold": threshold,
        "fred_iou": evaluation["metrics"]["fred"]["iou"],
        "fred_dice": evaluation["metrics"]["fred"]["dice"],
        "roadway_iou": evaluation["metrics"]["roadway_flooding"]["iou"],
        "roadway_dice": evaluation["metrics"]["roadway_flooding"]["dice"],
        "test_evaluated": False,
    }
    csv_path = ROOT / "reports/water_runs.csv"
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            csv_writer.writeheader()
        csv_writer.writerow(row)
    log_shared_run(report, evaluation)
    LOG.info("FROZEN %s", json.dumps(row))
    print(json.dumps({"ready": True, **row, "parity": parity}, indent=2))


if __name__ == "__main__":
    main()
