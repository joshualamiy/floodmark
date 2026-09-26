# train stage a or b, log to tensorboard + reports/runs.csv
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train.data import (
    MODE_CROP,
    MODES,
    STAGE_A,
    STAGE_B,
    VARIANT_SPEC,
    VARIANTS,
    benchmark_throughput,
    compute_sample_weights,
    load_manifest,
    make_dataset,
    select_stage_rows,
)

logger = logging.getLogger(__name__)

RUNS_CSV_COLUMNS = [
    "date", "run_id", "stage", "variant", "model", "data_version", "img_size",
    "epochs_head", "epochs_ft", "n_train", "n_val", "val_auc_roc", "val_auc_pr",
    "val_precision", "val_recall", "val_f1", "threshold", "val_false_alarm_rate",
    "train_time_s", "logdir", "notes",
]

DATA_VERSION_PATH = Path("data/processed/VERSION")


def read_data_version(path: Path = DATA_VERSION_PATH) -> str:
    text = path.read_text().strip()
    return text.split("=", 1)[1].strip() if "=" in text else text


def set_seeds(seed: int) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_run_id(stage: str, variant: str | None, backbone: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d-%H%M%S")
    tag = f"{stage}{variant}" if variant else stage
    return f"{ts}_{tag}_{backbone}"


@dataclass
class ValMetrics:
    auc_roc: float
    auc_pr: float
    precision: float
    recall: float
    f1: float
    threshold: float
    false_alarm_rate: float
    false_alarm_num: int
    false_alarm_den: int
    threshold_note: str


def _youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best = int(np.argmax(j))
    thr = float(thresholds[best])
    return float(np.clip(thr, 0.0, 1.0))


def _precision_target_threshold(
    y_true: np.ndarray, scores: np.ndarray, target_precision: float = 0.90
) -> tuple[float, str]:
    candidates = np.unique(np.clip(scores, 0.0, 1.0))
    candidates = np.concatenate([candidates, [1.0]])
    candidates.sort()
    best_reaching = None
    best_f05 = (-1.0, 0.5)
    for thr in candidates:
        pred = scores >= thr
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta2 = 0.25
        denom = beta2 * precision + recall
        f05 = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
        if (tp + fp) > 0 and precision >= target_precision and best_reaching is None:
            best_reaching = float(thr)
        if f05 > best_f05[0]:
            best_f05 = (f05, float(thr))
    if best_reaching is not None:
        return best_reaching, f"lowest threshold reaching precision>={target_precision}"
    return best_f05[1], f"precision>={target_precision} unreachable on val; maximized F0.5 instead"


def evaluate_val(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    stage: str,
    false_alarm_label: np.ndarray,
) -> ValMetrics:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).reshape(-1)

    auc_roc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")
    auc_pr = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")

    if stage == STAGE_A:
        threshold = _youden_threshold(y_true, probs)
        threshold_note = "Youden J on Stage A val ROC"
    else:
        threshold, threshold_note = _precision_target_threshold(y_true, probs, target_precision=0.90)

    pred = (probs >= threshold).astype(int)
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))

    den = int(np.sum(false_alarm_label))
    num = int(np.sum(pred[false_alarm_label] == 1)) if den > 0 else 0
    far = (num / den) if den > 0 else float("nan")

    return ValMetrics(
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=threshold,
        false_alarm_rate=far,
        false_alarm_num=num,
        false_alarm_den=den,
        threshold_note=threshold_note,
    )


def append_run_row(row: dict, runs_csv: Path) -> None:
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not runs_csv.exists()
    with open(runs_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RUNS_CSV_COLUMNS})


def build_callbacks(logdir: Path, monitor: str, patience: int):
    import keras

    logdir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.TensorBoard(log_dir=str(logdir)),
        keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=patience, restore_best_weights=True
        ),
    ]


def run_training(args: argparse.Namespace) -> dict:
    import keras

    from train.model import build_model, set_backbone_trainable

    set_seeds(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = load_manifest(args.manifest)
    train_rows = select_stage_rows(df, args.stage, args.variant, split="train")
    val_rows = select_stage_rows(df, args.stage, args.variant, split="val")
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise RuntimeError(f"no rows for stage={args.stage} variant={args.variant} (train={len(train_rows)}, val={len(val_rows)})")

    train_rows["sample_weight"] = compute_sample_weights(
        train_rows, args.stage, args.variant, wet_upweight=args.wet_upweight
    )
    logger.info(
        "stage=%s variant=%s n_train=%d n_val=%d train_label_counts=%s",
        args.stage, args.variant, len(train_rows), len(val_rows),
        train_rows["label"].value_counts().to_dict(),
    )

    train_ds = make_dataset(
        train_rows, stage=args.stage, variant=args.variant, training=True,
        batch_size=args.batch_size, img_size=args.img_size, mode=args.mode, seed=args.seed,
        night_aug=not args.no_night_aug,
    )
    val_rows_for_ds = val_rows.copy()
    val_rows_for_ds["sample_weight"] = 1.0
    val_ds = make_dataset(
        val_rows_for_ds, stage=args.stage, variant=args.variant, training=False,
        batch_size=args.batch_size, img_size=args.img_size, mode=args.mode,
    )

    if args.benchmark:
        bench = benchmark_throughput(train_ds, n_batches=args.benchmark_batches)
        logger.info("train pipeline throughput: %.1f img/s (batch=%d, n=%d, %.2fs)",
                    bench.images_per_sec, bench.batch_size, bench.n_images, bench.seconds)

    run_id = args.run_id or make_run_id(args.stage, args.variant, args.backbone)
    logdir_root = Path(args.logdir_root) / run_id
    model_dir = Path(args.models_root) / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    built = build_model(args.backbone, input_size=args.img_size, dropout=args.dropout, weights=args.weights)
    metrics = [
        keras.metrics.AUC(curve="PR", name="auc_pr"),
        keras.metrics.AUC(curve="ROC", name="auc_roc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]

    start = time.perf_counter()

    set_backbone_trainable(built, 0)
    built.model.compile(optimizer=keras.optimizers.Adam(args.lr_head), loss="binary_crossentropy", metrics=metrics)
    callbacks = build_callbacks(logdir_root / "head", args.monitor, args.patience)
    history_head = built.model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs_head, callbacks=callbacks, verbose=args.verbose,
    )
    epochs_head_ran = len(history_head.history.get("loss", []))

    epochs_ft_ran = 0
    if args.epochs_ft > 0:
        set_backbone_trainable(built, args.ft_layers)
        built.model.compile(optimizer=keras.optimizers.Adam(args.lr_ft), loss="binary_crossentropy", metrics=metrics)
        callbacks = build_callbacks(logdir_root / "ft", args.monitor, args.patience)
        history_ft = built.model.fit(
            train_ds, validation_data=val_ds, epochs=args.epochs_ft, callbacks=callbacks, verbose=args.verbose,
        )
        epochs_ft_ran = len(history_ft.history.get("loss", []))

    train_time_s = time.perf_counter() - start

    val_probs = built.model.predict(val_ds, verbose=0).reshape(-1)
    val_labels = val_rows["label_bin"].to_numpy().astype(int)
    if args.stage == STAGE_A:
        false_alarm_mask = val_rows["label"].to_numpy() == "dry"
    else:
        false_alarm_mask = val_rows["label"].to_numpy() == "wet"

    val_metrics = evaluate_val(val_labels, val_probs, stage=args.stage, false_alarm_label=false_alarm_mask)

    ckpt_path = model_dir / "model.keras"
    built.model.save(ckpt_path)
    logger.info("saved checkpoint to %s", ckpt_path)

    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "run_id": run_id,
        "stage": args.stage,
        "variant": args.variant or "",
        "model": args.backbone,
        "data_version": read_data_version(),
        "img_size": args.img_size,
        "epochs_head": epochs_head_ran,
        "epochs_ft": epochs_ft_ran,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "val_auc_roc": f"{val_metrics.auc_roc:.4f}",
        "val_auc_pr": f"{val_metrics.auc_pr:.4f}",
        "val_precision": f"{val_metrics.precision:.4f}",
        "val_recall": f"{val_metrics.recall:.4f}",
        "val_f1": f"{val_metrics.f1:.4f}",
        "threshold": f"{val_metrics.threshold:.4f}",
        "val_false_alarm_rate": (
            f"{val_metrics.false_alarm_rate:.4f} ({val_metrics.false_alarm_num}/{val_metrics.false_alarm_den})"
            if val_metrics.false_alarm_den > 0 else "n/a (0 denom)"
        ),
        "train_time_s": f"{train_time_s:.1f}",
        "logdir": str(logdir_root),
        "notes": (
            f"mode={args.mode}; night_aug={not args.no_night_aug}; "
            f"{val_metrics.threshold_note}; {args.notes}"
        ).strip("; "),
    }
    append_run_row(row, Path(args.runs_csv))
    logger.info("run complete: %s", json.dumps(row, indent=2))

    return {
        "run_id": run_id,
        "model_dir": str(model_dir),
        "row": row,
        "val_metrics": asdict(val_metrics),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=(STAGE_A, STAGE_B), required=True)
    p.add_argument("--variant", choices=VARIANTS, default=None, help="required for stage b")
    p.add_argument("--backbone", choices=("mobilenetv3small", "efficientnetb0"), default="mobilenetv3small")
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    p.add_argument("--epochs-head", type=int, default=8)
    p.add_argument("--epochs-ft", type=int, default=0)
    p.add_argument("--ft-layers", type=int, default=30, help="number of top backbone layers to unfreeze in phase 2")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-ft", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--mode", choices=MODES, default=MODE_CROP, help="input geometry: crop/squash/letterbox")
    p.add_argument("--no-night-aug", action="store_true", help="disable the dark/glare augmentation op")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--wet-upweight", type=float, default=8.0)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--monitor", default="val_auc_pr")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest", default="data/processed/manifest.csv")
    p.add_argument("--runs-csv", default="reports/runs.csv")
    p.add_argument("--logdir-root", default="logs")
    p.add_argument("--models-root", default="models")
    p.add_argument("--run-id", default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--benchmark", action="store_true", help="benchmark the train pipeline's throughput before training")
    p.add_argument("--benchmark-batches", type=int, default=20)
    p.add_argument("--verbose", type=int, default=2)
    return p


def parse_args(argv=None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    if args.weights.lower() in ("none", "null", ""):
        args.weights = None
    if args.stage == STAGE_B and args.variant is None:
        args.variant = VARIANT_SPEC
    return args


if __name__ == "__main__":
    run_training(parse_args())

