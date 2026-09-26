# two-stage val eval + threshold picking
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from train.data import load_manifest, select_all_rows
from train.train import _youden_threshold

logger = logging.getLogger(__name__)

SMALL_SAMPLE_WARN_N = 30


def predict_probs(
    model, rows: pd.DataFrame, img_size: int, batch_size: int = 64, mode: str = "crop",
) -> np.ndarray:
    from train.data import make_dataset

    fake = rows.copy()
    fake["label_bin"] = 0.0
    fake["sample_weight"] = 1.0
    ds = make_dataset(
        fake, stage="a", variant=None, training=False, batch_size=batch_size, img_size=img_size, mode=mode,
    )
    probs = model.predict(ds, verbose=0).reshape(-1)
    return probs


@dataclass
class PipelineReport:
    tA: float
    tB: float
    tB_note: str
    n_val: int
    stage_a_auc_roc: float
    stage_a_auc_pr: float
    stage_a_accuracy: float
    stage_a_wet_recall: float
    stage_a_wet_recall_num: int
    stage_a_wet_recall_den: int
    pipeline_precision_flooded: float
    pipeline_recall_flooded: float
    pipeline_precision_num: int
    pipeline_precision_den: int
    pipeline_recall_num: int
    pipeline_recall_den: int
    false_alarm_dry_rate: float
    false_alarm_dry_num: int
    false_alarm_dry_den: int
    false_alarm_wet_rate: float
    false_alarm_wet_num: int
    false_alarm_wet_den: int
    false_alarm_not_flooded_rate: float
    false_alarm_not_flooded_num: int
    false_alarm_not_flooded_den: int
    small_sample_flags: list[str]
    confusion: dict[str, dict[str, int]]


def status_from_probs(pA: np.ndarray, pB: np.ndarray, tA: float, tB: float) -> np.ndarray:
    status = np.full(len(pA), "wet", dtype=object)
    status[pA < tA] = "dry"
    flooded_mask = (pA >= tA) & (pB >= tB)
    status[flooded_mask] = "flooded"
    return status


def _rate(numerator_mask: np.ndarray, denom_mask: np.ndarray) -> tuple[float, int, int]:
    den = int(np.sum(denom_mask))
    num = int(np.sum(numerator_mask & denom_mask))
    rate = (num / den) if den > 0 else float("nan")
    return rate, num, den


def sweep_tb_for_precision(
    true_label: np.ndarray, pA: np.ndarray, pB: np.ndarray, tA: float, target_precision: float = 0.90
) -> tuple[float, str]:
    candidates = np.unique(np.clip(pB, 0.0, 1.0))
    candidates = np.concatenate([candidates, [1.0]])
    candidates.sort()
    is_flooded_true = true_label == "flooded"

    best_reaching = None
    best_f05 = (-1.0, 0.5)
    for tb in candidates:
        status = status_from_probs(pA, pB, tA, tb)
        pred_flooded = status == "flooded"
        tp = int(np.sum(pred_flooded & is_flooded_true))
        fp = int(np.sum(pred_flooded & ~is_flooded_true))
        fn = int(np.sum(~pred_flooded & is_flooded_true))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta2 = 0.25
        denom = beta2 * precision + recall
        f05 = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
        if (tp + fp) > 0 and precision >= target_precision and best_reaching is None:
            best_reaching = float(tb)
        if f05 > best_f05[0]:
            best_f05 = (f05, float(tb))
    if best_reaching is not None:
        return best_reaching, f"lowest tB reaching pipeline precision>={target_precision}"
    return best_f05[1], f"pipeline precision>={target_precision} unreachable on val; maximized pipeline F0.5 instead"


def evaluate_pipeline(
    val_rows: pd.DataFrame, pA: np.ndarray, pB: np.ndarray, target_precision: float = 0.90,
    tA: float | None = None, tB: float | None = None,
) -> PipelineReport:
    from sklearn.metrics import average_precision_score, roc_auc_score

    true_label = val_rows["label"].to_numpy()
    is_dry_true = true_label == "dry"
    is_wet_true = true_label == "wet"
    is_flooded_true = true_label == "flooded"
    is_not_flooded_true = true_label == "not_flooded"
    stage_a_mask = ~is_not_flooded_true

    stage_a_true_full = (~is_dry_true).astype(int)
    stage_a_true = stage_a_true_full[stage_a_mask]
    pA_stage_a = pA[stage_a_mask]
    if tA is None:
        tA = _youden_threshold(stage_a_true, pA_stage_a)
    stage_a_auc_roc = float(roc_auc_score(stage_a_true, pA_stage_a))
    stage_a_auc_pr = float(average_precision_score(stage_a_true, pA_stage_a))
    stage_a_pred = (pA_stage_a >= tA).astype(int)
    stage_a_accuracy = float(np.mean(stage_a_pred == stage_a_true))
    wet_recall_rate, wet_recall_num, wet_recall_den = _rate(pA >= tA, is_wet_true)

    if tB is None:
        tB, tB_note = sweep_tb_for_precision(true_label, pA, pB, tA, target_precision=target_precision)
    else:
        tB_note = "fixed threshold, not tuned on this val"
    status = status_from_probs(pA, pB, tA, tB)
    pred_flooded = status == "flooded"

    precision_rate, precision_num, precision_den = _rate(is_flooded_true, pred_flooded)
    recall_rate, recall_num, recall_den = _rate(pred_flooded, is_flooded_true)

    far_dry_rate, far_dry_num, far_dry_den = _rate(pred_flooded, is_dry_true)
    far_wet_rate, far_wet_num, far_wet_den = _rate(pred_flooded, is_wet_true)
    far_nf_rate, far_nf_num, far_nf_den = _rate(pred_flooded, is_not_flooded_true)

    confusion: dict[str, dict[str, int]] = {}
    for true_l in ("dry", "wet", "flooded"):
        row_mask = true_label == true_l
        confusion[true_l] = {
            pred_l: int(np.sum((status == pred_l) & row_mask)) for pred_l in ("dry", "wet", "flooded")
        }

    flags = []
    for name, den in (
        ("pipeline_precision_flooded", precision_den),
        ("pipeline_recall_flooded", recall_den),
        ("false_alarm_dry_rate", far_dry_den),
        ("false_alarm_wet_rate", far_wet_den),
        ("false_alarm_not_flooded_rate", far_nf_den),
        ("stage_a_wet_recall", wet_recall_den),
    ):
        if den < SMALL_SAMPLE_WARN_N:
            flags.append(f"{name} rests on only {den} val examples (<{SMALL_SAMPLE_WARN_N})")

    return PipelineReport(
        tA=tA, tB=tB, tB_note=tB_note, n_val=len(val_rows),
        stage_a_auc_roc=stage_a_auc_roc, stage_a_auc_pr=stage_a_auc_pr, stage_a_accuracy=stage_a_accuracy,
        stage_a_wet_recall=wet_recall_rate, stage_a_wet_recall_num=wet_recall_num,
        stage_a_wet_recall_den=wet_recall_den,
        pipeline_precision_flooded=precision_rate, pipeline_recall_flooded=recall_rate,
        pipeline_precision_num=precision_num, pipeline_precision_den=precision_den,
        pipeline_recall_num=recall_num, pipeline_recall_den=recall_den,
        false_alarm_dry_rate=far_dry_rate, false_alarm_dry_num=far_dry_num, false_alarm_dry_den=far_dry_den,
        false_alarm_wet_rate=far_wet_rate, false_alarm_wet_num=far_wet_num, false_alarm_wet_den=far_wet_den,
        false_alarm_not_flooded_rate=far_nf_rate, false_alarm_not_flooded_num=far_nf_num,
        false_alarm_not_flooded_den=far_nf_den,
        small_sample_flags=flags, confusion=confusion,
    )


def per_source_breakdown(
    val_rows: pd.DataFrame, pA: np.ndarray, pB: np.ndarray, tA: float, tB: float,
) -> dict[str, dict]:
    true_label = val_rows["label"].to_numpy()
    source = val_rows["source"].to_numpy()
    status = status_from_probs(pA, pB, tA, tB)
    pred_flooded = status == "flooded"

    out: dict[str, dict] = {}
    for src in sorted(set(source)):
        m = source == src
        is_flooded_true = (true_label == "flooded") & m
        is_dry_true = (true_label == "dry") & m
        is_wet_true = (true_label == "wet") & m
        is_nf_true = (true_label == "not_flooded") & m

        precision_rate, precision_num, precision_den = _rate(is_flooded_true, pred_flooded & m)
        recall_rate, recall_num, recall_den = _rate(pred_flooded, is_flooded_true)
        far_dry_rate, far_dry_num, far_dry_den = _rate(pred_flooded, is_dry_true)
        far_wet_rate, far_wet_num, far_wet_den = _rate(pred_flooded, is_wet_true)
        far_nf_rate, far_nf_num, far_nf_den = _rate(pred_flooded, is_nf_true)

        out[src] = {
            "n": int(m.sum()),
            "pipeline_precision_flooded": precision_rate,
            "pipeline_precision_num": precision_num, "pipeline_precision_den": precision_den,
            "pipeline_recall_flooded": recall_rate,
            "pipeline_recall_num": recall_num, "pipeline_recall_den": recall_den,
            "false_alarm_dry_rate": far_dry_rate, "false_alarm_dry_num": far_dry_num, "false_alarm_dry_den": far_dry_den,
            "false_alarm_wet_rate": far_wet_rate, "false_alarm_wet_num": far_wet_num, "false_alarm_wet_den": far_wet_den,
            "false_alarm_not_flooded_rate": far_nf_rate, "false_alarm_not_flooded_num": far_nf_num,
            "false_alarm_not_flooded_den": far_nf_den,
        }
    return out


def load_checkpoint(run_id: str, models_root: str = "models"):
    import keras

    path = Path(models_root) / run_id / "model.keras"
    return keras.models.load_model(path)


def variant_selection_key(precision: float, recall: float, target_precision: float = 0.90) -> tuple:
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    meets_target = precision >= target_precision
    return (meets_target, f1 if meets_target else precision)


def run_pipeline_selection(
    stage_a_run: str,
    stage_b_runs: dict[str, str],
    *,
    manifest: str = "data/processed/manifest.csv",
    img_size: int = 224,
    mode: str = "crop",
    models_root: str = "models",
    target_precision: float = 0.90,
) -> dict:
    df = load_manifest(manifest)
    val_rows = select_all_rows(df, split="val")

    stage_a_model = load_checkpoint(stage_a_run, models_root)
    pA = predict_probs(stage_a_model, val_rows, img_size, mode=mode)

    reports = {}
    for variant, run_id in stage_b_runs.items():
        stage_b_model = load_checkpoint(run_id, models_root)
        pB = predict_probs(stage_b_model, val_rows, img_size, mode=mode)
        report = evaluate_pipeline(val_rows, pA, pB, target_precision=target_precision)
        reports[variant] = {"run_id": run_id, "report": asdict(report)}
        logger.info("variant=%s report=%s", variant, json.dumps(asdict(report), indent=2))

    def _key(item):
        r = item[1]["report"]
        return variant_selection_key(
            r["pipeline_precision_flooded"], r["pipeline_recall_flooded"], target_precision
        )

    chosen_variant = max(reports.items(), key=_key)[0]

    return {
        "stage_a_run": stage_a_run,
        "variants": reports,
        "chosen_variant": chosen_variant,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-run", required=True)
    parser.add_argument("--stage-b-spec-run", default=None)
    parser.add_argument("--stage-b-mixed-run", default=None)
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--mode", default="crop", help="input geometry: crop/squash/letterbox")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--out-json", default="reports/pipeline_selection.json")
    args = parser.parse_args()

    stage_b_runs = {}
    if args.stage_b_spec_run:
        stage_b_runs["spec"] = args.stage_b_spec_run
    if args.stage_b_mixed_run:
        stage_b_runs["mixed"] = args.stage_b_mixed_run
    if not stage_b_runs:
        raise SystemExit("pass at least one of --stage-b-spec-run / --stage-b-mixed-run")

    result = run_pipeline_selection(
        args.stage_a_run, stage_b_runs, manifest=args.manifest, img_size=args.img_size, mode=args.mode,
        models_root=args.models_root, target_precision=args.target_precision,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

