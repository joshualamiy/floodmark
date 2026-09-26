"""Pipeline-level validation: combines a trained Stage A checkpoint with a
trained Stage B checkpoint (either variant -- "spec" wet-vs-flooded, or
"mixed" flooded-vs-not-flooded) and scores PLAN.md section 3's status logic
on ALL val rows (never test; see PLAN.md rule 4 / the Phase 3 brief):

    dry if pA < tA; else flooded if pB >= tB; else wet

This is the level the brief asks us to pick the Stage B variant at, and the
level tB is actually tuned at (tA comes from Stage A's own ROC). Stage-level
metrics logged per-run in `train.py` are a cheaper proxy computed during
training and are NOT what selects the shipped variant or threshold.

Threshold rule:
    tA: Youden's J on the Stage A val ROC (balances Stage A's own errors).
    tB: swept holding tA fixed. The lowest tB where PIPELINE precision for
    status == "flooded" (vs. true label == "flooded") on val is >= 0.90.
    If unreachable, maximize pipeline F0.5 instead and say so.

Every number that rests on fewer than 30 val examples (the true-wet slice is
6) is flagged in the returned report, not just quietly computed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from train.data import load_manifest, select_stage_rows
from train.train import _youden_threshold  # reuse Stage A's own threshold rule

logger = logging.getLogger(__name__)

SMALL_SAMPLE_WARN_N = 30


def predict_probs(model, rows: pd.DataFrame, img_size: int, batch_size: int = 64) -> np.ndarray:
    """Runs `model` over `rows["path"]` in order (no shuffling), returning a
    1-D float32 array of probabilities aligned to `rows`'s row order.
    """
    from train.data import make_dataset

    fake = rows.copy()
    fake["label_bin"] = 0.0
    fake["sample_weight"] = 1.0
    ds = make_dataset(fake, stage="a", variant=None, training=False, batch_size=batch_size, img_size=img_size)
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
        # require at least one predicted positive: otherwise precision is
        # vacuously 1.0 (0/0) and would let a threshold that flags nothing
        # "reach" the target, which is meaningless (recall 0).
        if (tp + fp) > 0 and precision >= target_precision and best_reaching is None:
            best_reaching = float(tb)
        if f05 > best_f05[0]:
            best_f05 = (f05, float(tb))
    if best_reaching is not None:
        return best_reaching, f"lowest tB reaching pipeline precision>={target_precision}"
    return best_f05[1], f"pipeline precision>={target_precision} unreachable on val; maximized pipeline F0.5 instead"


def evaluate_pipeline(
    val_rows: pd.DataFrame, pA: np.ndarray, pB: np.ndarray, target_precision: float = 0.90
) -> PipelineReport:
    from sklearn.metrics import average_precision_score, roc_auc_score

    true_label = val_rows["label"].to_numpy()
    is_dry_true = true_label == "dry"
    is_wet_true = true_label == "wet"
    is_flooded_true = true_label == "flooded"

    stage_a_true = (~is_dry_true).astype(int)
    tA = _youden_threshold(stage_a_true, pA)
    stage_a_auc_roc = float(roc_auc_score(stage_a_true, pA))
    stage_a_auc_pr = float(average_precision_score(stage_a_true, pA))
    stage_a_pred = (pA >= tA).astype(int)
    stage_a_accuracy = float(np.mean(stage_a_pred == stage_a_true))

    tB, tB_note = sweep_tb_for_precision(true_label, pA, pB, tA, target_precision=target_precision)
    status = status_from_probs(pA, pB, tA, tB)
    pred_flooded = status == "flooded"

    precision_rate, precision_num, precision_den = _rate(is_flooded_true, pred_flooded)
    # recall: of true flooded, how many were caught (denominator = true flooded)
    recall_rate, recall_num, recall_den = _rate(pred_flooded, is_flooded_true)

    far_dry_rate, far_dry_num, far_dry_den = _rate(pred_flooded, is_dry_true)
    far_wet_rate, far_wet_num, far_wet_den = _rate(pred_flooded, is_wet_true)

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
    ):
        if den < SMALL_SAMPLE_WARN_N:
            flags.append(f"{name} rests on only {den} val examples (<{SMALL_SAMPLE_WARN_N})")

    return PipelineReport(
        tA=tA, tB=tB, tB_note=tB_note, n_val=len(val_rows),
        stage_a_auc_roc=stage_a_auc_roc, stage_a_auc_pr=stage_a_auc_pr, stage_a_accuracy=stage_a_accuracy,
        pipeline_precision_flooded=precision_rate, pipeline_recall_flooded=recall_rate,
        pipeline_precision_num=precision_num, pipeline_precision_den=precision_den,
        pipeline_recall_num=recall_num, pipeline_recall_den=recall_den,
        false_alarm_dry_rate=far_dry_rate, false_alarm_dry_num=far_dry_num, false_alarm_dry_den=far_dry_den,
        false_alarm_wet_rate=far_wet_rate, false_alarm_wet_num=far_wet_num, false_alarm_wet_den=far_wet_den,
        small_sample_flags=flags, confusion=confusion,
    )


def load_checkpoint(run_id: str, models_root: str = "models"):
    import keras

    path = Path(models_root) / run_id / "model.keras"
    return keras.models.load_model(path)


def variant_selection_key(precision: float, recall: float, target_precision: float = 0.90) -> tuple:
    """Ranking key for choosing between Stage B variants from their
    PIPELINE-level precision/recall on "flooded" (higher sorts better).

    Among variants that clear `target_precision`, rank by F1 -- NOT by
    precision alone, which would let a variant win by posting a slightly
    higher precision while its recall collapses. That is exactly what
    B-spec does: trained only on wet+flooded, it has no calibration for the
    dry images Stage A inevitably lets through as false "wet surface" calls,
    and most real floods end up predicted "wet" instead of "flooded" (see
    docs/phase_reports/phase3_modeling.md for the actual numbers -- B-spec's
    stage-level val AUC is a perfect-looking 1.0 while its pipeline recall
    for "flooded" is 0.125). Below the precision target, precision alone
    takes priority, since avoiding false flood alarms is this project's
    stated priority.
    """
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    meets_target = precision >= target_precision
    return (meets_target, f1 if meets_target else precision)


def run_pipeline_selection(
    stage_a_run: str,
    stage_b_runs: dict[str, str],
    *,
    manifest: str = "data/processed/manifest.csv",
    img_size: int = 224,
    models_root: str = "models",
    target_precision: float = 0.90,
) -> dict:
    """`stage_b_runs`: e.g. {"spec": "<run_id>", "mixed": "<run_id>"}.
    Evaluates the pipeline for each Stage B variant against the same Stage A
    model, on ALL val rows, and returns a dict with both variants' reports
    plus the recommended variant (see `variant_selection_key`).
    """
    df = load_manifest(manifest)
    val_rows = select_stage_rows(df, "a", split="val")  # all val rows, any label

    stage_a_model = load_checkpoint(stage_a_run, models_root)
    pA = predict_probs(stage_a_model, val_rows, img_size)

    reports = {}
    for variant, run_id in stage_b_runs.items():
        stage_b_model = load_checkpoint(run_id, models_root)
        pB = predict_probs(stage_b_model, val_rows, img_size)
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
        args.stage_a_run, stage_b_runs, manifest=args.manifest, img_size=args.img_size,
        models_root=args.models_root, target_precision=args.target_precision,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
