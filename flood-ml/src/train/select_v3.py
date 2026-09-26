# v3 candidate selection on val (rule fixed before looking)
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("select_v3")

CAMPAIGN_JSON = Path("reports/eval/campaign_v3_runs.json")
OUT_JSON = Path("reports/eval/v3_candidates.json")
CONFIG_JSON = Path("models/config.json")

SMALL_N = 30


def _candidate_report(stage_a_run, stage_b_run, mode, img_size, label, tA=None, tB=None):
    from train.candidate_eval import live_false_alarms, perturb_flip_rate
    from train.data import load_manifest, select_all_rows
    from train.pipeline_eval import (
        evaluate_pipeline,
        load_checkpoint,
        per_source_breakdown,
        predict_probs,
    )

    df = load_manifest()
    val_rows = select_all_rows(df, split="val")

    stage_a_model = load_checkpoint(stage_a_run)
    stage_b_model = load_checkpoint(stage_b_run)
    pA = predict_probs(stage_a_model, val_rows, img_size, mode=mode)
    pB = predict_probs(stage_b_model, val_rows, img_size, mode=mode)

    report = evaluate_pipeline(val_rows, pA, pB, tA=tA, tB=tB)
    by_source = per_source_breakdown(val_rows, pA, pB, report.tA, report.tB)

    flooded_rows = val_rows[val_rows["label"] == "flooded"].reset_index(drop=True)
    perturb = perturb_flip_rate(stage_a_model, stage_b_model, flooded_rows, img_size, mode, report.tA, report.tB)
    live = live_false_alarms(stage_a_model, stage_b_model, img_size, mode, report.tA, report.tB)

    return {
        "label": label, "stage_a_run": stage_a_run, "stage_b_run": stage_b_run,
        "mode": mode, "img_size": img_size,
        "pipeline_report": asdict(report), "by_source": by_source,
        "perturb": perturb, "live_false_alarms": live,
    }


def main():
    config = json.loads(CONFIG_JSON.read_text())
    shipped_a_run = config["stage_a"]["run_id"]
    shipped_b_run = config["stage_b"]["run_id"]
    shipped_tA = config["stage_a"]["threshold_tA"]
    shipped_tB = config["stage_b"]["threshold_tB"]

    campaign = json.loads(CAMPAIGN_JSON.read_text())

    results = {}
    logger.info("evaluating baseline_asis (shipped weights, shipped tA/tB)")
    results["baseline_asis"] = _candidate_report(
        shipped_a_run, shipped_b_run, "crop", 224, "baseline_asis", tA=shipped_tA, tB=shipped_tB,
    )
    logger.info("evaluating baseline_retuned (shipped weights, tA/tB retuned on new val)")
    results["baseline_retuned"] = _candidate_report(
        shipped_a_run, shipped_b_run, "crop", 224, "baseline_retuned",
    )

    new_candidates = {
        "crop224": ("crop", 224),
        "letterbox224": ("letterbox", 224),
        "letterbox320": ("letterbox", 320),
    }
    for tag, (mode, size) in new_candidates.items():
        if tag not in campaign:
            logger.warning("campaign json missing %s, skipping", tag)
            continue
        a_run = campaign[tag]["stage_a"]
        b_run = campaign[tag]["stage_b_mixed"]
        logger.info("evaluating %s (%s, %s@%d)", tag, a_run, mode, size)
        results[tag] = _candidate_report(a_run, b_run, mode, size, tag)
        logger.info("done %s: %s", tag, json.dumps(results[tag]["pipeline_report"]))

    baseline = results["baseline_asis"]
    baseline_far_dry = baseline["pipeline_report"]["false_alarm_dry_rate"]
    baseline_live_flooded = baseline["live_false_alarms"]["flooded"]
    baseline_recall = baseline["pipeline_report"]["pipeline_recall_flooded"]

    eligible = []
    for tag in new_candidates:
        if tag not in results:
            continue
        pr = results[tag]["pipeline_report"]
        live_flooded = results[tag]["live_false_alarms"]["flooded"]
        ok = (
            pr["pipeline_precision_flooded"] >= 0.90
            and pr["false_alarm_dry_rate"] <= baseline_far_dry
            and live_flooded <= 2 * baseline_live_flooded
        )
        if ok:
            eligible.append(tag)

    def _tiebreak(tag):
        pr = results[tag]["pipeline_report"]
        perturb = results[tag]["perturb"]
        flips = perturb["dark"]["flipped_to_dry"] + perturb["label_box"]["flipped_to_dry"]
        return (pr["pipeline_recall_flooded"], pr["stage_a_wet_recall"], -flips)

    winner = None
    if eligible:
        best = max(eligible, key=_tiebreak)
        if results[best]["pipeline_report"]["pipeline_recall_flooded"] > baseline_recall:
            winner = best

    summary = {
        "candidates_evaluated": list(results.keys()),
        "eligible": eligible,
        "winner": winner,
        "baseline_recall": baseline_recall,
        "baseline_far_dry": baseline_far_dry,
        "baseline_live_flooded": baseline_live_flooded,
    }
    logger.info("selection summary: %s", json.dumps(summary, indent=2))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"results": results, "selection": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

