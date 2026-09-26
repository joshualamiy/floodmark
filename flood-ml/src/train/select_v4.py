# v4 vs v3 selection on the new (daytime) val (rule fixed before looking)
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("select_v4")

CAMPAIGN_JSON = Path("reports/eval/campaign_v4_runs.json")
OUT_JSON = Path("reports/eval/v4_candidates.json")
CONFIG_JSON = Path("models/config.json")


def _candidate_report(stage_a_run, stage_b_run, mode, img_size, label, tA=None, tB=None, models_root="models"):
    from train.candidate_eval import (
        ga511_daytime_false_alarms,
        live_false_alarms,
        perturb_flip_rate,
    )
    from train.data import load_manifest, select_all_rows
    from train.pipeline_eval import (
        evaluate_pipeline,
        load_checkpoint,
        per_source_breakdown,
        predict_probs,
    )

    df = load_manifest()
    val_rows = select_all_rows(df, split="val")

    stage_a_model = load_checkpoint(stage_a_run, models_root)
    stage_b_model = load_checkpoint(stage_b_run, models_root)
    pA = predict_probs(stage_a_model, val_rows, img_size, mode=mode)
    pB = predict_probs(stage_b_model, val_rows, img_size, mode=mode)

    report = evaluate_pipeline(val_rows, pA, pB, tA=tA, tB=tB)
    by_source = per_source_breakdown(val_rows, pA, pB, report.tA, report.tB)
    daytime = ga511_daytime_false_alarms(val_rows, pA, pB, report.tA, report.tB)

    flooded_rows = val_rows[val_rows["label"] == "flooded"].reset_index(drop=True)
    perturb = perturb_flip_rate(stage_a_model, stage_b_model, flooded_rows, img_size, mode, report.tA, report.tB)
    live = live_false_alarms(stage_a_model, stage_b_model, img_size, mode, report.tA, report.tB)

    return {
        "label": label, "stage_a_run": stage_a_run, "stage_b_run": stage_b_run,
        "mode": mode, "img_size": img_size,
        "pipeline_report": asdict(report), "by_source": by_source,
        "ga511_daytime_false_alarms": daytime,
        "perturb": perturb, "live_false_alarms": live,
    }


def apply_selection_rule(v3_asis: dict, v4: dict) -> dict:
    # rule fixed before looking; baseline = v3 as deployed
    v3_pr = v3_asis["pipeline_report"]
    v4_pr = v4["pipeline_report"]
    v3_day = v3_asis["ga511_daytime_false_alarms"]["combined"]["rate"]
    v4_day = v4["ga511_daytime_false_alarms"]["combined"]["rate"]

    rule_a = v4_pr["pipeline_precision_flooded"] >= 0.90
    rule_b = v4_pr["pipeline_recall_flooded"] >= (v3_pr["pipeline_recall_flooded"] - 0.02)
    rule_c = v4_day < v3_day
    rule_d = v4_pr["false_alarm_not_flooded_rate"] <= (v3_pr["false_alarm_not_flooded_rate"] + 0.05)
    ship_v4 = bool(rule_a and rule_b and rule_c and rule_d)

    return {
        "rule_a_precision_>=_0.90": rule_a,
        "rule_b_recall_>=_v3_asis_minus_0.02": rule_b,
        "rule_c_daytime_far_combined_strictly_lower_than_v3_asis": rule_c,
        "rule_d_not_flooded_far_<=_v3_asis_plus_0.05": rule_d,
        "ship_v4": ship_v4,
        "v3_asis_precision": v3_pr["pipeline_precision_flooded"],
        "v3_asis_recall": v3_pr["pipeline_recall_flooded"],
        "v3_asis_daytime_far_combined": v3_day,
        "v3_asis_not_flooded_far": v3_pr["false_alarm_not_flooded_rate"],
        "v4_precision": v4_pr["pipeline_precision_flooded"],
        "v4_recall": v4_pr["pipeline_recall_flooded"],
        "v4_daytime_far_combined": v4_day,
        "v4_not_flooded_far": v4_pr["false_alarm_not_flooded_rate"],
    }


def main() -> None:
    config = json.loads(CONFIG_JSON.read_text())
    shipped_a_run = config["stage_a"]["run_id"]
    shipped_b_run = config["stage_b"]["run_id"]
    shipped_tA = config["stage_a"]["threshold_tA"]
    shipped_tB = config["stage_b"]["threshold_tB"]
    shipped_mode = config["preprocess"]["mode"]
    shipped_size = config["preprocess"]["size"]

    campaign = json.loads(CAMPAIGN_JSON.read_text())

    results = {}
    logger.info("evaluating v3_asis (shipped weights, deployed tA/tB)")
    results["v3_asis"] = _candidate_report(
        shipped_a_run, shipped_b_run, shipped_mode, shipped_size, "v3_asis", tA=shipped_tA, tB=shipped_tB,
    )
    logger.info("evaluating v3_retuned (shipped weights, tA/tB retuned on new val)")
    results["v3_retuned"] = _candidate_report(
        shipped_a_run, shipped_b_run, shipped_mode, shipped_size, "v3_retuned",
    )

    a_run, b_run = campaign["stage_a"], campaign["stage_b_mixed"]
    logger.info("evaluating v4 (%s, %s, letterbox320)", a_run, b_run)
    results["v4"] = _candidate_report(a_run, b_run, "letterbox", 320, "v4", models_root="models/candidates")

    selection = apply_selection_rule(results["v3_asis"], results["v4"])
    logger.info("selection: %s", json.dumps(selection, indent=2))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"results": results, "selection": selection}, indent=2))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
