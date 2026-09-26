# v3 training campaign (detached)
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_campaign_v3")

STATUS_PATH = Path("logs/jobs/campaign_v3_status.json")
RESULTS_PATH = Path("reports/eval/campaign_v3_runs.json")

HEAD_ONLY = {"epochs_head": 6, "epochs_ft": 0}
FULL = {"epochs_head": 6, "epochs_ft": 10, "ft_layers": 30}


def _status(**kw) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
    STATUS_PATH.write_text(json.dumps(payload, indent=2))


def _train(*, stage, mode, img_size, tag, recipe, variant=None, night_aug=True, notes=""):
    from train.train import parse_args, run_training

    ts = time.strftime("%Y%m%d-%H%M%S")
    stage_tag = f"{stage}{variant}" if variant else stage
    run_id = f"{ts}_{stage_tag}_mobilenetv3small_{tag}"
    argv = [
        "--stage", stage, "--backbone", "mobilenetv3small",
        "--img-size", str(img_size), "--mode", mode, "--run-id", run_id,
        "--epochs-head", str(recipe["epochs_head"]), "--epochs-ft", str(recipe.get("epochs_ft", 0)),
        "--notes", f"v3 {tag}; {notes}".strip("; "),
    ]
    if variant:
        argv += ["--variant", variant]
    if "ft_layers" in recipe:
        argv += ["--ft-layers", str(recipe["ft_layers"])]
    if not night_aug:
        argv += ["--no-night-aug"]
    args = parse_args(argv)
    logger.info("training: %s", " ".join(argv))
    result = run_training(args)
    logger.info("done: %s -> %s", run_id, result["row"]["val_auc_pr"])
    return result["run_id"]


def _pipeline_report(stage_a_run, stage_b_run, mode, img_size, label):
    from train.pipeline_eval import run_pipeline_selection

    sel = run_pipeline_selection(
        stage_a_run, {label: stage_b_run}, img_size=img_size, mode=mode, target_precision=0.90,
    )
    return sel["variants"][label]["report"]


def main():
    all_runs: dict[str, dict] = {}
    _status(phase="night_aug_screen", done=[])

    screen = {}
    for tag, night_aug in (("nightaug_on", True), ("nightaug_off", False)):
        a_run = _train(
            stage="a", mode="crop", img_size=224, tag=f"crop224_{tag}", recipe=HEAD_ONLY,
            variant=None, night_aug=night_aug,
        )
        b_run = _train(
            stage="b", mode="crop", img_size=224, tag=f"crop224_{tag}", recipe=HEAD_ONLY,
            variant="mixed", night_aug=night_aug,
        )
        screen[tag] = (a_run, b_run)
        all_runs[f"screen_{tag}"] = {"stage_a": a_run, "stage_b_mixed": b_run}
        _status(phase="night_aug_screen", done=list(all_runs.keys()))

    reports = {tag: _pipeline_report(a, b, "crop", 224, tag) for tag, (a, b) in screen.items()}
    from train.pipeline_eval import variant_selection_key

    def _key(tag):
        r = reports[tag]
        return variant_selection_key(r["pipeline_precision_flooded"], r["pipeline_recall_flooded"])

    winner_tag = max(screen, key=_key)
    night_aug_winner = winner_tag == "nightaug_on"
    logger.info(
        "night_aug screening winner: %s (night_aug=%s) | reports=%s",
        winner_tag, night_aug_winner, json.dumps(reports),
    )
    all_runs["screening_reports"] = reports
    all_runs["night_aug_winner"] = night_aug_winner
    _status(phase="full_training", night_aug=night_aug_winner, done=list(all_runs.keys()))

    for mode, size in (("crop", 224), ("letterbox", 224), ("letterbox", 320)):
        tag = f"{mode}{size}"
        a_run = _train(
            stage="a", mode=mode, img_size=size, tag=tag, recipe=FULL, variant=None,
            night_aug=night_aug_winner,
        )
        b_run = _train(
            stage="b", mode=mode, img_size=size, tag=tag, recipe=FULL, variant="mixed",
            night_aug=night_aug_winner,
        )
        all_runs[tag] = {"stage_a": a_run, "stage_b_mixed": b_run}
        _status(phase="full_training", night_aug=night_aug_winner, done=list(all_runs.keys()))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_runs, indent=2))
    _status(phase="done", night_aug=night_aug_winner, done=list(all_runs.keys()))
    logger.info("v3 campaign complete: %s", json.dumps(all_runs, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("v3 campaign failed")
        _status(phase="failed")
        raise

