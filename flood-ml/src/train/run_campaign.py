"""improve_v2 training campaign: runs every training job for the geometry
(crop/squash/letterbox @ 224/320) and night-augmentation experiments as ONE
long-lived detached process, so an interrupted foreground session can't kill
mid-run training (see docs/phase_reports/improve_v2.md). Meant to be started
with `subprocess.Popen([...], start_new_session=True)` and polled via
`logs/jobs/campaign_status.json`.

Stages: 4 screening runs (squash@224, letterbox@224, head-only) -> pick the
better mode by pipeline recall at precision>=0.90 -> full two-phase runs
(head+ft) for crop@224 (isolates the new augmentation alone), the winning
mode @224, and the winning mode @320.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_campaign")

STATUS_PATH = Path("logs/jobs/campaign_status.json")
RESULTS_PATH = Path("reports/eval/campaign_runs.json")

HEAD_ONLY = {"epochs_head": 8, "epochs_ft": 0}
FULL = {"epochs_head": 6, "epochs_ft": 10, "ft_layers": 30}  # matches the shipped fine-tune recipe


def _status(**kw) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
    STATUS_PATH.write_text(json.dumps(payload, indent=2))


def _train(*, stage, mode, img_size, tag, recipe, variant=None, notes=""):
    from train.train import parse_args, run_training

    ts = time.strftime("%Y%m%d-%H%M%S")
    stage_tag = f"{stage}{variant}" if variant else stage
    run_id = f"{ts}_{stage_tag}_mobilenetv3small_{tag}"
    argv = [
        "--stage", stage, "--backbone", "mobilenetv3small",
        "--img-size", str(img_size), "--mode", mode, "--run-id", run_id,
        "--epochs-head", str(recipe["epochs_head"]), "--epochs-ft", str(recipe.get("epochs_ft", 0)),
        "--notes", f"improve_v2 {tag}; {notes}".strip("; "),
    ]
    if variant:
        argv += ["--variant", variant]
    if "ft_layers" in recipe:
        argv += ["--ft-layers", str(recipe["ft_layers"])]
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
    _status(phase="screening", done=[])

    screen = {}
    for mode in ("squash", "letterbox"):
        a_run = _train(stage="a", mode=mode, img_size=224, tag=f"{mode}224", recipe=HEAD_ONLY, variant=None)
        b_run = _train(
            stage="b", mode=mode, img_size=224, tag=f"{mode}224", recipe=HEAD_ONLY, variant="mixed",
        )
        screen[mode] = (a_run, b_run)
        all_runs[f"screen_{mode}224"] = {"stage_a": a_run, "stage_b_mixed": b_run}
        _status(phase="screening", done=list(all_runs.keys()))

    reports = {
        mode: _pipeline_report(a, b, mode, 224, mode) for mode, (a, b) in screen.items()
    }
    from train.pipeline_eval import variant_selection_key

    def _key(mode):
        r = reports[mode]
        return variant_selection_key(r["pipeline_precision_flooded"], r["pipeline_recall_flooded"])

    winner_mode = max(screen, key=_key)
    logger.info("screening winner mode: %s | reports=%s", winner_mode, json.dumps(reports))
    all_runs["screening_reports"] = reports
    all_runs["winner_mode"] = winner_mode
    _status(phase="full_training", winner_mode=winner_mode, done=list(all_runs.keys()))

    # crop@224 + the new augmentation only (isolates augmentation from geometry)
    a_run = _train(stage="a", mode="crop", img_size=224, tag="crop224newaug", recipe=FULL, variant=None,
                    notes="new camera_style: night op + label-aware overlay rate")
    b_run = _train(stage="b", mode="crop", img_size=224, tag="crop224newaug", recipe=FULL, variant="mixed",
                    notes="new camera_style: night op + label-aware overlay rate")
    all_runs["crop224_newaug"] = {"stage_a": a_run, "stage_b_mixed": b_run}
    _status(phase="full_training", winner_mode=winner_mode, done=list(all_runs.keys()))

    for size in (224, 320):
        a_run = _train(stage="a", mode=winner_mode, img_size=size, tag=f"{winner_mode}{size}", recipe=FULL,
                        variant=None)
        b_run = _train(stage="b", mode=winner_mode, img_size=size, tag=f"{winner_mode}{size}", recipe=FULL,
                        variant="mixed")
        all_runs[f"{winner_mode}{size}"] = {"stage_a": a_run, "stage_b_mixed": b_run}
        _status(phase="full_training", winner_mode=winner_mode, done=list(all_runs.keys()))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_runs, indent=2))
    _status(phase="done", winner_mode=winner_mode, done=list(all_runs.keys()))
    logger.info("campaign complete: %s", json.dumps(all_runs, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("campaign failed")
        _status(phase="failed")
        raise
