# v4 daytime retrain: letterbox320 pair only, exact v3 recipe (data changes only)
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_campaign_v4")

STATUS_PATH = Path("logs/jobs/campaign_v4_status.json")
RESULTS_PATH = Path("reports/eval/campaign_v4_runs.json")
MODELS_ROOT = "models/candidates"

FULL = {"epochs_head": 6, "epochs_ft": 10, "ft_layers": 30}


def _status(**kw) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
    STATUS_PATH.write_text(json.dumps(payload, indent=2))


def _train(*, stage: str, variant: str | None = None, notes: str = "") -> str:
    from train.train import parse_args, run_training

    ts = time.strftime("%Y%m%d-%H%M%S")
    stage_tag = f"{stage}{variant}" if variant else stage
    run_id = f"{ts}_{stage_tag}_mobilenetv3small_v4_letterbox320"
    argv = [
        "--stage", stage, "--backbone", "mobilenetv3small",
        "--img-size", "320", "--mode", "letterbox", "--run-id", run_id,
        "--epochs-head", str(FULL["epochs_head"]), "--epochs-ft", str(FULL["epochs_ft"]),
        "--ft-layers", str(FULL["ft_layers"]), "--no-night-aug",
        "--models-root", MODELS_ROOT,
        "--notes", f"v4 daytime retrain letterbox320; {notes}".strip("; "),
    ]
    if variant:
        argv += ["--variant", variant]
    args = parse_args(argv)
    logger.info("training: %s", " ".join(argv))
    result = run_training(args)
    logger.info("done: %s -> val_auc_pr=%s", run_id, result["row"]["val_auc_pr"])
    return result["run_id"]


def main() -> None:
    _status(phase="training", done=[])
    a_run = _train(stage="a")
    _status(phase="training", done=["stage_a"], stage_a_run=a_run)
    b_run = _train(stage="b", variant="mixed")
    _status(phase="done", done=["stage_a", "stage_b_mixed"], stage_a_run=a_run, stage_b_run=b_run)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"stage_a": a_run, "stage_b_mixed": b_run}, indent=2))
    logger.info("v4 campaign complete: stage_a=%s stage_b_mixed=%s", a_run, b_run)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("v4 campaign failed")
        _status(phase="failed")
        raise
