# shared paths + logging for ga511
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
SRC_DIR = _THIS_FILE.parents[1]
FLOOD_ML_DIR = _THIS_FILE.parents[2]


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".env").exists():
            return parent
    return start.parents[0]


REPO_ROOT = _find_repo_root(FLOOD_ML_DIR)
ENV_PATH = REPO_ROOT / ".env"

DATA_DIR = FLOOD_ML_DIR / "data" / "ga511"
FRAMES_DIR = DATA_DIR / "frames"
PLACEHOLDER_DIR = DATA_DIR / "placeholders"
DEAD_SAMPLES_DIR = DATA_DIR / "dead_samples"
FLOOD_EVENT_FRAMES_DIR = DATA_DIR / "flood_event_frames"
WEATHER_CACHE_DIR = DATA_DIR / "weather_cache"

CAMERAS_CACHE_PATH = DATA_DIR / "cameras.json"
CAMERAS_ATLANTA_CSV = DATA_DIR / "cameras_atlanta.csv"
FRAMES_CSV = DATA_DIR / "frames.csv"
FRAMES_CSV_LOCK = DATA_DIR / "frames.csv.lock"
RATELIMIT_STATE_PATH = DATA_DIR / "ratelimit_state.json"
LAST_FETCH_PATH = DATA_DIR / "last_fetch.json"
PREV_PHASH_PATH = DATA_DIR / "prev_phash.json"
SEEN_EVENTS_PATH = DATA_DIR / "seen_events.json"
FLOOD_EVENTS_LOG = DATA_DIR / "flood_events.log"
COLLECTOR_PID_PATH = DATA_DIR / "collector.pid"

REPORTS_DIR = FLOOD_ML_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_JOBS_DIR = FLOOD_ML_DIR / "logs" / "jobs"


def ensure_dirs() -> None:
    for d in (
        DATA_DIR,
        FRAMES_DIR,
        PLACEHOLDER_DIR,
        DEAD_SAMPLES_DIR,
        FLOOD_EVENT_FRAMES_DIR,
        WEATHER_CACHE_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        LOGS_JOBS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(f"ga511.{name}")
    logger.setLevel(level)
    log_path = LOGS_JOBS_DIR / f"{name}.log"
    has_file = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_path)
        for h in logger.handlers
    )
    if not has_file:
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10_000_000, backupCount=3
        )
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(fh)
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(
        h, logging.FileHandler) for h in logger.handlers)
    if not has_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(ch)
    logger.propagate = False
    return logger

