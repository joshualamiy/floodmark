"""Paths for src/othercams, keyed by source name. Mirrors ga511/paths.py."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
FLOOD_ML_DIR = _THIS_FILE.parents[2]

DATA_ROOT = FLOOD_ML_DIR / "data" / "othercams"
REPORTS_DIR = FLOOD_ML_DIR / "reports"
LOGS_JOBS_DIR = FLOOD_ML_DIR / "logs" / "jobs"


def source_dir(source: str) -> Path:
    return DATA_ROOT / source


def frames_dir(source: str) -> Path:
    return source_dir(source) / "frames"


def frames_csv(source: str) -> Path:
    return source_dir(source) / "frames.csv"


def frames_csv_lock(source: str) -> Path:
    return source_dir(source) / "frames.csv.lock"


def dead_samples_dir(source: str) -> Path:
    return source_dir(source) / "dead_samples"


def cache_dir(source: str) -> Path:
    return source_dir(source) / "cache"


def last_fetch_path(source: str) -> Path:
    return source_dir(source) / "last_fetch.json"


def prev_phash_path(source: str) -> Path:
    return source_dir(source) / "prev_phash.json"


def pid_path(source: str) -> Path:
    return source_dir(source) / "collect.pid"


def ensure_dirs(source: str) -> None:
    for d in (source_dir(source), frames_dir(source), dead_samples_dir(source), cache_dir(source)):
        d.mkdir(parents=True, exist_ok=True)
    LOGS_JOBS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    LOGS_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"othercams.{name}")
    logger.setLevel(level)
    log_path = LOGS_JOBS_DIR / f"{name}.log"
    has_file = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_path)
        for h in logger.handlers
    )
    if not has_file:
        fh = logging.handlers.RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=3)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(fh)
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(ch)
    logger.propagate = False
    return logger
