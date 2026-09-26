# puts src on the path; ci skips tests whose training deps aren't installed
import importlib.util
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


NEEDS = {
    "test_flood_master.py": ("imagehash", "pandas"),
    "test_ga511_cameras.py": ("requests", "dotenv"),
    "test_ga511_events.py": ("requests", "dotenv"),
    "test_ga511_weather.py": ("requests", "dotenv"),
    "test_ga511_quality.py": ("imagehash",),
    "test_prep_augment.py": ("cv2",),
    "test_train_*.py": ("tensorflow", "keras", "pandas"),
    "test_othercams_*.py": ("requests", "imagehash", "pandas", "dotenv"),
}
collect_ignore_glob = [pat for pat, mods in NEEDS.items() if not all(_has(m) for m in mods)]


def pytest_report_header(config):
    if collect_ignore_glob:
        return "skipped, training deps not installed: " + ", ".join(collect_ignore_glob)

