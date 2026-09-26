"""Make `src/` importable regardless of the working directory pytest is run
from. CI runs `pytest backend flood-ml` from the repo root, where
`flood-ml/pyproject.toml`'s pytest settings (pythonpath = ["src"]) don't
apply, so we insert the path manually here.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
