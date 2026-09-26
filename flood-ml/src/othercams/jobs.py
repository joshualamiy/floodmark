"""Generic detached-job PID helpers, same pattern as ga511/daemon.py."""
from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def write_pid(pid_path: Path) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))


def read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def collection_lock(pid_path: Path):
    """Reject overlapping runs, including the older PID-only collector."""
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    with pid_path.with_suffix(".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("collection is already running") from exc
        previous = read_pid(pid_path)
        if previous and previous != os.getpid() and pid_alive(previous):
            raise RuntimeError(f"collection is already running (pid={previous})")
        write_pid(pid_path)
        try:
            yield
        finally:
            if read_pid(pid_path) == os.getpid():
                pid_path.unlink(missing_ok=True)
            fcntl.flock(lock, fcntl.LOCK_UN)


def start_detached(module: str, args: list[str], log_path: Path, cwd: Path) -> int:
    """Launch `-m module args` fully detached, own session, log to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    with open(log_path, "ab") as logf:
        proc = subprocess.Popen(
            [sys.executable, "-m", module, *args],
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def stop(pid_path: Path, timeout_s: float = 15.0) -> bool:
    pid = read_pid(pid_path)
    if pid is None or not pid_alive(pid):
        pid_path.unlink(missing_ok=True)
        return False
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.5)
    return False
