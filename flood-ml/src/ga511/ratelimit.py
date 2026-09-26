# cross-process api rate limit (8/min), file locks, key redaction
from __future__ import annotations

import fcntl
import json
import os
import random
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

# 511ga allows 10 calls/min per key, stay under it
DEFAULT_MAX_CALLS = 8
DEFAULT_WINDOW_S = 60.0

_KEY_RE = re.compile(r"([?&]key=)[^&\s]*", re.IGNORECASE)


def redact(text: str | None) -> str:
    if text is None:
        return ""
    return _KEY_RE.sub(r"\1REDACTED", str(text))


@contextmanager
def locked_file(path, create_mode: int = 0o644):
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, create_mode)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def read_json_fd(fd: int, default):
    os.lseek(fd, 0, os.SEEK_SET)
    raw = os.read(fd, 1 << 24)
    if not raw:
        return default
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return default


def write_json_fd(fd: int, obj) -> None:
    raw = json.dumps(obj).encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, raw)
    os.ftruncate(fd, len(raw))


def backoff_delay(
    attempt: int, base: float = 1.0, cap: float = 60.0, rng: Callable[[], float] = random.random
) -> float:
    exp = min(cap, base * (2**attempt))
    return rng() * exp


class RateLimiter:
    def __init__(
        self,
        state_path,
        max_calls: int = DEFAULT_MAX_CALLS,
        window_s: float = DEFAULT_WINDOW_S,
        time_func: Callable[[], float] = time.time,
        sleep_func: Callable[[float], None] = time.sleep,
    ):
        self.state_path = str(state_path)
        self.max_calls = max_calls
        self.window_s = window_s
        self._time = time_func
        self._sleep = sleep_func

    def acquire(self, max_wait_s: float = 180.0) -> None:
        start = self._time()
        while True:
            with locked_file(self.state_path) as fd:
                state = read_json_fd(fd, {"calls": []})
                now = self._time()
                calls = [t for t in state.get("calls", []) if now - t < self.window_s]
                if len(calls) < self.max_calls:
                    calls.append(now)
                    state["calls"] = calls
                    write_json_fd(fd, state)
                    return
                wait = self.window_s - (now - calls[0]) + 0.05
                state["calls"] = calls
                write_json_fd(fd, state)
            elapsed = self._time() - start
            if elapsed + wait > max_wait_s:
                raise TimeoutError("ratelimit: exceeded max_wait_s waiting for a call slot")
            self._sleep(max(wait, 0.01))

    def current_call_count(self) -> int:
        with locked_file(self.state_path) as fd:
            state = read_json_fd(fd, {"calls": []})
            now = self._time()
            calls = [t for t in state.get("calls", []) if now - t < self.window_s]
            return len(calls)


def get_default_limiter(state_path=None) -> RateLimiter:
    if state_path is None:
        from ga511.paths import RATELIMIT_STATE_PATH

        state_path = RATELIMIT_STATE_PATH
    return RateLimiter(Path(state_path))

