"""Cross-process rate limiting and small file-locking helpers shared by every
ga511 module (collector, event poller, ad-hoc scripts).

The limiter enforces a cap on API calls in any rolling time window by keeping
call timestamps in a small JSON state file, guarded by `fcntl.flock` so
multiple OS processes pointed at the same file never race.
"""
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

DEFAULT_MAX_CALLS = 8
DEFAULT_WINDOW_S = 60.0

# Matches `key=...` (case-insensitive) up to the next `&` or end of string, in
# a URL query string or in the text of an exception (requests exceptions
# embed the full URL, key and all).
_KEY_RE = re.compile(r"([?&]key=)[^&\s]*", re.IGNORECASE)


def redact(text: str | None) -> str:
    """Strip `key=<value>` query params from a string.

    Use this on every log line and exception message that might contain a
    511GA URL, since the API key travels in the query string.
    """
    if text is None:
        return ""
    return _KEY_RE.sub(r"\1REDACTED", str(text))


@contextmanager
def locked_file(path, create_mode: int = 0o644):
    """Open `path` (creating it if needed) and hold an exclusive flock on it
    for the duration of the `with` block. Yields the raw file descriptor.

    This is a general-purpose cross-process mutex: callers may read/write the
    file itself (as ratelimit's state does), or just use the lock to guard
    operations on a *different* path (as snapshot.py and collect.py do for
    last-fetch state and frames.csv).
    """
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
    """Exponential backoff with full jitter for a 0-indexed retry `attempt`."""
    exp = min(cap, base * (2**attempt))
    return rng() * exp


class RateLimiter:
    """Cross-process limiter: at most `max_calls` calls in any rolling
    `window_s` seconds, shared by every process pointed at `state_path`.

    `time_func` / `sleep_func` are injectable so tests can use a fake clock
    instead of the wall clock.
    """

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
        """Block until a call slot is available (cross-process), then
        reserve it. Raises TimeoutError if the projected wait would exceed
        `max_wait_s`, rather than sleeping past the caller's budget.
        """
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
        """Number of calls counted in the current window (for status/debug)."""
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
