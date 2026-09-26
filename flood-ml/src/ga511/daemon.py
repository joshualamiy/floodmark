"""Long-lived process: runs the frame collector and the event poller together,
with graceful SIGTERM handling. Meant to be started detached (nohup ... &
disown) so it keeps sampling across times of day.

Start:
    cd flood-ml && PYTHONPATH=src nohup ../my_env/bin/python -m ga511.daemon run \\
        --interval-min 60 --event-poll-min 5 \\
        > logs/jobs/ga511_collector.log 2>&1 &
    disown

Stop:
    cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m ga511.daemon stop

Status:
    cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m ga511.daemon status
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import threading
import time
from collections import Counter

from ga511 import collect, events
from ga511.paths import (
    COLLECTOR_PID_PATH,
    FLOOD_EVENT_FRAMES_DIR,
    FRAMES_CSV,
    SEEN_EVENTS_PATH,
    ensure_dirs,
    setup_logging,
)
from ga511.ratelimit import locked_file, read_json_fd

log = setup_logging("ga511_daemon")


def _write_pid(path=COLLECTOR_PID_PATH) -> None:
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def _read_pid(path=COLLECTOR_PID_PATH):
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _weather_fill_loop(interval_s: float, stop_flag) -> None:
    from ga511 import weather

    while not stop_flag.is_set():
        try:
            n = weather.fill_weak_labels()
            if n:
                log.info("weather fill: updated %d rows", n)
        except Exception as e:  # noqa: BLE001
            log.error("weather fill failed: %s", e)
        stop_flag.wait(interval_s)


def run(interval_min: float, event_poll_min: float, weather_fill_min: float = 15.0) -> None:
    ensure_dirs()
    _write_pid()
    stop_flag = threading.Event()

    def _handle_sigterm(signum, frame):
        log.info("received signal %s, shutting down", signum)
        stop_flag.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    collector_thread = threading.Thread(
        target=collect.run_forever,
        kwargs={"interval_min": interval_min, "with_weather": False, "stop_flag": stop_flag},
        daemon=True,
        name="ga511-collector",
    )
    poller_thread = threading.Thread(
        target=events.run_forever,
        kwargs={"interval_s": event_poll_min * 60.0, "stop_flag": stop_flag},
        daemon=True,
        name="ga511-event-poller",
    )
    weather_thread = threading.Thread(
        target=_weather_fill_loop,
        kwargs={"interval_s": weather_fill_min * 60.0, "stop_flag": stop_flag},
        daemon=True,
        name="ga511-weather-fill",
    )
    log.info(
        "daemon starting (pid=%d): sweep every %.0f min, poll events every %.0f min, "
        "fill weather every %.0f min",
        os.getpid(),
        interval_min,
        event_poll_min,
        weather_fill_min,
    )
    collector_thread.start()
    poller_thread.start()
    weather_thread.start()
    try:
        while not stop_flag.is_set():
            stop_flag.wait(1.0)
    finally:
        stop_flag.set()
        collector_thread.join(timeout=30)
        weather_thread.join(timeout=30)
        poller_thread.join(timeout=30)
        if COLLECTOR_PID_PATH.exists():
            COLLECTOR_PID_PATH.unlink()
        log.info("daemon stopped")


def stop(timeout_s: float = 15.0) -> bool:
    pid = _read_pid()
    if pid is None:
        print("no PID file; daemon does not appear to be running")
        return False
    if not _pid_alive(pid):
        print(f"pid {pid} not alive; removing stale PID file")
        COLLECTOR_PID_PATH.unlink(missing_ok=True)
        return False
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _pid_alive(pid):
            print(f"stopped pid {pid}")
            return True
        time.sleep(0.5)
    print(f"pid {pid} did not exit within {timeout_s}s")
    return False


def status() -> dict:
    pid = _read_pid()
    alive = pid is not None and _pid_alive(pid)

    n_ok = 0
    dead_counts: Counter = Counter()
    last_ts = None
    if FRAMES_CSV.exists():
        with open(FRAMES_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = row.get("timestamp_utc")
                if ts:
                    try:
                        ts_i = int(ts)
                        last_ts = ts_i if last_ts is None else max(last_ts, ts_i)
                    except ValueError:
                        pass
                if row.get("dead_reason"):
                    dead_counts[row["dead_reason"]] += 1
                else:
                    n_ok += 1

    n_flagged_events = 0
    with locked_file(SEEN_EVENTS_PATH) as fd:
        seen = read_json_fd(fd, {"seen": []})
        n_flagged_events = len(seen.get("seen", []))

    n_flood_event_dirs = 0
    if FLOOD_EVENT_FRAMES_DIR.exists():
        n_flood_event_dirs = sum(1 for _ in FLOOD_EVENT_FRAMES_DIR.iterdir())

    result = {
        "running": alive,
        "pid": pid,
        "frames_ok": n_ok,
        "frames_dead_by_reason": dict(dead_counts),
        "last_frame_timestamp_utc": last_ts,
        "flagged_events_total": n_flagged_events,
        "flood_event_capture_dirs": n_flood_event_dirs,
    }
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="ga511 collector + event poller daemon")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run in the foreground (use nohup to detach)")
    run_p.add_argument("--interval-min", type=float, default=60.0)
    run_p.add_argument("--event-poll-min", type=float, default=5.0)
    run_p.add_argument("--weather-fill-min", type=float, default=15.0)

    sub.add_parser("stop", help="send SIGTERM to the running daemon")
    sub.add_parser("status", help="print collector/daemon status")

    args = parser.parse_args()
    if args.command == "run":
        run(args.interval_min, args.event_poll_min, args.weather_fill_min)
    elif args.command == "stop":
        stop()
    elif args.command == "status":
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    _main()
