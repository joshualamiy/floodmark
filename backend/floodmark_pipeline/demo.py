"""Demo cameras: a scripted storm replayed through the real pipeline.

A camera row with ``source = 'DEMO'`` is captured from the replay server
(``demo_replay.py``) instead of 511GA, and its rain check is answered with a
simulated storm so the alert rules behave as they would in real weather.
Everything else (model, baseline, streak, map, email) runs unchanged, and every
alert note says the frame came from a demo replay.

Fire captures faster than the five-minute schedule with::

    python -m floodmark_pipeline.demo fire --every 20 --count 8
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

import asyncpg
from arq.connections import RedisSettings, create_pool

from .config import Settings
from .contracts import CaptureJob
from .database import active_cameras
from .keys import fast_poll_job_id
from .queues import CAPTURE_QUEUE_NAME

logger = logging.getLogger(__name__)

DEMO_SOURCE = "DEMO"
DEMO_NOTE = "demo replay with simulated storm"


def is_demo(job: CaptureJob) -> bool:
    return job.source.upper() == DEMO_SOURCE


def frame_source_url(job: CaptureJob, camera_base_url: str, demo_frame_base_url: str) -> str:
    base = demo_frame_base_url if is_demo(job) else camera_base_url
    return f"{base}/{quote(job.source_view_id, safe='')}"


def demo_note(note: str | None) -> str:
    return f"{note}; {DEMO_NOTE}" if note else DEMO_NOTE


async def fire(settings: Settings, every: float, count: int, view_id: str | None) -> int:
    """Enqueue ``count`` captures of each demo camera, ``every`` seconds apart."""
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=2)
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    try:
        cameras = [
            camera
            for camera in await active_cameras(pool)
            if camera["source"].upper() == DEMO_SOURCE and (view_id is None or camera["source_view_id"] == view_id)
        ]
        if not cameras:
            logger.error("no active DEMO camera found; insert one first (see backend/README.md)")
            return 0
        fired = 0
        for shot in range(count):
            now = datetime.now(timezone.utc)
            for camera in cameras:
                base = CaptureJob(
                    camera_id=str(camera["id"]),
                    source=camera["source"],
                    source_camera_id=camera["source_camera_id"],
                    source_view_id=camera["source_view_id"],
                    scheduled_at=now,
                    capture_id=str(uuid.uuid4()),
                    latitude=None if camera["latitude"] is None else float(camera["latitude"]),
                    longitude=None if camera["longitude"] is None else float(camera["longitude"]),
                )
                # a follow-up job gets second-resolution keys, so shots seconds apart never collide
                job = base.follow_up(now)
                await redis.enqueue_job(
                    "capture_camera",
                    job.payload(),
                    _job_id=fast_poll_job_id(job.source, job.source_camera_id, now),
                    _queue_name=CAPTURE_QUEUE_NAME,
                )
                fired += 1
                logger.info("demo shot %d/%d camera=%s", shot + 1, count, camera["source_view_id"])
            if shot + 1 < count:
                await asyncio.sleep(every)
        return fired
    finally:
        await redis.close()
        await pool.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fire_cmd = commands.add_parser("fire", help="capture the demo camera(s) repeatedly")
    fire_cmd.add_argument("--every", type=float, default=20.0, help="seconds between shots (default 20)")
    fire_cmd.add_argument("--count", type=int, default=8, help="number of shots (default 8)")
    fire_cmd.add_argument("--view-id", default=None, help="only this demo camera's source_view_id")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fired = asyncio.run(fire(Settings.from_env(), args.every, args.count, args.view_id))
    return 0 if fired else 1


if __name__ == "__main__":
    raise SystemExit(main())
