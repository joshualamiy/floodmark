from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import asyncpg
from arq import cron
from arq.connections import RedisSettings

from .config import Settings
from .contracts import CaptureJob
from .database import active_cameras
from .keys import capture_job_id, capture_slot
from .queues import CAPTURE_QUEUE_NAME, SCHEDULER_QUEUE_NAME

logger = logging.getLogger(__name__)
settings = Settings.from_env()


async def startup(ctx: dict) -> None:
    ctx["pool"] = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=settings.database_pool_size)


async def shutdown(ctx: dict) -> None:
    await ctx["pool"].close()


async def schedule_captures(ctx: dict) -> None:
    scheduled_at = capture_slot(datetime.now(timezone.utc))
    cameras = await active_cameras(ctx["pool"])
    enqueued = 0
    for camera in cameras:
        job_id = capture_job_id(camera["source"], camera["source_camera_id"], scheduled_at)
        job = CaptureJob(
            camera_id=str(camera["id"]),
            source=camera["source"],
            source_camera_id=camera["source_camera_id"],
            source_view_id=camera["source_view_id"],
            scheduled_at=scheduled_at,
            capture_id=str(uuid.uuid5(uuid.NAMESPACE_URL, job_id)),
            latitude=None if camera["latitude"] is None else float(camera["latitude"]),
            longitude=None if camera["longitude"] is None else float(camera["longitude"]),
        )
        result = await ctx["redis"].enqueue_job(
            "capture_camera", job.payload(), _job_id=job_id, _queue_name=CAPTURE_QUEUE_NAME
        )
        enqueued += result is not None
    logger.info("scheduled %d of %d active camera captures for %s", enqueued, len(cameras), scheduled_at.isoformat())


class SchedulerSettings:
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    queue_name = SCHEDULER_QUEUE_NAME
    functions = [schedule_captures]
    on_startup = startup
    on_shutdown = shutdown
    cron_jobs = [
        cron(
            schedule_captures,
            minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55},
            run_at_startup=settings.scheduler_run_at_startup,
        )
    ]
