from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from urllib.parse import quote

import aiohttp
import asyncpg
from arq import Retry
from arq.connections import RedisSettings

from .config import Settings
from .contracts import CaptureJob
from .database import insert_image, mark_image_error, persist_prediction
from .inference import model_run
from .image_quality import InvalidImageError, ensure_usable_image
from .keys import object_key, skipped_key
from .queues import CAPTURE_QUEUE_NAME
from .resources import ByteBudget, content_sha256, download_image, normalize_jpeg, s3_client, upload_object

logger = logging.getLogger(__name__)
settings = Settings.from_env()


async def startup(ctx: dict) -> None:
    timeout = aiohttp.ClientTimeout(total=settings.request_timeout_seconds)
    ctx["session"] = aiohttp.ClientSession(timeout=timeout)
    ctx["pool"] = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=settings.database_pool_size)
    ctx["s3"] = s3_client(settings)
    ctx["download_limit"] = asyncio.Semaphore(settings.download_concurrency)
    ctx["upload_limit"] = asyncio.Semaphore(settings.s3_upload_concurrency)
    ctx["inference_limit"] = asyncio.Semaphore(settings.inference_concurrency)
    ctx["database_limit"] = asyncio.Semaphore(settings.database_concurrency)
    ctx["byte_budget"] = ByteBudget(settings.max_inflight_bytes)


async def shutdown(ctx: dict) -> None:
    await ctx["session"].close()
    await ctx["pool"].close()


async def capture_camera(ctx: dict, payload: dict[str, str]) -> None:
    job = CaptureJob.from_payload(payload)
    image_id: str | None = None
    normalized: bytes | None = None
    total_started = time.perf_counter()
    download_seconds = 0.0
    s3_seconds = 0.0
    processing_seconds = 0.0
    database_seconds = 0.0
    if settings.debug:
        logger.info("capture started capture_id=%s camera_id=%s", job.capture_id, job.source_camera_id)
    try:
        source_url = f"{settings.camera_base_url}/{quote(job.source_view_id, safe='')}"
        async with ctx["download_limit"]:
            download_started = time.perf_counter()
            raw = await download_image(ctx["session"], source_url, settings, ctx["byte_budget"])
            download_seconds = time.perf_counter() - download_started
        try:
            quality_error: str | None = None
            try:
                ensure_usable_image(raw)
            except InvalidImageError as error:
                quality_error = str(error)
            if quality_error is None:
                processing_started = time.perf_counter()
                normalized = await normalize_jpeg(raw)
                processing_seconds += time.perf_counter() - processing_started
                await ctx["byte_budget"].acquire(len(normalized))
        finally:
            await ctx["byte_budget"].release(len(raw))

        fetched_at = datetime.now(timezone.utc)
        if quality_error is not None:
            async with ctx["database_limit"]:
                database_started = time.perf_counter()
                await insert_image(
                    ctx["pool"],
                    job,
                    settings.s3_bucket,
                    skipped_key(job.source_view_id, job.scheduled_at),
                    len(raw),
                    content_sha256(raw),
                    fetched_at,
                    processing_status="skipped",
                    processing_error=quality_error,
                )
                database_seconds += time.perf_counter() - database_started
            logger.info(
                "capture skipped capture_id=%s camera_id=%s reason=%s",
                job.capture_id,
                job.source_camera_id,
                quality_error,
            )
            return

        frame_key = object_key("captures", job.source_view_id, job.scheduled_at)
        async with ctx["upload_limit"]:
            upload_started = time.perf_counter()
            await upload_object(ctx["s3"], settings.s3_bucket, frame_key, normalized, "image/jpeg")
            s3_seconds += time.perf_counter() - upload_started
        async with ctx["database_limit"]:
            database_started = time.perf_counter()
            image_id = await insert_image(
                ctx["pool"], job, settings.s3_bucket, frame_key, len(normalized), content_sha256(normalized), fetched_at
            )
            database_seconds += time.perf_counter() - database_started

        started_at = datetime.now(timezone.utc)
        async with ctx["inference_limit"]:
            processing_started = time.perf_counter()
            prediction = await asyncio.to_thread(model_run, normalized)
            processing_seconds += time.perf_counter() - processing_started
        heatmap_key = None
        if prediction.heatmap_bytes:
            heatmap_key = object_key("heatmaps", job.source_view_id, job.scheduled_at)
            async with ctx["upload_limit"]:
                upload_started = time.perf_counter()
                await upload_object(ctx["s3"], settings.s3_bucket, heatmap_key, prediction.heatmap_bytes, "image/png")
                s3_seconds += time.perf_counter() - upload_started
        completed_at = datetime.now(timezone.utc)
        async with ctx["database_limit"]:
            database_started = time.perf_counter()
            await persist_prediction(ctx["pool"], image_id, prediction, heatmap_key, started_at, completed_at)
            database_seconds += time.perf_counter() - database_started
        if settings.debug:
            logger.info(
                "capture completed capture_id=%s camera_id=%s download_seconds=%.3f s3_seconds=%.3f "
                "processing_seconds=%.3f database_seconds=%.3f total_seconds=%.3f",
                job.capture_id,
                job.source_camera_id,
                download_seconds,
                s3_seconds,
                processing_seconds,
                database_seconds,
                time.perf_counter() - total_started,
            )
    except Exception as error:
        if image_id and ctx["job_try"] >= settings.arq_max_tries:
            async with ctx["database_limit"]:
                await mark_image_error(ctx["pool"], image_id, str(error))
        logger.exception(
            "capture failed capture_id=%s camera_id=%s total_seconds=%.3f",
            job.capture_id,
            job.source_camera_id,
            time.perf_counter() - total_started,
        )
        if ctx["job_try"] < settings.arq_max_tries:
            raise Retry(defer=settings.arq_retry_delay_seconds) from error
        raise
    finally:
        if normalized is not None:
            await ctx["byte_budget"].release(len(normalized))


class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    queue_name = CAPTURE_QUEUE_NAME
    functions = [capture_camera]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = settings.download_concurrency
    max_tries = settings.arq_max_tries
    keep_result = settings.arq_result_ttl_seconds
