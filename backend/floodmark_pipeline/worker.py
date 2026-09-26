from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import aiohttp
import asyncpg
from arq import Retry
from arq.connections import RedisSettings

from .alerting import AlertDecision, apply_rain_gate, decide_alert
from .config import Settings
from .contracts import CaptureJob, Prediction
from .database import camera_flood_baseline, camera_frame_history, insert_image, mark_image_error, persist_prediction
from .demo import demo_note, frame_source_url, is_demo
from .inference import model_run
from .image_quality import InvalidImageError, ensure_usable_image
from .keys import fast_poll_job_id, object_key, skipped_key
from .queues import CAPTURE_QUEUE_NAME
from .resources import ByteBudget, content_sha256, download_image, normalize_jpeg, s3_client, upload_object
from .notifications import send_flood_alerts
from .weather import RainLookup

logger = logging.getLogger(__name__)
settings = Settings.from_env()


async def startup(ctx: dict) -> None:
    timeout = aiohttp.ClientTimeout(total=settings.request_timeout_seconds)
    ctx["session"] = aiohttp.ClientSession(timeout=timeout)
    ctx["rain"] = RainLookup(ctx["session"], timeout_seconds=settings.weather_timeout_seconds)
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


async def resolve_alert(ctx: dict, job: CaptureJob, image_id: str, prediction: Prediction, checksum: str) -> AlertDecision:
    """Decide the alert status from the camera's history; dry frames cost no lookups at all."""
    history = []
    baseline = None
    rain_mm = None
    if prediction.status != "dry":
        async with ctx["database_limit"]:
            # enough rows to look past frozen repeats when counting the streak
            history = await camera_frame_history(ctx["pool"], image_id, settings.alert_streak_frames * 3 + 1)
            baseline = await camera_flood_baseline(
                ctx["pool"], image_id, settings.alert_baseline_days, settings.alert_baseline_min_frames
            )
        if is_demo(job):
            rain_mm = settings.demo_rain_mm  # the demo replays a storm, so treat it as one
        elif settings.alert_require_rain and job.latitude is not None and job.longitude is not None:
            # cached per grid cell, so this is cheap enough to fetch before deciding
            rain_mm = await ctx["rain"].rain_mm(
                job.latitude, job.longitude, job.scheduled_at, settings.alert_rain_window_hours
            )
    # storm mode: when it has rained nearby, fewer frames are needed to confirm
    storm = rain_mm is not None and rain_mm >= settings.alert_min_rain_mm
    streak_frames = settings.alert_storm_streak_frames if storm else settings.alert_streak_frames
    decision = decide_alert(
        camera_id=job.source_camera_id,
        status=prediction.status,
        flood_score=prediction.stage_probabilities.get("flooded", 0.0),
        sha256=checksum,
        history=history,
        baseline=baseline,
        streak_frames=streak_frames,
        baseline_margin=settings.alert_baseline_margin,
        blocklist=settings.alert_blocklist,
    )
    if decision.confirmed_flood and settings.alert_require_rain:
        decision = apply_rain_gate(
            decision,
            rain_mm,
            min_rain_mm=settings.alert_min_rain_mm,
            window_hours=settings.alert_rain_window_hours,
            streak_frames=streak_frames,
        )
    if is_demo(job):
        decision = replace(decision, note=demo_note(decision.note))
    if prediction.status == "flooded" and decision.status != "flooded":
        # these are the cameras to review for the blocklist
        logger.info(
            "flood call suppressed camera_id=%s score=%.3f baseline=%s reason=%s",
            job.source_camera_id,
            prediction.stage_probabilities.get("flooded", 0.0),
            "none" if baseline is None else f"{baseline:.3f}",
            decision.note,
        )
    return decision


async def schedule_fast_poll(ctx: dict, job: CaptureJob) -> None:
    """Re-capture a camera whose confirmation is in progress, instead of waiting for the next slot."""
    if job.fast_poll >= settings.alert_fast_poll_max:
        return
    run_at = datetime.now(timezone.utc) + timedelta(seconds=settings.alert_fast_poll_seconds)
    follow_up = job.follow_up(run_at)
    await ctx["redis"].enqueue_job(
        "capture_camera",
        follow_up.payload(),
        _job_id=fast_poll_job_id(job.source, job.source_camera_id, run_at),
        _defer_by=settings.alert_fast_poll_seconds,
        _queue_name=CAPTURE_QUEUE_NAME,
    )
    if settings.debug:
        logger.info(
            "fast poll scheduled camera_id=%s attempt=%d run_at=%s",
            job.source_camera_id,
            follow_up.fast_poll,
            run_at.isoformat(),
        )


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
        source_url = frame_source_url(job, settings.camera_base_url, settings.demo_frame_base_url)
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
                    source_url,
                    settings.s3_bucket,
                    skipped_key(job.source_view_id, job.scheduled_at, seconds=job.fast_poll > 0),
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

        # fast re-polls land between slots, so their keys carry seconds to avoid overwriting the slot capture
        frame_key = object_key("captures", job.source_view_id, job.scheduled_at, seconds=job.fast_poll > 0)
        async with ctx["upload_limit"]:
            upload_started = time.perf_counter()
            await upload_object(ctx["s3"], settings.s3_bucket, frame_key, normalized, "image/jpeg")
            s3_seconds += time.perf_counter() - upload_started
        checksum = content_sha256(normalized)
        async with ctx["database_limit"]:
            database_started = time.perf_counter()
            image_id = await insert_image(
                ctx["pool"], job, source_url, settings.s3_bucket, frame_key, len(normalized), checksum, fetched_at
            )
            database_seconds += time.perf_counter() - database_started

        started_at = datetime.now(timezone.utc)
        async with ctx["inference_limit"]:
            processing_started = time.perf_counter()
            prediction = await asyncio.to_thread(model_run, normalized)
            processing_seconds += time.perf_counter() - processing_started
        heatmap_key = None
        if prediction.heatmap_bytes:
            heatmap_key = object_key("heatmaps", job.source_view_id, job.scheduled_at, seconds=job.fast_poll > 0)
            async with ctx["upload_limit"]:
                upload_started = time.perf_counter()
                await upload_object(ctx["s3"], settings.s3_bucket, heatmap_key, prediction.heatmap_bytes, "image/png")
                s3_seconds += time.perf_counter() - upload_started
        alert = await resolve_alert(ctx, job, image_id, prediction, checksum)
        prediction = replace(prediction, alert_status=alert.status, alert_note=alert.note)
        completed_at = datetime.now(timezone.utc)
        async with ctx["database_limit"]:
            database_started = time.perf_counter()
            _, flood_transition = await persist_prediction(
                ctx["pool"], image_id, prediction, heatmap_key, started_at, completed_at,
            )
            database_seconds += time.perf_counter() - database_started
        if flood_transition:
            await send_flood_alerts(
                ctx["pool"],
                ctx["session"],
                settings,
                job.camera_id,
                job.scheduled_at,
                prediction.confidence,
            )
        if alert.pending:
            await schedule_fast_poll(ctx, job)
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
