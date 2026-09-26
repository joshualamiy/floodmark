from __future__ import annotations

import json
from datetime import datetime

import asyncpg

from .alerting import HistoryFrame
from .contracts import CaptureJob, Prediction


async def active_cameras(pool: asyncpg.Pool) -> list[asyncpg.Record]:
    return await pool.fetch(
        """SELECT id, source, source_camera_id, source_view_id, latitude, longitude
           FROM cameras WHERE is_active = true AND source_view_id IS NOT NULL"""
    )


async def camera_frame_history(pool: asyncpg.Pool, image_id: str, limit: int) -> list[HistoryFrame]:
    """Earlier predicted frames of the same camera, newest first."""
    rows = await pool.fetch(
        """SELECT i.sha256, p.status, p.alert_status,
                  COALESCE((p.stage_probabilities->>'flooded')::float, 0.0) AS flood_score
           FROM predictions p
           JOIN images i ON i.id = p.image_id
           WHERE i.camera_id = (SELECT camera_id FROM images WHERE id = $1)
             AND i.id <> $1
             AND i.captured_at < (SELECT captured_at FROM images WHERE id = $1)
           ORDER BY i.captured_at DESC NULLS LAST, p.created_at DESC
           LIMIT $2""",
        image_id,
        limit,
    )
    return [HistoryFrame(row["sha256"], row["status"], row["alert_status"], row["flood_score"]) for row in rows]


async def camera_flood_baseline(pool: asyncpg.Pool, image_id: str, days: int, min_frames: int) -> float | None:
    """Median flood score of the camera over the last ``days``; None until ``min_frames`` exist."""
    row = await pool.fetchrow(
        """SELECT count(*) AS frames,
                  percentile_cont(0.5) WITHIN GROUP (
                    ORDER BY COALESCE((p.stage_probabilities->>'flooded')::float, 0.0)
                  ) AS median
           FROM predictions p
           JOIN images i ON i.id = p.image_id
           WHERE i.camera_id = (SELECT camera_id FROM images WHERE id = $1)
             AND i.id <> $1
             AND i.captured_at >= now() - make_interval(days => $2)""",
        image_id,
        days,
    )
    if row is None or row["frames"] < min_frames or row["median"] is None:
        return None
    return float(row["median"])


async def insert_image(
    pool: asyncpg.Pool,
    job: CaptureJob,
    source_url: str,
    bucket: str,
    key: str,
    byte_size: int,
    checksum: str,
    fetched_at: datetime,
    processing_status: str = "unprocessed",
    processing_error: str | None = None,
) -> str:
    return await pool.fetchval(
        """INSERT INTO images (
             camera_id, source_url, s3_bucket, s3_key, content_type, byte_size, sha256,
             captured_at, fetched_at, processing_status, processing_error
           ) VALUES ($1, $2, $3, $4, 'image/jpeg', $5, $6, $7, $8, $9::image_processing_status, $10)
           ON CONFLICT (s3_bucket, s3_key) DO UPDATE SET s3_key = EXCLUDED.s3_key
           RETURNING id""",
        job.camera_id,
        source_url,
        bucket,
        key,
        byte_size,
        checksum,
        job.scheduled_at,
        fetched_at,
        processing_status,
        processing_error,
    )


async def persist_prediction(
    pool: asyncpg.Pool,
    image_id: str,
    prediction: Prediction,
    heatmap_key: str | None,
    started_at: datetime,
    completed_at: datetime,
) -> tuple[bool, bool]:
    async with pool.acquire() as connection:
        async with connection.transaction():
            previous_status = await connection.fetchval(
                """SELECT p.alert_status
                   FROM predictions p
                   JOIN images i ON i.id = p.image_id
                   WHERE i.camera_id = (SELECT camera_id FROM images WHERE id = $1)
                     AND p.image_id <> $1
                     AND i.captured_at < (SELECT captured_at FROM images WHERE id = $1)
                   ORDER BY i.captured_at DESC NULLS LAST, p.created_at DESC
                   LIMIT 1""",
                image_id,
            )
            inserted_id = await connection.fetchval(
                """INSERT INTO predictions (
                      image_id, model_version, status, confidence, stage_a_probabilities, stage_b_probabilities,
                      stage_probabilities, thresholds, note, heatmap_r2_key, alert_status, alert_note,
                      heatmap_status, heatmap_note, inference_started_at, inference_completed_at
                    ) VALUES ($1, $2::jsonb, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (image_id, model_version) DO NOTHING
                    RETURNING id""",
                image_id, json.dumps(prediction.model_version), prediction.status, prediction.confidence,
                json.dumps(prediction.stage_a_probabilities), json.dumps(prediction.stage_b_probabilities),
                json.dumps(prediction.stage_probabilities), json.dumps(prediction.thresholds), prediction.note,
                heatmap_key, prediction.alert_status, prediction.alert_note, prediction.heatmap_status,
                prediction.heatmap_note, started_at, completed_at,
            )
            await connection.execute(
                """UPDATE images SET processing_status = 'processed', processed_at = $2, processing_error = NULL
                   WHERE id = $1""",
                image_id, completed_at,
            )
            inserted = inserted_id is not None
            return inserted, inserted and prediction.alert_status == "flooded" and previous_status != "flooded"


async def mark_image_error(pool: asyncpg.Pool, image_id: str, error: str) -> None:
    await pool.execute(
        "UPDATE images SET processing_status = 'error', processing_error = $2 WHERE id = $1",
        image_id,
        error[:4000],
    )
