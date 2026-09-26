from __future__ import annotations

import json
from datetime import datetime

import asyncpg

from .contracts import CaptureJob, Prediction


async def active_cameras(pool: asyncpg.Pool) -> list[asyncpg.Record]:
    return await pool.fetch(
        """SELECT id, source, source_camera_id, source_view_id
           FROM cameras WHERE is_active = true AND source_view_id IS NOT NULL"""
    )


async def insert_image(
    pool: asyncpg.Pool, job: CaptureJob, bucket: str, key: str, byte_size: int, checksum: str, fetched_at: datetime
) -> str:
    return await pool.fetchval(
        """INSERT INTO images (camera_id, source_url, r2_bucket, r2_key, content_type, byte_size, sha256, captured_at, fetched_at)
           VALUES ($1, $2, $3, $4, 'image/jpeg', $5, $6, $7, $8)
           ON CONFLICT (r2_bucket, r2_key) DO UPDATE SET r2_key = EXCLUDED.r2_key
           RETURNING id""",
        job.camera_id,
        f"https://511ga.org/map/Cctv/{job.source_view_id}",
        bucket,
        key,
        byte_size,
        checksum,
        job.scheduled_at,
        fetched_at,
    )


async def persist_prediction(
    pool: asyncpg.Pool, image_id: str, prediction: Prediction, heatmap_key: str | None, started_at: datetime, completed_at: datetime
) -> None:
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """INSERT INTO predictions (
                     image_id, model_version, status, confidence, stage_a_probabilities, stage_b_probabilities,
                     stage_probabilities, thresholds, note, heatmap_r2_key, inference_started_at, inference_completed_at
                   ) VALUES ($1, $2::jsonb, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb, $9, $10, $11, $12)
                   ON CONFLICT (image_id, model_version) DO NOTHING""",
                image_id, json.dumps(prediction.model_version), prediction.status, prediction.confidence,
                json.dumps(prediction.stage_a_probabilities), json.dumps(prediction.stage_b_probabilities),
                json.dumps(prediction.stage_probabilities), json.dumps(prediction.thresholds), prediction.note,
                heatmap_key, started_at, completed_at,
            )
            await connection.execute(
                """UPDATE images SET processing_status = 'processed', processed_at = $2, processing_error = NULL
                   WHERE id = $1""",
                image_id, completed_at,
            )


async def mark_image_error(pool: asyncpg.Pool, image_id: str, error: str) -> None:
    await pool.execute(
        "UPDATE images SET processing_status = 'error', processing_error = $2 WHERE id = $1",
        image_id,
        error[:4000],
    )
