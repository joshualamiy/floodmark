from __future__ import annotations

import asyncio
import io
from hashlib import sha256

import aiohttp
import boto3
from botocore.config import Config
from PIL import Image

from .config import Settings


class ByteBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.available = limit
        self.condition = asyncio.Condition()

    async def acquire(self, amount: int) -> None:
        if amount > self.limit:
            raise ValueError("item exceeds the in-flight byte budget")
        async with self.condition:
            await self.condition.wait_for(lambda: self.available >= amount)
            self.available -= amount

    async def release(self, amount: int) -> None:
        async with self.condition:
            self.available += amount
            self.condition.notify_all()


async def download_image(session: aiohttp.ClientSession, url: str, settings: Settings, budget: ByteBudget) -> bytes:
    chunks: list[bytes] = []
    total = 0
    reserved = 0
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if content_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise ValueError(f"unsupported camera content type: {content_type or 'missing'}")
            async for chunk in response.content.iter_chunked(64 * 1024):
                total += len(chunk)
                if total > settings.max_response_bytes:
                    raise ValueError("camera response exceeds MAX_RESPONSE_BYTES")
                await budget.acquire(len(chunk))
                reserved += len(chunk)
                chunks.append(chunk)
        return b"".join(chunks)
    except BaseException:
        await budget.release(reserved)
        raise


async def normalize_jpeg(image_bytes: bytes) -> bytes:
    def convert() -> bytes:
        with Image.open(io.BytesIO(image_bytes)) as image:
            output = io.BytesIO()
            image.convert("RGB").save(output, format="JPEG", quality=90, optimize=True)
            return output.getvalue()

    return await asyncio.to_thread(convert)


def content_sha256(image_bytes: bytes) -> str:
    return sha256(image_bytes).hexdigest()


def r2_client(settings: Settings):
    return boto3.client(
        "s3",
        endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name="auto",
        config=Config(retries={"max_attempts": 3, "mode": "standard"}),
    )


async def upload_object(client, bucket: str, key: str, body: bytes, content_type: str) -> None:
    await asyncio.to_thread(client.put_object, Bucket=bucket, Key=key, Body=body, ContentType=content_type)
