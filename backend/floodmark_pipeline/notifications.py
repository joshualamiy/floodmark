from __future__ import annotations

import hashlib
import hmac
import html
import logging
from typing import Any
from urllib.parse import urlencode

import aiohttp
import asyncpg

from .config import Settings

logger = logging.getLogger(__name__)


def unsubscribe_token(email_id: str, camera_id: str, secret: str) -> str:
    return hmac.new(secret.encode(), f"{email_id}:{camera_id}".encode(), hashlib.sha256).hexdigest()


async def send_flood_alerts(
    pool: asyncpg.Pool,
    session: aiohttp.ClientSession,
    settings: Settings,
    camera_id: str,
    captured_at: Any,
    confidence: float,
) -> None:
    if not settings.resend_api_key:
        logger.warning("RESEND_API_KEY is not configured; flood alert was not sent camera_id=%s", camera_id)
        return
    if not settings.notification_unsubscribe_secret:
        logger.warning("NOTIFICATION_UNSUBSCRIBE_SECRET is not configured; flood alert was not sent camera_id=%s", camera_id)
        return

    recipients = await pool.fetch(
        """SELECT ne.id AS email_id, ne.email, c.id AS camera_id, c.name
           FROM camera_notification_emails cne
           JOIN notification_emails ne ON ne.id = cne.email_id
           JOIN cameras c ON c.id = cne.camera_id
           WHERE cne.camera_id = $1
             AND ne.verified_at IS NOT NULL
             AND ne.confirmed = true""",
        camera_id,
    )
    for recipient in recipients:
        unsubscribe_query = urlencode(
            {
                "email": str(recipient["email_id"]),
                "camera": str(recipient["camera_id"]),
                "token": unsubscribe_token(str(recipient["email_id"]), str(recipient["camera_id"]), settings.notification_unsubscribe_secret),
            }
        )
        camera_name = html.escape(recipient["name"])
        unsubscribe_url = f"{settings.public_app_url}/api/notifications/unsubscribe?{unsubscribe_query}"
        payload = {
            "from": settings.resend_alerts_from,
            "to": [recipient["email"]],
            "subject": f"Flood detected at {recipient['name']}",
            "html": (
                f"<p>Floodmark detected flooding at <strong>{camera_name}</strong>.</p>"
                f"<p>Detection time: {captured_at.isoformat()}<br>Confidence: {confidence}</p>"
                f"<p><a href='{settings.public_app_url}/map'>Open Floodmark</a></p>"
                f"<p><a href='{unsubscribe_url}'>Unsubscribe from this camera</a></p>"
            ),
        }
        try:
            async with session.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {settings.resend_api_key}", "Content-Type": "application/json"},
                json=payload,
            ) as response:
                if response.status >= 300:
                    logger.error("Resend flood alert failed status=%s email_id=%s", response.status, recipient["email_id"])
        except Exception:
            logger.exception("Resend flood alert request failed email_id=%s", recipient["email_id"])
