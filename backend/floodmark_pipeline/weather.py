"""Recent rainfall near a camera from Open-Meteo, cached per grid cell."""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import aiohttp

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
GRID_STEP_DEGREES = 0.1
CACHE_TTL_SECONDS = 900

Fetch = Callable[[float, float], Awaitable[dict | None]]


def grid_key(latitude: float, longitude: float, step: float = GRID_STEP_DEGREES) -> tuple[float, float]:
    return round(round(latitude / step) * step, 3), round(round(longitude / step) * step, 3)


def rain_in_window(payload: dict, at: datetime, window_hours: int) -> float | None:
    """Sum hourly precipitation for the ``window_hours`` ending at the hour containing ``at``."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    precipitation = hourly.get("precipitation") or []
    if not times or len(times) != len(precipitation):
        return None
    target = at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00")
    try:
        end = times.index(target)
    except ValueError:
        return None
    values = [value for value in precipitation[max(0, end - window_hours + 1) : end + 1] if value is not None]
    if not values:
        return None
    return float(sum(values))


class RainLookup:
    def __init__(
        self,
        session: aiohttp.ClientSession | None,
        timeout_seconds: float = 5.0,
        fetch: Fetch | None = None,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    ) -> None:
        self._session = session
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._fetch = fetch or self._fetch_open_meteo
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[tuple[float, float], tuple[float, dict]] = {}

    async def _fetch_open_meteo(self, latitude: float, longitude: float) -> dict | None:
        params = {
            "latitude": f"{latitude:.3f}",
            "longitude": f"{longitude:.3f}",
            "hourly": "precipitation",
            "past_days": "1",
            "forecast_days": "1",
            "timezone": "UTC",
        }
        try:
            async with self._session.get(OPEN_METEO_URL, params=params, timeout=self._timeout) as response:
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, TimeoutError) as error:
            logger.warning("rain lookup failed lat=%.3f lon=%.3f: %s", latitude, longitude, error)
            return None

    async def rain_mm(self, latitude: float, longitude: float, at: datetime, window_hours: int) -> float | None:
        key = grid_key(latitude, longitude)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached is None or now - cached[0] > self._cache_ttl:
            payload = await self._fetch(key[0], key[1])
            if payload is None:
                return None
            cached = (now, payload)
            self._cache[key] = cached
        return rain_in_window(cached[1], at, window_hours)
