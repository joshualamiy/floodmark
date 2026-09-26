from datetime import datetime, timezone

import pytest

from floodmark_pipeline.weather import RainLookup, grid_key, rain_in_window

AT = datetime(2026, 9, 26, 14, 20, tzinfo=timezone.utc)


def payload(values: list[float | None]) -> dict:
    times = [f"2026-09-26T{hour:02d}:00" for hour in range(len(values))]
    return {"hourly": {"time": times, "precipitation": values}}


def test_rain_in_window_sums_the_hours_ending_at_the_capture_hour():
    values = [0.0] * 24
    values[8] = 5.0  # a 6 h window ending at 14:00 covers 09:00-14:00, so this is outside
    values[10] = 1.5
    values[14] = 0.5
    assert rain_in_window(payload(values), AT, 6) == pytest.approx(2.0)


def test_rain_in_window_ignores_missing_hours_and_returns_none_when_all_missing():
    values = [None] * 24
    assert rain_in_window(payload(values), AT, 6) is None
    values[13] = 0.7
    assert rain_in_window(payload(values), AT, 6) == pytest.approx(0.7)


def test_rain_in_window_returns_none_when_the_hour_is_not_in_the_payload():
    assert rain_in_window(payload([0.0] * 3), AT, 6) is None
    assert rain_in_window({}, AT, 6) is None


def test_grid_key_groups_nearby_cameras():
    assert grid_key(33.7601, -84.3921) == grid_key(33.7712, -84.3719)
    assert grid_key(33.7601, -84.3921) != grid_key(33.9601, -84.3921)


async def test_lookup_caches_per_grid_cell_and_fails_open():
    calls: list[tuple[float, float]] = []

    async def fake_fetch(latitude: float, longitude: float) -> dict | None:
        calls.append((latitude, longitude))
        return payload([0.0] * 12 + [2.0] * 12)

    lookup = RainLookup(session=None, fetch=fake_fetch)
    assert await lookup.rain_mm(33.7601, -84.3921, AT, 6) == pytest.approx(6.0)
    assert await lookup.rain_mm(33.7712, -84.3719, AT, 6) == pytest.approx(6.0)
    assert len(calls) == 1

    async def failing_fetch(latitude: float, longitude: float) -> dict | None:
        return None

    assert await RainLookup(session=None, fetch=failing_fetch).rain_mm(33.7, -84.4, AT, 6) is None
