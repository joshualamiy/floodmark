from datetime import datetime, timezone

from floodmark_pipeline.contracts import CaptureJob


def job(**overrides) -> CaptureJob:
    fields = dict(
        camera_id="uuid",
        source="SKYLINE",
        source_camera_id="13417",
        source_view_id="20827",
        scheduled_at=datetime(2026, 9, 26, 14, 20, tzinfo=timezone.utc),
        capture_id="capture",
    )
    return CaptureJob(**{**fields, **overrides})


def test_payload_round_trips_coordinates():
    original = job(latitude=33.677959, longitude=-84.318102)
    assert CaptureJob.from_payload(original.payload()) == original


def test_payload_without_coordinates_still_deserializes():
    payload = job().payload()
    del payload["latitude"]
    del payload["longitude"]
    restored = CaptureJob.from_payload(payload)
    assert restored.latitude is None
    assert restored.longitude is None
