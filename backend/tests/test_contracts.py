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


def test_follow_up_is_a_fast_poll_of_the_same_camera_with_a_new_capture_id():
    original = job(latitude=33.677959, longitude=-84.318102)
    run_at = datetime(2026, 9, 26, 14, 21, 30, tzinfo=timezone.utc)
    first = original.follow_up(run_at)
    assert first.fast_poll == 1
    assert first.scheduled_at == run_at
    assert first.capture_id != original.capture_id
    assert (first.source_camera_id, first.latitude, first.longitude) == ("13417", 33.677959, -84.318102)
    second = first.follow_up(run_at)
    assert second.fast_poll == 2
    assert second.capture_id != first.capture_id
    assert CaptureJob.from_payload(second.payload()) == second


def test_payload_without_fast_poll_defaults_to_a_regular_capture():
    payload = job().payload()
    del payload["fast_poll"]
    assert CaptureJob.from_payload(payload).fast_poll == 0
