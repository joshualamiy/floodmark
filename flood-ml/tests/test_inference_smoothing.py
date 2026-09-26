from __future__ import annotations

from inference.smoothing import SmoothedStatus, TemporalSmoother


def test_reports_potential_flooding_until_n_consecutive_water_detections():
    sm = TemporalSmoother(n=3)
    first = sm.update("cam1", "flooded")
    second = sm.update("cam1", "flooded")
    assert first.status == "wet"
    assert first.note == "potential flooding; water frame 1/3; awaiting confirmation"
    assert second.status == "wet"
    assert second.note == "potential flooding; water frame 2/3; awaiting confirmation"
    assert sm.update("cam1", "flooded").status == "flooded"


def test_three_wet_detections_report_wet():
    sm = TemporalSmoother(n=3)
    assert sm.update("cam1", "wet").status == "wet"
    assert sm.update("cam1", "wet").status == "wet"
    assert sm.update("cam1", "wet").status == "wet"


def test_wet_and_flooded_detections_share_water_streak():
    sm = TemporalSmoother(n=3)
    assert sm.update("cam1", "wet").status == "wet"
    assert sm.update("cam1", "flooded").status == "wet"
    assert sm.update("cam1", "wet").status == "wet"


def test_non_flooded_frame_resets_streak():
    sm = TemporalSmoother(n=3)
    sm.update("cam1", "flooded")
    sm.update("cam1", "flooded")
    assert sm.update("cam1", "dry").status == "dry"
    assert sm.update("cam1", "flooded").status == "wet"


def test_dry_passes_through_and_wet_reports_potential_flooding():
    sm = TemporalSmoother(n=3)
    assert sm.update("cam1", "dry").status == "dry"
    assert sm.update("cam1", "wet").status == "wet"


def test_skip_does_not_count_or_reset_streak():
    sm = TemporalSmoother(n=3)
    sm.update("cam1", "flooded")
    sm.skip("cam1")
    sm.skip("cam1")
    assert sm.update("cam1", "flooded").status == "wet"
    assert sm.update("cam1", "flooded").status == "flooded"


def test_blocklisted_camera_never_reports_flooded():
    sm = TemporalSmoother(n=1, blocklist=["11372"])
    for _ in range(5):
        result = sm.update("11372", "flooded")
        assert result.status == "wet"
        assert result.note == "camera blocklisted; flood alerts suppressed"


def test_blocklisted_camera_still_reports_dry():
    sm = TemporalSmoother(n=1, blocklist=["11372"])
    assert sm.update("11372", "dry").status == "dry"


def test_cameras_are_independent():
    sm = TemporalSmoother(n=2)
    sm.update("cam1", "flooded")
    assert sm.update("cam2", "flooded").status == "wet"
    assert sm.update("cam1", "flooded").status == "flooded"


def test_accepts_prediction_like_object_via_status_attr():
    class FakePred:
        status = "flooded"

    sm = TemporalSmoother(n=1)
    assert sm.update("cam1", FakePred()).status == "flooded"


def test_serialization_round_trip():
    sm = TemporalSmoother(n=4, blocklist=["11372", "99"])
    sm.update("cam1", "flooded")
    sm.update("cam1", "flooded")
    d = sm.to_dict()
    sm2 = TemporalSmoother.from_dict(d)
    assert sm2.n == 4
    assert sm2.blocklist == {"11372", "99"}
    assert sm2.update("cam1", "flooded").status == "wet"
    assert sm2.update("cam1", "flooded").status == "flooded"


def test_smoothed_status_unpacks_as_tuple():
    status, note = SmoothedStatus("wet", "hello")
    assert status == "wet"
    assert note == "hello"
