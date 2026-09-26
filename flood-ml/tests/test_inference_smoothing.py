"""TemporalSmoother: N-consecutive, skip semantics, blocklist, serialization."""
from __future__ import annotations

from inference.smoothing import SmoothedStatus, TemporalSmoother


def test_reports_wet_until_n_consecutive_flooded():
    sm = TemporalSmoother(n=3)
    assert sm.update("cam1", "flooded").status == "wet"
    assert sm.update("cam1", "flooded").status == "wet"
    assert sm.update("cam1", "flooded").status == "flooded"


def test_non_flooded_frame_resets_streak():
    sm = TemporalSmoother(n=3)
    sm.update("cam1", "flooded")
    sm.update("cam1", "flooded")
    assert sm.update("cam1", "dry").status == "dry"
    assert sm.update("cam1", "flooded").status == "wet"  # streak restarted


def test_dry_and_wet_pass_through_unsmoothed():
    sm = TemporalSmoother(n=3)
    assert sm.update("cam1", "dry").status == "dry"
    assert sm.update("cam1", "wet").status == "wet"


def test_skip_does_not_count_or_reset_streak():
    sm = TemporalSmoother(n=3)
    sm.update("cam1", "flooded")
    sm.skip("cam1")
    sm.skip("cam1")
    assert sm.update("cam1", "flooded").status == "wet"  # streak now 2, not reset by skips
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
    # streak state carried over: 2 more flooded frames should now complete n=4
    assert sm2.update("cam1", "flooded").status == "wet"
    assert sm2.update("cam1", "flooded").status == "flooded"


def test_smoothed_status_unpacks_as_tuple():
    status, note = SmoothedStatus("wet", "hello")
    assert status == "wet"
    assert note == "hello"
