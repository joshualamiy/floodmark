from floodmark_pipeline.alerting import AlertDecision, HistoryFrame, apply_rain_gate, decide_alert, water_streak

BLOCKLIST = frozenset({"11372"})


def frame(status: str, score: float = 0.5, sha: str | None = None, alert: str | None = None) -> HistoryFrame:
    return HistoryFrame(sha256=sha, status=status, alert_status=alert, flood_score=score)


def decide(status="flooded", score=0.5, sha="cur", history=(), baseline=None, n=3, margin=0.10, camera="cam1"):
    return decide_alert(
        camera_id=camera,
        status=status,
        flood_score=score,
        sha256=sha,
        history=list(history),
        baseline=baseline,
        streak_frames=n,
        baseline_margin=margin,
        blocklist=BLOCKLIST,
    )


def test_dry_passes_through_without_history():
    assert decide(status="dry", score=0.0) == AlertDecision("dry")


def test_first_water_frame_reports_potential_flooding():
    result = decide(history=[frame("dry", 0.0, "a")])
    assert result.status == "wet"
    assert result.note == "potential flooding; water frame 1/3; awaiting confirmation"
    assert not result.confirmed_flood


def test_three_flooded_frames_confirm_a_flood():
    history = [frame("flooded", 0.5, "b"), frame("flooded", 0.5, "a")]
    result = decide(history=history)
    assert result.status == "flooded"
    assert result.confirmed_flood


def test_three_wet_frames_stay_wet():
    history = [frame("wet", 0.2, "b"), frame("wet", 0.2, "a")]
    result = decide(status="wet", score=0.2, history=history)
    assert result == AlertDecision("wet")


def test_dry_frame_in_history_resets_the_streak():
    history = [frame("flooded", 0.5, "c"), frame("dry", 0.0, "b"), frame("flooded", 0.5, "a")]
    assert decide(history=history).note == "potential flooding; water frame 2/3; awaiting confirmation"


def test_frozen_repeat_of_previous_frame_is_not_counted():
    # same bytes as the last capture: a stuck feed cannot confirm itself
    history = [frame("flooded", 0.5, "same", alert="wet"), frame("flooded", 0.5, "a")]
    result = decide(sha="same", history=history)
    assert result.status == "wet"
    assert result.note == "frame identical to the previous capture; not counted toward confirmation"
    assert not result.confirmed_flood


def test_frozen_repeats_inside_history_are_skipped_not_reset():
    # a-b-b-c: the repeated b counts once, so the streak before the current frame is 2
    history = [frame("flooded", 0.5, "b"), frame("flooded", 0.5, "b"), frame("flooded", 0.5, "a")]
    assert water_streak(history, 0.0) == 2
    assert decide(sha="c", history=history).status == "flooded"


def test_frozen_frame_carries_previous_alert_status():
    history = [frame("flooded", 0.5, "same", alert="flooded")]
    assert decide(sha="same", history=history).status == "flooded"


def test_score_within_camera_baseline_does_not_count():
    # a camera whose scene always scores about 0.5 needs a clear jump above it
    history = [frame("flooded", 0.55, "b"), frame("flooded", 0.52, "a")]
    result = decide(score=0.55, history=history, baseline=0.50)
    assert result.status == "wet"
    assert "within this camera's normal range (median 0.50)" in result.note
    assert not result.confirmed_flood


def test_score_above_camera_baseline_counts_and_history_is_rechecked():
    history = [frame("flooded", 0.85, "b"), frame("flooded", 0.55, "a")]
    # the older 0.55 frame is within range, so only one prior frame counts
    assert decide(score=0.9, history=history, baseline=0.50).note.endswith("water frame 2/3; awaiting confirmation")
    history = [frame("flooded", 0.85, "b"), frame("flooded", 0.80, "a")]
    assert decide(score=0.9, history=history, baseline=0.50).status == "flooded"


def test_normal_camera_baseline_does_not_block_ordinary_floods():
    history = [frame("flooded", 0.4, "b"), frame("flooded", 0.4, "a")]
    assert decide(score=0.4, history=history, baseline=0.002).status == "flooded"


def test_blocklisted_camera_never_confirms():
    history = [frame("flooded", 0.9, "b"), frame("flooded", 0.9, "a")]
    result = decide(camera="11372", score=0.9, history=history)
    assert result == AlertDecision("wet", "camera blocklisted; flood alerts suppressed")
    assert decide(camera="11372", status="dry", score=0.0) == AlertDecision("dry")


def test_rain_gate_suppresses_confirmed_flood_without_rain():
    confirmed = AlertDecision("flooded", confirmed_flood=True)
    result = apply_rain_gate(confirmed, 0.0, min_rain_mm=1.0, window_hours=6, streak_frames=3)
    assert result.status == "wet"
    assert result.note == "flood signal in 3 frames but only 0.0 mm of rain in the last 6 h; alert suppressed"


def test_rain_gate_allows_confirmed_flood_with_rain():
    confirmed = AlertDecision("flooded", confirmed_flood=True)
    result = apply_rain_gate(confirmed, 12.4, min_rain_mm=1.0, window_hours=6, streak_frames=3)
    assert result.status == "flooded"
    assert result.note == "12.4 mm of rain in the last 6 h"


def test_rain_gate_fails_open_when_rain_is_unknown():
    confirmed = AlertDecision("flooded", confirmed_flood=True)
    result = apply_rain_gate(confirmed, None, min_rain_mm=1.0, window_hours=6, streak_frames=3)
    assert result.status == "flooded"
    assert result.note == "rain check unavailable; alert allowed"


def test_rain_gate_leaves_unconfirmed_decisions_alone():
    pending = AlertDecision("wet", "potential flooding; water frame 1/3; awaiting confirmation")
    assert apply_rain_gate(pending, 0.0, min_rain_mm=1.0, window_hours=6, streak_frames=3) == pending
