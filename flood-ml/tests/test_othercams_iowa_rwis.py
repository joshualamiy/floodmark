import logging
import os

import pytest

from othercams import iowa_rwis as m
from othercams import jobs

FIXTURE_HTML = """
<select name='cid'><option value='IDOT-000-02'>Adair I80 Bridge Deck -- (2009-12-20)</option>
<option value='IDOT-000-03'>Adair I80 East -- (2009-12-20)</option>
<option value='IDOT-088-01'>Ainsworth -- (2024-11-13)</option>
<option value='-1'>Select a camera first...</option>
</select>
"""

FIXTURE_SITES = {
    "features": [
        {"properties": {"sid": "X1", "sname": "Adair (I-80)", "county": "Cass"},
         "geometry": {"coordinates": [-94.7263, 41.4963]}},
        {"properties": {"sid": "X2", "sname": "Ainsworth", "county": "Washington"},
         "geometry": {"coordinates": [-91.5332, 41.2913]}},
    ]
}


def test_parse_camera_options():
    opts = m.parse_camera_options(FIXTURE_HTML)
    assert len(opts) == 3  # the "-1" placeholder option is not matched
    assert opts[0] == {
        "view_id": "IDOT-000-02", "group": "IDOT-000",
        "label": "Adair I80 Bridge Deck", "first_seen": "2009-12-20",
    }
    assert {o["group"] for o in opts} == {"IDOT-000", "IDOT-088"}


def test_parse_site_features():
    sites = m.parse_site_features(FIXTURE_SITES)
    assert len(sites) == 2
    assert sites[0]["sid"] == "X1"
    assert sites[0]["lat"] == 41.4963
    assert sites[0]["lon"] == -94.7263


def test_match_group_to_site_by_town_name():
    sites = m.parse_site_features(FIXTURE_SITES)
    site, score = m.match_group_to_site("Ainsworth", sites)
    assert site["sid"] == "X2"
    assert score >= 0.5


def test_match_group_to_site_no_match():
    sites = m.parse_site_features(FIXTURE_SITES)
    site, score = m.match_group_to_site("Zzyzx Nowhere Junction", sites)
    assert site is None
    assert score == 0.0


def test_parse_roadway():
    assert m.parse_roadway("Ames (I-35) N/B") == "I-35"
    assert m.parse_roadway("Anamosa (US 151) EB Bridge") == "US 151"
    assert m.parse_roadway("Ainsworth") == ""


def test_choose_preferred_view_avoids_bridge_deck_and_zoom():
    opts = [
        {"view_id": "IDOT-000-02", "label": "Adair I80 Bridge Deck"},
        {"view_id": "IDOT-000-03", "label": "Adair I80 East"},
        {"view_id": "IDOT-000-01", "label": "Adair I80 West Zoom"},
    ]
    chosen = m.choose_preferred_view(opts)
    assert chosen["view_id"] == "IDOT-000-03"


def test_build_camera_table_matches_and_picks_view():
    opts = m.parse_camera_options(FIXTURE_HTML)
    sites = m.parse_site_features(FIXTURE_SITES)
    rows = m.build_camera_table(opts, sites)
    by_group = {r["camera_id"]: r for r in rows}
    assert by_group["IDOT-000"]["view_id"] == "IDOT-000-03"  # not the bridge deck
    assert by_group["IDOT-000"]["lat"] == 41.4963
    assert by_group["IDOT-088"]["county"] == "Washington"


def test_pick_target_days_wet_and_dry():
    series = [
        ("2026-01-15", 15.0),  # winter: excluded from wet months
        ("2026-06-01", 0.0),
        ("2026-06-02", 0.0),
        ("2026-06-03", 0.0),  # dry: 2 prior days also 0
        ("2026-06-04", 12.0),  # wet: >= WET_DAY_MIN_MM
        ("2026-06-05", 4.0),  # also wet, less than 06-04
    ]
    picked = m.pick_target_days(series)
    assert picked["wet"] == ["2026-06-04", "2026-06-05"]  # wettest first
    assert picked["dry"] == ["2026-06-03"]


def test_pick_target_days_requires_prior_days_present():
    # 06-01 has no data for 05-30/05-31, so it must not count as a safe dry day
    series = [("2026-06-01", 0.0)]
    picked = m.pick_target_days(series)
    assert picked["dry"] == []


def test_is_daytime():
    assert m.is_daytime("2026-06-04T15:00:00Z") is True
    assert m.is_daytime("2026-06-04T04:00:00Z") is False


def test_subsample_keeps_all_when_fewer_than_k():
    items = [1, 2, 3]
    assert m.subsample(items, 6) == items


def test_subsample_spreads_across_range():
    items = list(range(24))
    out = m.subsample(items, 6)
    assert len(out) == 6
    assert out[0] == 0
    assert out == sorted(out)


def test_valid_to_ts():
    ts = m._valid_to_ts("2024-05-07T05:03:00Z")
    assert ts == 1715058180


def test_resume_completed_camera_makes_no_requests(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("completed camera must not fetch")

    monkeypatch.setattr(m, "daily_precip", unexpected)
    existing = [{"frame_id": "saved", "weak_label": "likely_wet", "dead_reason": ""}]
    assert m.collect_group({}, 1, 0, 5, 3, {}, logging.getLogger(), existing) == []


def test_saves_each_frame_before_a_later_interruption(monkeypatch):
    from ga511 import weather

    cam = {"camera_id": "IDOT-000", "view_id": "IDOT-000-01", "lat": 41, "lon": -94}
    dates = [f"2026-06-04T{hour}:00:00Z" for hour in (13, 14, 15)]
    existing = [{"frame_id": f"IDOT-000-01_{m._valid_to_ts(dates[0])}",
                 "weak_label": "likely_wet", "dead_reason": ""}]
    monkeypatch.setattr(m, "daily_precip", lambda *a: [])
    monkeypatch.setattr(m, "pick_target_days", lambda *a: {"wet": ["2026-06-04"], "dry": []})
    monkeypatch.setattr(m, "list_archive_images", lambda *a: [
        {"valid": d, "href": "https://example.invalid"} for d in dates])
    monkeypatch.setattr(m, "_pace", lambda: None)
    monkeypatch.setattr(weather, "get_precip_and_label", lambda *a: {"weak_label": "likely_wet"})
    fetched, saved = [], []

    def fetch(cam, ts, *args):
        fetched.append(ts)
        if len(fetched) == 2:
            raise RuntimeError("interrupted")
        return {"frame_id": f"{cam['view_id']}_{ts}", "weak_label": "likely_wet", "dead_reason": ""}

    monkeypatch.setattr(m, "fetch_and_build_row", fetch)
    with pytest.raises(RuntimeError, match="interrupted"):
        m.collect_group(cam, 3, 0, 5, 3, {}, logging.getLogger(), existing, saved.append)
    assert len(saved) == 1
    assert fetched[0] == m._valid_to_ts(dates[1])


def test_collection_lock_rejects_overlap_and_cleans_up(tmp_path):
    pid = tmp_path / "collect.pid"
    with jobs.collection_lock(pid):
        assert jobs.read_pid(pid) == os.getpid()
        with pytest.raises(RuntimeError, match="already running"), jobs.collection_lock(pid):
            pass
    assert not pid.exists()


def test_collection_lock_preserves_existing_collector(tmp_path, monkeypatch):
    pid = tmp_path / "collect.pid"
    pid.write_text("123456")
    monkeypatch.setattr(jobs, "pid_alive", lambda p: p == 123456)
    with pytest.raises(RuntimeError, match="already running"), jobs.collection_lock(pid):
        pass
    assert jobs.read_pid(pid) == 123456
