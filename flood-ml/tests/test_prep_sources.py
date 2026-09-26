import csv

from prep.sources import (
    iter_alleyfloodnet,
    iter_eu_flood_2013,
    iter_ga511,
    iter_iowa_rwis,
)

FIELDS = ["frame_id", "path", "camera_id", "dead_reason", "weak_label"]


def _frames(tmp_path, monkeypatch, rows):
    monkeypatch.chdir(tmp_path)
    for r in rows:
        img = tmp_path / "data" / "ga511" / r["path"]
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"x")
    csv_path = tmp_path / "frames.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return csv_path


def _row(fid, weak):
    return {"frame_id": fid, "path": f"frames/1/{fid}.jpg", "camera_id": "1",
            "dead_reason": "", "weak_label": weak}


def test_ai_wet_conflicting_with_likely_dry_is_unlabeled(tmp_path, monkeypatch):
    p = _frames(tmp_path, monkeypatch, [_row("a", "likely_dry")])
    out = iter_ga511(p, {}, {"a": {"label": "wet", "note": ""}})
    assert out[0]["label"] is None
    assert "conflicts" in out[0]["notes"]


def test_ai_wet_without_conflict_is_kept(tmp_path, monkeypatch):
    p = _frames(tmp_path, monkeypatch, [_row("a", "uncertain")])
    out = iter_ga511(p, {}, {"a": {"label": "wet", "note": ""}})
    assert (out[0]["label"], out[0]["label_source"]) == ("wet", "ai_review")


def test_manual_label_overrides_everything(tmp_path, monkeypatch):
    p = _frames(tmp_path, monkeypatch, [_row("a", "likely_dry")])
    out = iter_ga511(p, {"a": {"label": "wet"}}, {"a": {"label": "dry"}})
    assert (out[0]["label"], out[0]["label_source"]) == ("wet", "manual")


def test_ai_dry_on_likely_dry_is_kept(tmp_path, monkeypatch):
    p = _frames(tmp_path, monkeypatch, [_row("a", "likely_dry")])
    out = iter_ga511(p, {}, {"a": {"label": "dry", "note": ""}})
    assert (out[0]["label"], out[0]["label_source"]) == ("dry", "ai_review")


IOWA_FIELDS = ["frame_id", "path", "camera_id", "dead_reason", "weak_label"]


def _iowa_setup(tmp_path, monkeypatch, frame_rows, label_rows):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "othercams" / "iowa_rwis"
    for r in frame_rows:
        img = root / r["path"]
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"x")
    with open(root / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=IOWA_FIELDS)
        w.writeheader()
        w.writerows(frame_rows)
    with open(root / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "label", "labeled_at"])
        w.writeheader()
        w.writerows(label_rows)


def test_iowa_rwis_only_manual_labels_never_weak_label(tmp_path, monkeypatch):
    frames = [
        {"frame_id": "a", "path": "frames/cam1/a.jpg", "camera_id": "cam1", "dead_reason": "", "weak_label": "likely_wet"},
    ]
    _iowa_setup(tmp_path, monkeypatch, frames, [])
    out = iter_iowa_rwis()
    assert out == []


def test_iowa_rwis_unusable_is_dropped_and_last_write_wins(tmp_path, monkeypatch):
    frames = [
        {"frame_id": "a", "path": "frames/cam1/a.jpg", "camera_id": "cam1", "dead_reason": "", "weak_label": ""},
        {"frame_id": "b", "path": "frames/cam1/b.jpg", "camera_id": "cam1", "dead_reason": "", "weak_label": ""},
    ]
    labels = [
        {"frame_id": "a", "label": "unusable", "labeled_at": "t0"},
        {"frame_id": "b", "label": "dry", "labeled_at": "t0"},
        {"frame_id": "b", "label": "wet", "labeled_at": "t1"},
    ]
    _iowa_setup(tmp_path, monkeypatch, frames, labels)
    out = {r["frame_id"]: r for r in iter_iowa_rwis()}
    assert set(out.keys()) == {"b"}
    assert out["b"]["label"] == "wet"
    assert out["b"]["label_source"] == "manual"
    assert out["b"]["camera_id"] == out["b"]["group_id"] == "cam1"
    assert out["b"]["restricted"] is False
    assert "Public domain" in out["b"]["license"]


def test_iowa_rwis_dead_frame_dropped(tmp_path, monkeypatch):
    frames = [
        {"frame_id": "a", "path": "frames/cam1/a.jpg", "camera_id": "cam1", "dead_reason": "frozen_repeat", "weak_label": ""},
    ]
    _iowa_setup(tmp_path, monkeypatch, frames, [{"frame_id": "a", "label": "dry", "labeled_at": "t0"}])
    assert iter_iowa_rwis() == []


EU_FIELDS = ["path", "label", "label_source", "road_score", "view_guess", "group_id", "license", "phash"]


def _eu_setup(tmp_path, monkeypatch, rows):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "raw" / "eu_flood_2013"
    root.mkdir(parents=True, exist_ok=True)
    for r in rows:
        img = tmp_path / r["path"]
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"x")
    with open(root / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EU_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _eu_row(path, label, road_score, license_str="CC BY-SA 3.0", group="euflood_user_x"):
    return {
        "path": f"data/raw/eu_flood_2013/images/{path}", "label": label, "label_source": "dataset_label",
        "road_score": str(road_score), "view_guess": "unknown", "group_id": group,
        "license": license_str, "phash": "0" * 16,
    }


def test_eu_flood_drops_unverified_license_and_unknown_label(tmp_path, monkeypatch):
    rows = [
        _eu_row("a.jpg", "flooded", 0.05),
        _eu_row("b.jpg", "unknown", 0.05),
        _eu_row("c.jpg", "flooded", 0.05, license_str="unverified (manually harvested search-engine image, not Wikimedia)"),
    ]
    _eu_setup(tmp_path, monkeypatch, rows)
    out = iter_eu_flood_2013()
    assert [r["orig_path"].endswith("a.jpg") for r in out] == [True]


def test_eu_flood_road_score_threshold_and_label_mapping(tmp_path, monkeypatch):
    rows = [
        _eu_row("below.jpg", "flooded", -0.02),
        _eu_row("above_flooded.jpg", "flooded", 0.02),
        _eu_row("above_notflooded.jpg", "not_flooded", 0.02),
    ]
    _eu_setup(tmp_path, monkeypatch, rows)
    out = {r["orig_path"].rsplit("/", 1)[-1]: r for r in iter_eu_flood_2013(road_score_threshold=0.0)}
    assert set(out.keys()) == {"above_flooded.jpg", "above_notflooded.jpg"}
    assert out["above_flooded.jpg"]["label"] == "flooded"
    assert out["above_notflooded.jpg"]["label"] == "not_flooded"
    assert out["above_flooded.jpg"]["view_type"] == "ground"
    assert out["above_flooded.jpg"]["restricted"] is False


def test_eu_flood_group_id_is_uploader(tmp_path, monkeypatch):
    rows = [_eu_row("a.jpg", "flooded", 0.5, group="euflood_user_Someone")]
    _eu_setup(tmp_path, monkeypatch, rows)
    out = iter_eu_flood_2013()
    assert out[0]["group_id"] == "euflood_user_Someone"


ALLEY_FIELDS = ["path", "label", "label_source", "road_score", "view_guess", "group_id", "license", "phash"]


def _alley_setup(tmp_path, monkeypatch, rows):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "raw" / "alleyfloodnet"
    root.mkdir(parents=True, exist_ok=True)
    for r in rows:
        img = tmp_path / r["path"]
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"x")
    with open(root / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALLEY_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _alley_row(path, label):
    return {
        "path": f"data/raw/alleyfloodnet/{path}", "label": label, "label_source": "dataset_label",
        "road_score": "0.1", "view_guess": "unknown", "group_id": "alleyfloodnet_unknown",
        "license": "CC BY 4.0 (dataset-level grant; per-photo origin unverified -- see report)",
        "phash": "0" * 16,
    }


def test_alleyfloodnet_labels_and_no_pregrouping(tmp_path, monkeypatch):
    rows = [_alley_row("flooded/a.jpg", "flooded"), _alley_row("non_flooded/b.jpg", "not_flooded")]
    _alley_setup(tmp_path, monkeypatch, rows)
    out = iter_alleyfloodnet(include=True)
    assert {r["label"] for r in out} == {"flooded", "not_flooded"}
    assert all(r["group_id"] is None for r in out)
    assert all("CC BY 4.0" in r["license"] for r in out)


def test_alleyfloodnet_include_false_drops_source_entirely(tmp_path, monkeypatch):
    rows = [_alley_row("flooded/a.jpg", "flooded")]
    _alley_setup(tmp_path, monkeypatch, rows)
    assert iter_alleyfloodnet(include=False) == []

