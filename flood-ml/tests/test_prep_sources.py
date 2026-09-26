"""iter_ga511 label priority: manual > ai_review > weak_precip, and an
ai_review wet/flooded that contradicts a likely_dry weather label is dropped."""
import csv

from prep.sources import iter_ga511

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
