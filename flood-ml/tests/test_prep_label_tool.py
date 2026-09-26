"""Unit tests for prep.label_tool: the server starts, serves the page and
the frames API, and a POST to /api/label appends a row to labels.csv with
last-write-wins semantics. Uses a temp dir standing in for data/ga511/, and
a real (but ephemeral, 127.0.0.1:0) HTTP server -- no external network.
"""
import csv
import json
import threading
import urllib.error
import urllib.request

import pytest

from prep.label_tool import create_server


@pytest.fixture
def ga511_root(tmp_path):
    root = tmp_path / "ga511"
    root.mkdir()
    frames_csv = root / "frames.csv"
    with open(frames_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_id", "path", "camera_id", "view_id", "lat", "lon", "roadway",
            "county", "timestamp_utc", "http_status", "bytes", "width", "height",
            "phash", "dead_reason", "weak_label", "precip_1h_mm", "precip_3h_mm",
            "precip_source",
        ])
        w.writerow([
            "f1", "frames/v1/1.jpg", "cam1", "v1", "33.8", "-84.3", "SR1",
            "Fulton", "2026-01-01T00:00:00Z", "200", "1000", "640", "480",
            "abc", "", "likely_dry", "0.0", "0.0", "test",
        ])
        w.writerow([
            "f2", "frames/v2/2.jpg", "cam2", "v2", "33.9", "-84.2", "SR2",
            "DeKalb", "2026-01-01T00:00:00Z", "200", "1000", "640", "480",
            "def", "", "likely_wet", "1.0", "2.0", "test",
        ])
    # one real (tiny) jpeg-ish file per frame so list_frames() finds them
    for sub, name in (("v1", "1.jpg"), ("v2", "2.jpg")):
        d = root / "frames" / sub
        d.mkdir(parents=True)
        (d / name).write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    return root


@pytest.fixture
def server(ga511_root):
    srv = create_server(ga511_root, host="127.0.0.1", port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _url(server, path):
    return f"http://127.0.0.1:{server.server_port}{path}"


def test_server_serves_index_page(server):
    with urllib.request.urlopen(_url(server, "/")) as resp:
        assert resp.status == 200
        body = resp.read().decode("utf-8")
    assert "<html>" in body.lower()


def test_frames_api_returns_unlabeled_frames(server):
    with urllib.request.urlopen(_url(server, "/api/frames?filter=unlabeled")) as resp:
        frames = json.loads(resp.read())
    frame_ids = {f["frame_id"] for f in frames}
    assert frame_ids == {"f1", "f2"}


def test_frames_api_likely_wet_filter(server):
    with urllib.request.urlopen(_url(server, "/api/frames?filter=likely_wet")) as resp:
        frames = json.loads(resp.read())
    assert [f["frame_id"] for f in frames] == ["f2"]


def test_post_label_appends_row_to_labels_csv(server, ga511_root):
    payload = json.dumps({"frame_id": "f1", "label": "dry"}).encode("utf-8")
    req = urllib.request.Request(
        _url(server, "/api/label"), data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    assert result["ok"] is True

    labels_csv = ga511_root / "labels.csv"
    assert labels_csv.exists()
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["frame_id"] == "f1"
    assert rows[0]["label"] == "dry"
    assert rows[0]["labeled_at"]


def test_post_label_last_write_wins(server, ga511_root):
    for label in ("dry", "wet", "flooded"):
        payload = json.dumps({"frame_id": "f1", "label": label}).encode("utf-8")
        req = urllib.request.Request(
            _url(server, "/api/label"), data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req).read()

    labels_csv = ga511_root / "labels.csv"
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # append-only on disk

    from prep.sources import load_labels_csv
    reduced = load_labels_csv(labels_csv)
    assert reduced["f1"]["label"] == "flooded"  # last write wins


def test_post_label_rejects_invalid_label(server):
    payload = json.dumps({"frame_id": "f1", "label": "not_a_real_label"}).encode("utf-8")
    req = urllib.request.Request(
        _url(server, "/api/label"), data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        urllib.request.urlopen(req)
        raised = False
    except urllib.error.HTTPError as e:
        raised = e.code == 400
    assert raised


def test_frames_api_disputed_filter(server, ga511_root):
    # f1 is likely_dry by weather but ai_review called it wet -> disputed.
    # f2 is likely_wet, so an ai wet on it is not a dispute.
    with open(ga511_root / "ai_review_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "label", "note"])
        w.writerow(["f1", "wet", ""])
        w.writerow(["f2", "wet", ""])
    with urllib.request.urlopen(_url(server, "/api/frames?filter=disputed")) as resp:
        frames = json.loads(resp.read())
    assert [f["frame_id"] for f in frames] == ["f1"]
