from datetime import datetime, timezone
from pathlib import Path

from floodmark_pipeline.contracts import CaptureJob
from floodmark_pipeline.demo import demo_note, frame_source_url, is_demo
from floodmark_pipeline.demo_replay import Replay, list_frames


def job(source: str) -> CaptureJob:
    return CaptureJob(
        camera_id="uuid",
        source=source,
        source_camera_id="demo-1",
        source_view_id="demo-1",
        scheduled_at=datetime(2026, 9, 26, 14, 20, tzinfo=timezone.utc),
        capture_id="capture",
    )


def test_demo_cameras_are_fetched_from_the_replay_server():
    assert is_demo(job("DEMO")) and is_demo(job("demo"))
    assert not is_demo(job("SKYLINE"))
    assert frame_source_url(job("DEMO"), "https://511ga.org/map/Cctv", "http://demo:8080") == "http://demo:8080/demo-1"
    assert (
        frame_source_url(job("SKYLINE"), "https://511ga.org/map/Cctv", "http://demo:8080")
        == "https://511ga.org/map/Cctv/demo-1"
    )


def test_demo_note_is_appended_to_every_alert_note():
    assert demo_note(None) == "demo replay with simulated storm"
    assert demo_note("potential flooding; water frame 1/2; awaiting confirmation") == (
        "potential flooding; water frame 1/2; awaiting confirmation; demo replay with simulated storm"
    )


def write_frames(root: Path, view_id: str, names: list[str]) -> None:
    folder = root / view_id
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_bytes(name.encode())


def test_replay_plays_frames_in_name_order_and_holds_the_last_one(tmp_path: Path):
    write_frames(tmp_path, "demo-1", ["02_dry.jpg", "01_dry.jpg", "03_flood.png", "notes.txt"])
    replay = Replay(tmp_path)

    assert [path.name for path in list_frames(tmp_path, "demo-1")] == ["01_dry.jpg", "02_dry.jpg", "03_flood.png"]
    assert [replay.next_frame("demo-1").name for _ in range(5)] == [
        "01_dry.jpg", "02_dry.jpg", "03_flood.png", "03_flood.png", "03_flood.png",
    ]
    assert replay.status() == {"demo-1": {"served": 3, "frames": 3}}

    replay.reset("demo-1")
    assert replay.next_frame("demo-1").name == "01_dry.jpg"


def test_replay_returns_none_for_unknown_or_empty_view(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    replay = Replay(tmp_path)
    assert replay.next_frame("missing") is None
    assert replay.next_frame("empty") is None
