from pathlib import Path

from ga511.collect import FRAME_CSV_FIELDS as GA511_FIELDS
from othercams.schema import FRAME_CSV_FIELDS, append_frame_rows, read_frame_rows


def test_schema_matches_ga511():
    assert FRAME_CSV_FIELDS == GA511_FIELDS


def test_append_and_read_round_trip(tmp_path: Path):
    csv_path = tmp_path / "frames.csv"
    lock_path = tmp_path / "frames.csv.lock"
    rows = [
        {"frame_id": "a_1", "camera_id": "IDOT-000", "view_id": "IDOT-000-01",
         "weak_label": "likely_wet", "dead_reason": ""},
        {"frame_id": "a_2", "camera_id": "IDOT-000", "view_id": "IDOT-000-01",
         "weak_label": "likely_dry", "dead_reason": ""},
    ]
    append_frame_rows(rows, csv_path, lock_path)
    got = read_frame_rows(csv_path)
    assert [r["frame_id"] for r in got] == ["a_1", "a_2"]
    assert list(got[0].keys()) == FRAME_CSV_FIELDS

    append_frame_rows([{"frame_id": "a_3"}], csv_path, lock_path)
    assert len(read_frame_rows(csv_path)) == 3


def test_read_missing_file_returns_empty(tmp_path: Path):
    assert read_frame_rows(tmp_path / "nope.csv") == []


def test_append_deduplicates_existing_and_batch_ids(tmp_path):
    path, lock = tmp_path / "frames.csv", tmp_path / "frames.lock"
    append_frame_rows([{"frame_id": "a"}], path, lock)
    append_frame_rows([{"frame_id": "a"}, {"frame_id": "b"}, {"frame_id": "b"}], path, lock)
    assert [r["frame_id"] for r in read_frame_rows(path)] == ["a", "b"]

