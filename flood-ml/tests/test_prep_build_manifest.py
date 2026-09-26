import csv

import pytest

pytest.importorskip("pandas")

from prep.build_manifest import (
    assert_legacy_test_rows_unchanged,
    backup_old_manifest,
    iowa_rwis_forced_splits,
    legacy_forced_splits,
)

OLD_COLUMNS = ["path", "label", "source", "split", "orig_path"]


def _write_old_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OLD_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def test_assert_legacy_test_rows_unchanged_passes_when_identical(tmp_path):
    old = tmp_path / "manifest.csv"
    _write_old_manifest(old, [
        {"path": "data/processed/images/fred/fred_000001.jpg", "label": "flooded",
         "source": "fred", "split": "test", "orig_path": "data/raw/fred/a.png"},
    ])
    new_rows = [
        {"path": "data/processed/images/fred/fred_000001.jpg", "label": "flooded",
         "source": "fred", "split": "test", "orig_path": "data/raw/fred/a.png"},
    ]
    result = assert_legacy_test_rows_unchanged(old, new_rows)
    assert result["n_mismatches"] == 0
    assert result["n_old_legacy_test_rows_checked"] == 1


def test_assert_legacy_test_rows_unchanged_tolerates_path_renumbering(tmp_path):
    old = tmp_path / "manifest.csv"
    _write_old_manifest(old, [
        {"path": "data/processed/images/fred/fred_000001.jpg", "label": "flooded",
         "source": "fred", "split": "test", "orig_path": "data/raw/fred/a.png"},
    ])
    new_rows = [
        {"path": "data/processed/images/fred/fred_000042.jpg", "label": "flooded",
         "source": "fred", "split": "test", "orig_path": "data/raw/fred/a.png"},
    ]
    result = assert_legacy_test_rows_unchanged(old, new_rows)
    assert result["n_mismatches"] == 0
    assert result["n_path_renumbered_same_label_split"] == 1


def test_assert_legacy_test_rows_unchanged_raises_on_label_change(tmp_path):
    old = tmp_path / "manifest.csv"
    _write_old_manifest(old, [
        {"path": "p.jpg", "label": "flooded", "source": "roadway_flooding",
         "split": "test", "orig_path": "data/raw/roadway_flooding/x.jpg"},
    ])
    new_rows = [
        {"path": "p.jpg", "label": "dry", "source": "roadway_flooding",
         "split": "test", "orig_path": "data/raw/roadway_flooding/x.jpg"},
    ]
    with pytest.raises(AssertionError):
        assert_legacy_test_rows_unchanged(old, new_rows)


def test_assert_legacy_test_rows_unchanged_also_catches_train_row_changes(tmp_path):
    old = tmp_path / "manifest.csv"
    _write_old_manifest(old, [
        {"path": "p.jpg", "label": "flooded", "source": "roadway_flooding",
         "split": "train", "orig_path": "data/raw/roadway_flooding/x.jpg"},
    ])
    new_rows = [
        {"path": "p.jpg", "label": "flooded", "source": "roadway_flooding",
         "split": "val", "orig_path": "data/raw/roadway_flooding/x.jpg"},
    ]
    with pytest.raises(AssertionError):
        assert_legacy_test_rows_unchanged(old, new_rows)


def test_assert_legacy_test_rows_unchanged_raises_on_missing_row(tmp_path):
    old = tmp_path / "manifest.csv"
    _write_old_manifest(old, [
        {"path": "p.jpg", "label": "wet", "source": "nysdot_road_surface",
         "split": "test", "orig_path": "data/raw/nysdot_road_surface/x.jpg"},
    ])
    with pytest.raises(AssertionError):
        assert_legacy_test_rows_unchanged(old, [])


def test_assert_legacy_test_rows_unchanged_skips_when_no_old_manifest(tmp_path):
    result = assert_legacy_test_rows_unchanged(tmp_path / "missing.csv", [])
    assert "skipped" in result


def test_backup_old_manifest_copies_once(tmp_path):
    old = tmp_path / "manifest.csv"
    old.write_text("a,b\n1,2\n")
    backup = tmp_path / "manifest_v1-old.csv"

    note = backup_old_manifest(old, backup)
    assert backup.exists()
    assert backup.read_text() == old.read_text()
    assert "copied" in note

    old.write_text("a,b\n999,999\n")
    note2 = backup_old_manifest(old, backup)
    assert "already exists" in note2
    assert backup.read_text() == "a,b\n1,2\n"


def test_legacy_forced_splits_pins_old_groups_by_orig_path(tmp_path):
    old = tmp_path / "manifest.csv"
    _write_old_manifest(old, [
        {"path": "old1.jpg", "label": "flooded", "source": "roadway_flooding",
         "split": "test", "orig_path": "data/raw/roadway_flooding/a.jpg"},
        {"path": "old2.jpg", "label": "flooded", "source": "roadway_flooding",
         "split": "train", "orig_path": "data/raw/roadway_flooding/b.jpg"},
    ])
    ext_rows = [
        {"source": "roadway_flooding", "orig_path": "data/raw/roadway_flooding/a.jpg", "group_id": "dc0009999"},
        {"source": "roadway_flooding", "orig_path": "data/raw/roadway_flooding/b.jpg", "group_id": "dc0009998"},
        {"source": "eu_flood_2013", "orig_path": "data/raw/eu_flood_2013/c.jpg", "group_id": "euflood_user_x"},
    ]
    forced = legacy_forced_splits(ext_rows, old_manifest_path=old)
    assert forced == {"dc0009999": "test", "dc0009998": "train"}


def test_legacy_forced_splits_resolves_merged_cluster_test_over_train(tmp_path):
    old = tmp_path / "manifest.csv"
    _write_old_manifest(old, [
        {"path": "old1.jpg", "label": "flooded", "source": "roadway_flooding",
         "split": "test", "orig_path": "data/raw/roadway_flooding/a.jpg"},
        {"path": "old2.jpg", "label": "flooded", "source": "roadway_flooding",
         "split": "train", "orig_path": "data/raw/roadway_flooding/b.jpg"},
    ])
    ext_rows = [
        {"source": "roadway_flooding", "orig_path": "data/raw/roadway_flooding/a.jpg", "group_id": "dc0000001"},
        {"source": "roadway_flooding", "orig_path": "data/raw/roadway_flooding/b.jpg", "group_id": "dc0000001"},
    ]
    forced = legacy_forced_splits(ext_rows, old_manifest_path=old)
    assert forced == {"dc0000001": "test"}


def test_iowa_rwis_forced_splits_pins_mixed_cameras_to_test_and_val():
    ext_rows = []
    mixed_cams = [f"IDOT-{i:03d}" for i in range(20)]
    for cam in mixed_cams:
        ext_rows.append({"source": "iowa_rwis", "camera_id": cam, "label": "wet"})
        ext_rows.append({"source": "iowa_rwis", "camera_id": cam, "label": "dry"})
    ext_rows.append({"source": "iowa_rwis", "camera_id": "IDOT-999", "label": "wet"})

    forced = iowa_rwis_forced_splits(ext_rows, n_test=6, n_val=6)
    assert sum(1 for v in forced.values() if v == "test") == 6
    assert sum(1 for v in forced.values() if v == "val") == 6
    assert "IDOT-999" not in forced
    assert set(forced.keys()) <= set(mixed_cams)


def test_iowa_rwis_forced_splits_is_deterministic():
    ext_rows = []
    for i in range(10):
        cam = f"IDOT-{i:03d}"
        ext_rows.append({"source": "iowa_rwis", "camera_id": cam, "label": "wet"})
        ext_rows.append({"source": "iowa_rwis", "camera_id": cam, "label": "dry"})
    first = iowa_rwis_forced_splits(ext_rows, n_test=2, n_val=2)
    second = iowa_rwis_forced_splits(ext_rows, n_test=2, n_val=2)
    assert first == second

