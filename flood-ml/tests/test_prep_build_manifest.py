import csv

import pytest

pytest.importorskip("pandas")

from prep.build_manifest import (
    assert_legacy_test_rows_unchanged,
    assert_v4_rebuild_invariants,
    backup_old_manifest,
    iowa_rwis_forced_splits,
    legacy_forced_splits,
    report_ga511_daytime,
    thin_and_dedup,
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


def test_assert_v4_rebuild_invariants_passes_when_identical(tmp_path):
    ref = tmp_path / "manifest_v1-7251bbd2.csv"
    _write_old_manifest(ref, [
        {"path": "p1.jpg", "label": "flooded", "source": "eu_flood_2013",
         "split": "train", "orig_path": "data/raw/eu_flood_2013/a.jpg"},
        {"path": "p2.jpg", "label": "dry", "source": "ga511",
         "split": "test", "orig_path": "data/ga511/frames/1/1.jpg"},
    ])
    new_rows = [
        {"path": "p1.jpg", "label": "flooded", "source": "eu_flood_2013",
         "split": "train", "orig_path": "data/raw/eu_flood_2013/a.jpg"},
        {"path": "p2.jpg", "label": "dry", "source": "ga511",
         "split": "test", "orig_path": "data/ga511/frames/1/1.jpg"},
        {"path": "p3.jpg", "label": "dry", "source": "ga511",
         "split": "train", "orig_path": "data/ga511/frames/1/2.jpg"},
    ]
    result = assert_v4_rebuild_invariants(ref, new_rows)
    assert result["n_mismatches"] == 0
    assert result["n_missing"] == 0


def test_assert_v4_rebuild_invariants_raises_on_non_ga511_change(tmp_path):
    ref = tmp_path / "manifest_v1-7251bbd2.csv"
    _write_old_manifest(ref, [
        {"path": "p1.jpg", "label": "flooded", "source": "eu_flood_2013",
         "split": "train", "orig_path": "data/raw/eu_flood_2013/a.jpg"},
    ])
    new_rows = [
        {"path": "p1.jpg", "label": "not_flooded", "source": "eu_flood_2013",
         "split": "train", "orig_path": "data/raw/eu_flood_2013/a.jpg"},
    ]
    with pytest.raises(AssertionError):
        assert_v4_rebuild_invariants(ref, new_rows)


def test_assert_v4_rebuild_invariants_raises_on_ga511_test_change(tmp_path):
    ref = tmp_path / "manifest_v1-7251bbd2.csv"
    _write_old_manifest(ref, [
        {"path": "p2.jpg", "label": "dry", "source": "ga511",
         "split": "test", "orig_path": "data/ga511/frames/1/1.jpg"},
    ])
    new_rows = [
        {"path": "p2.jpg", "label": "flooded", "source": "ga511",
         "split": "test", "orig_path": "data/ga511/frames/1/1.jpg"},
    ]
    with pytest.raises(AssertionError):
        assert_v4_rebuild_invariants(ref, new_rows)


def test_assert_v4_rebuild_invariants_raises_on_new_ga511_test_row(tmp_path):
    ref = tmp_path / "manifest_v1-7251bbd2.csv"
    _write_old_manifest(ref, [
        {"path": "p2.jpg", "label": "dry", "source": "ga511",
         "split": "test", "orig_path": "data/ga511/frames/1/1.jpg"},
    ])
    new_rows = [
        {"path": "p2.jpg", "label": "dry", "source": "ga511",
         "split": "test", "orig_path": "data/ga511/frames/1/1.jpg"},
        {"path": "p9.jpg", "label": "dry", "source": "ga511",
         "split": "test", "orig_path": "data/ga511/frames/2/9.jpg"},
    ]
    with pytest.raises(AssertionError):
        assert_v4_rebuild_invariants(ref, new_rows)


def test_assert_v4_rebuild_invariants_allows_new_ga511_train_val_rows(tmp_path):
    ref = tmp_path / "manifest_v1-7251bbd2.csv"
    _write_old_manifest(ref, [
        {"path": "p2.jpg", "label": "dry", "source": "ga511",
         "split": "test", "orig_path": "data/ga511/frames/1/1.jpg"},
    ])
    new_rows = [
        {"path": "p2.jpg", "label": "dry", "source": "ga511",
         "split": "test", "orig_path": "data/ga511/frames/1/1.jpg"},
        {"path": "p10.jpg", "label": "not_flooded", "source": "ga511",
         "split": "val", "orig_path": "data/ga511/frames/2/10.jpg"},
    ]
    result = assert_v4_rebuild_invariants(ref, new_rows)
    assert result["n_mismatches"] == 0


def test_assert_v4_rebuild_invariants_skips_when_no_reference(tmp_path):
    result = assert_v4_rebuild_invariants(tmp_path / "missing.csv", [])
    assert "skipped" in result


# 2026-09-26 12:00:00 America/New_York
DAY_PATH = "data/ga511/frames/1/1790438400.jpg"
# 2026-09-26 02:00:00 America/New_York (night, same date)
NIGHT_PATH = "data/ga511/frames/1/1790402400.jpg"


def test_report_ga511_daytime_counts_by_split_and_thinned():
    pre = [
        {"source": "ga511", "orig_path": DAY_PATH, "label": "dry", "split": "train", "label_source": "weak_precip"},
        {"source": "ga511", "orig_path": "data/ga511/frames/2/1790438401.jpg", "label": "not_flooded",
         "split": "val", "label_source": "manual"},
        {"source": "ga511", "orig_path": NIGHT_PATH, "label": "dry", "split": "train", "label_source": "manual"},
    ]
    # simulate one daytime row dropped by dedup/thinning downstream
    final = [pre[1]]
    report = report_ga511_daytime(pre, final)
    assert report["n_labeled_pre_dedup"] == 2
    assert report["n_final"] == 1
    assert report["n_thinned_or_deduped"] == 1
    assert report["by_split"] == {"val": 1}


def test_thin_and_dedup_keeps_disjoint_camera_network_phash_collisions():
    # ga511 (Georgia) and iowa_rwis (Iowa) are disjoint physical camera networks; a phash
    # collision between them is noise, not a real duplicate photo -- both must survive
    rows = [
        {"source": "ga511", "orig_path": "a", "path": "a", "phash": "0" * 16, "group_id": "1"},
        {"source": "iowa_rwis", "orig_path": "b", "path": "b", "phash": "0" * 16, "group_id": "IDOT-1"},
    ]
    stats, alive = thin_and_dedup(rows)
    assert stats["cross_source_duplicates_collapsed"] == 0
    assert len(alive) == 2


def test_thin_and_dedup_still_collapses_genuine_cross_source_duplicates():
    rows = [
        {"source": "roadway_flooding", "orig_path": "a", "path": "a", "phash": "0" * 16,
         "mask_path": "m", "group_id": None},
        {"source": "eu_flood_2013", "orig_path": "b", "path": "b", "phash": "0" * 16, "group_id": "g1"},
    ]
    stats, alive = thin_and_dedup(rows)
    assert stats["cross_source_duplicates_collapsed"] == 1
    assert len(alive) == 1


def test_iowa_rwis_forced_splits_is_deterministic():
    ext_rows = []
    for i in range(10):
        cam = f"IDOT-{i:03d}"
        ext_rows.append({"source": "iowa_rwis", "camera_id": cam, "label": "wet"})
        ext_rows.append({"source": "iowa_rwis", "camera_id": cam, "label": "dry"})
    first = iowa_rwis_forced_splits(ext_rows, n_test=2, n_val=2)
    second = iowa_rwis_forced_splits(ext_rows, n_test=2, n_val=2)
    assert first == second

