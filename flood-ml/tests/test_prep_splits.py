from collections import Counter

from prep.splits import (
    assert_disjoint,
    ga511_camera_splits,
    greedy_group_stratified_split,
)


def test_ga511_camera_splits_covers_every_camera():
    cams = [f"cam{i}" for i in range(200)]
    splits = ga511_camera_splits(cams)
    assert set(splits.keys()) == set(cams)
    assert set(splits.values()) <= {"train", "val", "test"}


def test_ga511_camera_splits_roughly_20_percent_test():
    cams = [f"cam{i}" for i in range(2000)]
    splits = ga511_camera_splits(cams)
    counts = Counter(splits.values())
    frac_test = counts["test"] / len(cams)
    assert 0.10 < frac_test < 0.30


def test_ga511_camera_splits_is_stable_across_reruns():
    cams = [f"cam{i}" for i in range(150)]
    first = ga511_camera_splits(cams)
    second = ga511_camera_splits(cams)
    assert first == second


def test_ga511_camera_splits_never_moves_an_existing_camera():
    cams = [f"cam{i}" for i in range(100)]
    first = ga511_camera_splits(cams)
    more_cams = cams + [f"cam{i}" for i in range(100, 130)]
    second = ga511_camera_splits(more_cams, existing=first)
    for cam, split in first.items():
        assert second[cam] == split


def test_ga511_camera_splits_pulls_reviewed_positive_new_cameras_toward_test():
    existing = {f"old{i}": "train" for i in range(50)}
    new_cams = [f"newcam{i}" for i in range(20)]
    splits = ga511_camera_splits(
        new_cams, existing=existing, reviewed_positive_cameras=set(new_cams)
    )
    n_test = sum(1 for c in new_cams if splits[c] == "test")
    assert n_test > 0


def test_greedy_group_stratified_split_respects_forced_assignments():
    group_labels = {
        "g1": Counter({"flooded": 10}),
        "g2": Counter({"dry": 10}),
        "g3": Counter({"wet": 5}),
    }
    forced = {"g1": "test", "g2": "test"}
    result = greedy_group_stratified_split(group_labels, forced=forced)
    assert result["g1"] == "test"
    assert result["g2"] == "test"
    assert result["g3"] in {"train", "val", "test"}


def test_greedy_group_stratified_split_covers_all_groups():
    group_labels = {f"g{i}": Counter({"dry": 3, "flooded": 1}) for i in range(30)}
    result = greedy_group_stratified_split(group_labels)
    assert set(result.keys()) == set(group_labels.keys())
    assert set(result.values()) <= {"train", "val", "test"}


def test_greedy_group_stratified_split_is_deterministic():
    group_labels = {f"g{i}": Counter({"wet": (i % 3) + 1}) for i in range(20)}
    first = greedy_group_stratified_split(group_labels)
    second = greedy_group_stratified_split(group_labels)
    assert first == second


def test_assert_disjoint_passes_when_groups_stay_in_one_split():
    rows = [
        {"group_id": "g1", "camera_id": "c1", "dup_cluster": "d1", "split": "train"},
        {"group_id": "g1", "camera_id": "c1", "dup_cluster": "d1", "split": "train"},
        {"group_id": "g2", "camera_id": "c2", "dup_cluster": "d2", "split": "test"},
    ]
    assert assert_disjoint(rows) == []


def test_assert_disjoint_flags_a_group_split_across_splits():
    rows = [
        {"group_id": "g1", "camera_id": "c1", "dup_cluster": "d1", "split": "train"},
        {"group_id": "g1", "camera_id": "c1", "dup_cluster": "d1", "split": "test"},
    ]
    violations = assert_disjoint(rows)
    assert violations
    assert any("g1" in v for v in violations)


def test_assert_disjoint_flags_a_dup_cluster_split_across_splits():
    rows = [
        {"group_id": "g1", "camera_id": "c1", "dup_cluster": "shared", "split": "train"},
        {"group_id": "g2", "camera_id": "c2", "dup_cluster": "shared", "split": "val"},
    ]
    violations = assert_disjoint(rows)
    assert any("shared" in v for v in violations)

