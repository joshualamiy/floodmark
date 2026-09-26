import numpy as np
import pytest
from PIL import Image

from train.water_data import annotated_rows, audit_splits, capped_rows, load_arrays, original_pair
from train.water_metrics import mask_counts, scores


def test_fred_water_label_two_and_nearest_geometry(tmp_path):
    Image.new("RGB", (8, 4), "white").save(tmp_path / "frame.png")
    Image.fromarray(np.array([[0, 1, 2, 2], [1, 1, 0, 2]], np.uint8)).save(tmp_path / "mask.png")
    row = {"orig_path": "frame.png", "mask_path": "mask.png", "source": "fred"}
    image, mask = original_pair(row, tmp_path)
    assert image.size == (8, 4)
    assert mask.sum() == 12
    _, target = load_arrays([row], 32, tmp_path)
    assert set(np.unique(target[..., 0])) == {0, 1}
    assert target[..., 1].sum() == 32 * 16
    assert target[..., 0][target[..., 1] == 0].sum() == 0


def test_roadway_one_and_unknown_values_rejected(tmp_path):
    Image.new("RGB", (2, 2)).save(tmp_path / "frame.png")
    Image.fromarray(np.array([[0, 1], [1, 0]], np.uint8)).save(tmp_path / "mask.png")
    row = {"orig_path": "frame.png", "mask_path": "mask.png", "source": "roadway_flooding"}
    assert original_pair(row, tmp_path)[1].sum() == 2
    Image.fromarray(np.full((2, 2), 255, np.uint8)).save(tmp_path / "mask.png")
    with pytest.raises(ValueError, match="semantics"):
        original_pair(row, tmp_path)


def test_unannotated_frames_are_not_invented_negatives():
    rows = [{"source": "fred", "split": "train", "orig_path": "dry.png", "mask_path": ""}]
    assert annotated_rows(rows, "train") == []
    with pytest.raises(ValueError, match="Pixel mask"):
        original_pair(rows[0])


def test_misaligned_mask_rejected(tmp_path):
    Image.new("RGB", (8, 4)).save(tmp_path / "frame.png")
    Image.new("L", (4, 4)).save(tmp_path / "mask.png")
    with pytest.raises(ValueError, match="aspect mismatch"):
        original_pair({"source": "fred", "orig_path": "frame.png", "mask_path": "mask.png"}, tmp_path)


@pytest.mark.parametrize("field", ["group_id", "dup_cluster", "orig_path", "mask_path", "phash"])
def test_cross_split_leakage_rejected(field):
    rows = [{"source": "fred", "split": split, field: "same"} for split in ("train", "val")]
    with pytest.raises(ValueError, match="leakage"):
        audit_splits(rows)


def test_deterministic_cap_respects_group_and_source():
    rows = [{"source": "fred", "group_id": group, "orig_path": f"{group}/{i}"}
            for group in ("a", "b") for i in range(20)]
    a = capped_rows(rows, per_source=5, per_group=3)
    assert a == capped_rows(list(reversed(rows)), per_source=5, per_group=3)
    assert len(a) == 5
    assert max(sum(r["group_id"] == g for r in a) for g in ("a", "b")) <= 3


def test_pixel_metrics_ignore_padding():
    counts = mask_counts([True, False, True], [True, True, False], [True, True, False])
    assert counts == {"tp": 1, "fp": 0, "fn": 1, "tn": 0}
    assert scores(counts) == {"iou": 0.5, "dice": 2 / 3}


def test_cross_source_duplicate_guard_ignores_only_unrelated_conflicts():
    unrelated = [
        {"source": "ga511", "split": split, "dup_cluster": "unrelated"}
        for split in ("train", "val")
    ]
    audit_splits(unrelated, relevant_sources=("fred", "roadway_flooding"))
    collision = unrelated + [{"source": "fred", "split": "train", "dup_cluster": "unrelated"}]
    with pytest.raises(ValueError, match="leakage"):
        audit_splits(collision, relevant_sources=("fred", "roadway_flooding"))


def test_test_evaluation_requires_reviewer_opt_in():
    from train.water_eval import evaluate
    with pytest.raises(ValueError, match="independent Phase4"):
        evaluate("absent-model", "absent-manifest", "test")
