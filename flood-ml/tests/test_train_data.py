import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from train.data import (
    MODE_CROP,
    MODE_LETTERBOX,
    MODE_SQUASH,
    STAGE_A,
    STAGE_B,
    VARIANT_MIXED,
    VARIANT_SPEC,
    compute_class_weights,
    compute_sample_weights,
    make_dataset,
    select_all_rows,
    select_stage_rows,
)


def _manifest_df():
    rows = []
    for i in range(20):
        rows.append({"path": f"p{i}.jpg", "label": "dry", "split": "train"})
    for i in range(3):
        rows.append({"path": f"w{i}.jpg", "label": "wet", "split": "train"})
    for i in range(10):
        rows.append({"path": f"f{i}.jpg", "label": "flooded", "split": "train"})
    for i in range(5):
        rows.append({"path": f"pv{i}.jpg", "label": "dry", "split": "val"})
    for i in range(2):
        rows.append({"path": f"wv{i}.jpg", "label": "wet", "split": "val"})
    for i in range(4):
        rows.append({"path": f"fv{i}.jpg", "label": "flooded", "split": "val"})
    return pd.DataFrame(rows)


def _manifest_df_with_not_flooded():
    df = _manifest_df()
    extra = pd.DataFrame([
        {"path": "nf0.jpg", "label": "not_flooded", "split": "train"},
        {"path": "nf1.jpg", "label": "not_flooded", "split": "train"},
        {"path": "nfv0.jpg", "label": "not_flooded", "split": "val"},
    ])
    return pd.concat([df, extra], ignore_index=True)


def test_stage_a_excludes_not_flooded_rows():
    df = _manifest_df_with_not_flooded()
    rows = select_stage_rows(df, STAGE_A, split="train")
    assert "not_flooded" not in set(rows["label"])
    assert len(rows) == 33


def test_stage_b_mixed_treats_not_flooded_as_negative():
    df = _manifest_df_with_not_flooded()
    rows = select_stage_rows(df, STAGE_B, VARIANT_MIXED, split="train")
    assert "not_flooded" in set(rows["label"])
    nf_mask = rows["label"] == "not_flooded"
    assert set(rows.loc[nf_mask, "label_bin"]) == {0.0}


def test_stage_b_spec_excludes_not_flooded_rows():
    df = _manifest_df_with_not_flooded()
    rows = select_stage_rows(df, STAGE_B, VARIANT_SPEC, split="train")
    assert "not_flooded" not in set(rows["label"])


def test_not_flooded_never_gets_wet_upweight():
    df = _manifest_df_with_not_flooded()
    rows = select_stage_rows(df, STAGE_B, VARIANT_MIXED, split="train")
    weights = compute_sample_weights(rows, STAGE_B, VARIANT_MIXED, wet_upweight=5.0)
    wet_w = weights[(rows["label"] == "wet").to_numpy()]
    nf_w = weights[(rows["label"] == "not_flooded").to_numpy()]
    assert not np.any(np.isin(nf_w, wet_w))


def test_select_all_rows_keeps_every_label():
    df = _manifest_df_with_not_flooded()
    rows = select_all_rows(df, split="train")
    assert "not_flooded" in set(rows["label"])
    assert len(rows) == len(df[df["split"] == "train"])


def test_stage_a_uses_all_rows_and_binarizes_dry_vs_rest():
    df = _manifest_df()
    rows = select_stage_rows(df, STAGE_A, split="train")
    assert len(rows) == 33
    assert set(rows.loc[rows["label"] == "dry", "label_bin"]) == {0.0}
    assert set(rows.loc[rows["label"] != "dry", "label_bin"]) == {1.0}


def test_stage_b_spec_keeps_only_wet_and_flooded():
    df = _manifest_df()
    rows = select_stage_rows(df, STAGE_B, VARIANT_SPEC, split="train")
    assert set(rows["label"]) == {"wet", "flooded"}
    assert len(rows) == 13
    assert set(rows.loc[rows["label"] == "wet", "label_bin"]) == {0.0}
    assert set(rows.loc[rows["label"] == "flooded", "label_bin"]) == {1.0}


def test_stage_b_mixed_keeps_all_rows_flooded_vs_rest():
    df = _manifest_df()
    rows = select_stage_rows(df, STAGE_B, VARIANT_MIXED, split="train")
    assert len(rows) == 33
    assert set(rows.loc[rows["label"] != "flooded", "label_bin"]) == {0.0}
    assert set(rows.loc[rows["label"] == "flooded", "label_bin"]) == {1.0}


def test_split_filter_never_mixes_splits():
    df = _manifest_df()
    rows = select_stage_rows(df, STAGE_A, split="val")
    assert len(rows) == 11
    assert set(rows["path"]).isdisjoint(set(df.loc[df["split"] == "train", "path"]))


def test_compute_class_weights_balances_by_inverse_frequency():
    label_bin = np.array([0] * 8 + [1] * 2)
    weights = compute_class_weights(label_bin)
    assert weights[1] > weights[0]
    assert weights[0] == pytest.approx(10 / (2 * 8))
    assert weights[1] == pytest.approx(10 / (2 * 2))


def test_compute_class_weights_degenerate_single_class():
    weights = compute_class_weights(np.array([1, 1, 1]))
    assert weights == {0: 1.0, 1: 1.0}


def test_sample_weights_upweight_wet_only_for_mixed_variant():
    df = _manifest_df()
    rows_mixed = select_stage_rows(df, STAGE_B, VARIANT_MIXED, split="train")
    w_mixed = compute_sample_weights(rows_mixed, STAGE_B, VARIANT_MIXED, wet_upweight=5.0)
    w_spec = compute_sample_weights(rows_mixed, STAGE_B, VARIANT_SPEC, wet_upweight=5.0)
    wet_mask = (rows_mixed["label"] == "wet").to_numpy()
    assert np.all(w_mixed[wet_mask] > w_spec[wet_mask])
    assert np.allclose(w_mixed[wet_mask] / w_spec[wet_mask], 5.0)


def _write_fixture_images(tmp_path, rows):
    from PIL import Image

    rng = np.random.default_rng(0)
    for _, row in rows.iterrows():
        full = tmp_path / row["path"]
        full.parent.mkdir(parents=True, exist_ok=True)
        arr = rng.integers(0, 256, size=(256, 300, 3), dtype=np.uint8)
        Image.fromarray(arr).save(full, quality=90)


def test_make_dataset_yields_expected_shapes_and_ranges(tmp_path):
    df = _manifest_df()
    rows = select_stage_rows(df, STAGE_A, split="train").iloc[:6].reset_index(drop=True)
    _write_fixture_images(tmp_path, rows)

    ds = make_dataset(
        rows, stage=STAGE_A, training=True, batch_size=4, img_size=32,
        repo_root=str(tmp_path),
    )
    images, labels, weights = next(iter(ds))
    assert images.shape[1:] == (32, 32, 3)
    assert images.dtype.name == "float32"
    assert float(images.numpy().min()) >= 0.0
    assert float(images.numpy().max()) <= 255.0
    assert labels.shape[0] == images.shape[0]
    assert weights.shape[0] == images.shape[0]


def test_make_dataset_val_is_deterministic_center_crop(tmp_path):
    df = _manifest_df()
    rows = select_stage_rows(df, STAGE_A, split="val").iloc[:3].reset_index(drop=True)
    _write_fixture_images(tmp_path, rows)

    ds1 = make_dataset(rows, stage=STAGE_A, training=False, batch_size=8, img_size=32, repo_root=str(tmp_path))
    ds2 = make_dataset(rows, stage=STAGE_A, training=False, batch_size=8, img_size=32, repo_root=str(tmp_path))
    images1, _, _ = next(iter(ds1))
    images2, _, _ = next(iter(ds2))
    assert np.array_equal(images1.numpy(), images2.numpy())


@pytest.mark.parametrize("mode", [MODE_CROP, MODE_SQUASH, MODE_LETTERBOX])
def test_make_dataset_every_mode_yields_expected_shape(tmp_path, mode):
    df = _manifest_df()
    rows = select_stage_rows(df, STAGE_A, split="train").iloc[:4].reset_index(drop=True)
    _write_fixture_images(tmp_path, rows)

    for training in (True, False):
        ds = make_dataset(
            rows, stage=STAGE_A, training=training, batch_size=4, img_size=48, mode=mode,
            repo_root=str(tmp_path),
        )
        images, labels, weights = next(iter(ds))
        assert images.shape[1:] == (48, 48, 3)
        assert float(images.numpy().min()) >= 0.0
        assert float(images.numpy().max()) <= 255.0
        assert labels.shape[0] == images.shape[0] == weights.shape[0]


def test_make_dataset_rejects_unknown_mode(tmp_path):
    df = _manifest_df()
    rows = select_stage_rows(df, STAGE_A, split="train").iloc[:2].reset_index(drop=True)
    _write_fixture_images(tmp_path, rows)
    with pytest.raises(ValueError):
        make_dataset(rows, stage=STAGE_A, training=False, img_size=32, mode="fisheye", repo_root=str(tmp_path))


def test_squash_and_letterbox_val_are_deterministic(tmp_path):
    df = _manifest_df()
    rows = select_stage_rows(df, STAGE_A, split="val").iloc[:3].reset_index(drop=True)
    _write_fixture_images(tmp_path, rows)
    for mode in (MODE_SQUASH, MODE_LETTERBOX):
        ds1 = make_dataset(rows, stage=STAGE_A, training=False, img_size=32, mode=mode, repo_root=str(tmp_path))
        ds2 = make_dataset(rows, stage=STAGE_A, training=False, img_size=32, mode=mode, repo_root=str(tmp_path))
        images1, _, _ = next(iter(ds1))
        images2, _, _ = next(iter(ds2))
        assert np.array_equal(images1.numpy(), images2.numpy())


def test_letterbox_pads_a_wide_image_with_mid_gray(tmp_path):
    df = _manifest_df()
    rows = select_stage_rows(df, STAGE_A, split="val").iloc[:1].reset_index(drop=True)
    _write_fixture_images(tmp_path, rows)
    ds = make_dataset(rows, stage=STAGE_A, training=False, img_size=64, mode=MODE_LETTERBOX, repo_root=str(tmp_path))
    images, _, _ = next(iter(ds))
    img = images.numpy()[0]
    assert img.shape == (64, 64, 3)
    assert img[0, 0, 0] == pytest.approx(128.0, abs=1.0)

