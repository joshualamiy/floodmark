from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from train.candidate_eval import ga511_daytime_false_alarms, live_false_alarms, perturb_flip_rate
from train.train import parse_args, run_training


def _write_tiny_manifest(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    rows = []
    labels_cycle = ["dry"] * 6 + ["wet"] * 2 + ["flooded"] * 6
    for split in ("train", "val"):
        for i, label in enumerate(labels_cycle):
            rel = f"images/{split}_{i}.jpg"
            full = tmp_path / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            arr = rng.integers(0, 256, size=(96, 110, 3), dtype=np.uint8)
            Image.fromarray(arr).save(full, quality=90)
            rows.append({"path": rel, "label": label, "split": split})
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    version_path = tmp_path / "data" / "processed" / "VERSION"
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text("data_version = vtest-00000000\n")
    return manifest_path


def _train_tiny(tmp_path, manifest_path, stage, run_id, variant=None):
    argv = [
        "--stage", stage, "--backbone", "mobilenetv3small", "--weights", "none",
        "--img-size", "64", "--batch-size", "4", "--epochs-head", "1", "--epochs-ft", "0",
        "--manifest", str(manifest_path), "--runs-csv", "reports/runs.csv",
        "--logdir-root", "logs", "--models-root", "models", "--run-id", run_id,
        "--patience", "1",
    ]
    if variant:
        argv += ["--variant", variant]
    result = run_training(parse_args(argv))
    return result["run_id"]


@pytest.fixture
def tiny_models(tmp_path, monkeypatch):
    manifest_path = _write_tiny_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)
    a_run = _train_tiny(tmp_path, manifest_path, "a", "unittest_cand_a")
    b_run = _train_tiny(tmp_path, manifest_path, "b", "unittest_cand_b", variant="mixed")

    from train.pipeline_eval import load_checkpoint

    return load_checkpoint(a_run), load_checkpoint(b_run), manifest_path


def test_perturb_flip_rate_returns_dark_and_label_box_counts(tiny_models):
    stage_a_model, stage_b_model, manifest_path = tiny_models
    df = pd.read_csv(manifest_path)
    flooded_rows = df[(df["split"] == "val") & (df["label"] == "flooded")].reset_index(drop=True)

    out = perturb_flip_rate(stage_a_model, stage_b_model, flooded_rows, img_size=64, mode="crop", ta=0.5, tb=0.5)
    assert set(out.keys()) == {"dark", "label_box"}
    for kind in ("dark", "label_box"):
        assert out[kind]["n"] == len(flooded_rows)
        assert 0 <= out[kind]["flipped_to_dry"] <= len(flooded_rows)


def test_live_false_alarms_counts_status_on_synthetic_frames(tmp_path, tiny_models):
    stage_a_model, stage_b_model, _manifest_path = tiny_models
    from PIL import Image

    rng = np.random.default_rng(1)
    ga511_root = tmp_path / "data" / "ga511"
    frames_dir = ga511_root / "frames" / "v1"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(4):
        rel = f"frames/v1/{i}.jpg"
        arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(ga511_root / rel, quality=90)
        rows.append({"frame_id": f"f{i}", "path": rel, "camera_id": str(100 + i % 2), "dead_reason": ""})
    pd.DataFrame(rows).to_csv(ga511_root / "frames.csv", index=False)

    splits_path = tmp_path / "data" / "processed" / "ga511_camera_splits.json"
    splits_path.write_text(json.dumps({"100": "train", "101": "val"}))

    out = live_false_alarms(
        stage_a_model, stage_b_model, img_size=64, mode="crop", ta=0.5, tb=0.5,
        frames_csv="data/ga511/frames.csv", cam_splits_json="data/processed/ga511_camera_splits.json",
        ga511_root="data/ga511",
    )
    assert out["n"] == 4
    assert out["n_train_cams"] + out["n_val_cams"] == 4
    assert 0 <= out["flooded"] <= 4


def test_live_false_alarms_splits_day_and_night(tmp_path, tiny_models):
    stage_a_model, stage_b_model, _manifest_path = tiny_models
    from PIL import Image

    rng = np.random.default_rng(2)
    ga511_root = tmp_path / "data" / "ga511"
    frames_dir = ga511_root / "frames" / "v1"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # 1790438400 = 2026-09-26 12:00 America/New_York (day); 1790402400 = 02:00 same date (night)
    timestamps = [1790438400, 1790438400, 1790402400, 1790402400]
    rows = []
    for i, ts in enumerate(timestamps):
        rel = f"frames/v1/{i}.jpg"
        arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(ga511_root / rel, quality=90)
        rows.append({
            "frame_id": f"f{i}", "path": rel, "camera_id": str(100 + i % 2),
            "dead_reason": "", "timestamp_utc": ts,
        })
    pd.DataFrame(rows).to_csv(ga511_root / "frames.csv", index=False)

    splits_path = tmp_path / "data" / "processed" / "ga511_camera_splits.json"
    splits_path.write_text(json.dumps({"100": "train", "101": "val"}))

    out = live_false_alarms(
        stage_a_model, stage_b_model, img_size=64, mode="crop", ta=0.5, tb=0.5,
        frames_csv="data/ga511/frames.csv", cam_splits_json="data/processed/ga511_camera_splits.json",
        ga511_root="data/ga511",
    )
    assert out["day"]["n"] == 2
    assert out["night"]["n"] == 2
    assert out["n_unknown_time"] == 0


def test_ga511_daytime_false_alarms_buckets_manual_and_weak():
    day_path = "data/ga511/frames/1/1790438400.jpg"
    night_path = "data/ga511/frames/1/1790402400.jpg"
    val_rows = pd.DataFrame([
        {"source": "ga511", "label": "dry", "label_source": "manual", "orig_path": day_path},
        {"source": "ga511", "label": "dry", "label_source": "weak_precip", "orig_path": day_path},
        {"source": "ga511", "label": "dry", "label_source": "manual", "orig_path": night_path},
        {"source": "ga511", "label": "flooded", "label_source": "manual", "orig_path": day_path},
        {"source": "fred", "label": "dry", "label_source": "sequence_condition", "orig_path": "data/raw/fred/x.png"},
    ])
    pA = np.array([0.95, 0.95, 0.95, 0.95, 0.95])
    pB = np.array([0.95, 0.1, 0.95, 0.95, 0.95])
    out = ga511_daytime_false_alarms(val_rows, pA, pB, ta=0.5, tb=0.5)
    assert out["manual"] == {"n": 1, "false_alarms": 1, "rate": 1.0}
    assert out["weak_precip"] == {"n": 1, "false_alarms": 0, "rate": 0.0}
    assert out["combined"]["n"] == 2
    assert out["combined"]["false_alarms"] == 1

