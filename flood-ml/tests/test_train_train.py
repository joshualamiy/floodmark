from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from train.train import (
    RUNS_CSV_COLUMNS,
    ValMetrics,
    _precision_target_threshold,
    _youden_threshold,
    append_run_row,
    evaluate_val,
    make_run_id,
    parse_args,
    read_data_version,
    run_training,
)


def test_make_run_id_format():
    now = datetime(2026, 9, 25, 12, 30, 5, tzinfo=timezone.utc)
    run_id = make_run_id("a", None, "mobilenetv3small", now=now)
    assert run_id == "20260925-123005_a_mobilenetv3small"
    run_id_b = make_run_id("b", "mixed", "efficientnetb0", now=now)
    assert run_id_b == "20260925-123005_bmixed_efficientnetb0"


def test_read_data_version(tmp_path):
    path = tmp_path / "VERSION"
    path.write_text("data_version = v1-c7dea35e\n")
    assert read_data_version(path) == "v1-c7dea35e"


def test_youden_threshold_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = _youden_threshold(y, scores)
    assert 0.3 < thr <= 0.7


def test_precision_target_threshold_reaches_target_when_possible():
    y = np.array([0] * 8 + [1] * 8)
    scores = np.concatenate([np.linspace(0, 0.4, 8), np.linspace(0.6, 1.0, 8)])
    thr, note = _precision_target_threshold(y, scores, target_precision=0.90)
    pred = scores >= thr
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    precision = tp / (tp + fp)
    assert precision >= 0.90
    assert "reaching precision" in note


def test_precision_target_threshold_falls_back_to_f05_when_unreachable():
    y = np.array([0] * 5 + [1] * 5 + [0] * 90)
    scores = np.concatenate([np.full(10, 0.99), np.full(90, 0.1)])
    _thr, note = _precision_target_threshold(y, scores, target_precision=0.90)
    assert "unreachable" in note


def test_evaluate_val_reports_false_alarm_rate_and_flags_small_n():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.1, 0.9, 0.2, 0.8, 0.9, 0.7, 0.6, 0.95])
    false_alarm_label = np.array([True, True, False, True, False, False, False, False, False, False])
    metrics = evaluate_val(y_true, probs, stage="a", false_alarm_label=false_alarm_label)
    assert isinstance(metrics, ValMetrics)
    assert metrics.false_alarm_den == 3
    assert 0.0 <= metrics.false_alarm_rate <= 1.0


def test_append_run_row_creates_header_once(tmp_path):
    runs_csv = tmp_path / "runs.csv"
    row = {col: f"v_{col}" for col in RUNS_CSV_COLUMNS}
    append_run_row(row, runs_csv)
    append_run_row(row, runs_csv)
    df = pd.read_csv(runs_csv)
    assert list(df.columns) == RUNS_CSV_COLUMNS
    assert len(df) == 2


def test_parse_args_defaults_variant_for_stage_b():
    args = parse_args(["--stage", "b", "--epochs-head", "1"])
    assert args.variant == "spec"


def test_parse_args_weights_none_string_becomes_none():
    args = parse_args(["--stage", "a", "--weights", "none"])
    assert args.weights is None


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
            rows.append({
                "path": rel, "label": label, "source": "synth", "group_id": f"g{split}{i}",
                "camera_id": f"c{i}", "view_type": "ground", "license": "test", "restricted": False,
                "split": split, "label_source": "manual", "orig_path": "", "mask_path": "",
                "water_frac_road": "", "phash": "", "width": 110, "height": 96, "dup_cluster": "",
                "weak_label": "", "notes": "",
            })
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    version_path = tmp_path / "data" / "processed" / "VERSION"
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text("data_version = vtest-00000000\n")
    return manifest_path, version_path


def test_run_training_end_to_end_tiny(tmp_path, monkeypatch):
    manifest_path, _version_path = _write_tiny_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)

    args = parse_args([
        "--stage", "a", "--backbone", "mobilenetv3small", "--weights", "none",
        "--img-size", "64", "--batch-size", "4", "--epochs-head", "1", "--epochs-ft", "0",
        "--manifest", str(manifest_path), "--runs-csv", "reports/runs.csv",
        "--logdir-root", "logs", "--models-root", "models", "--run-id", "unittest_a",
        "--patience", "1",
    ])
    result = run_training(args)
    assert result["run_id"] == "unittest_a"
    assert (tmp_path / "models" / "unittest_a" / "model.keras").exists()

    runs_csv = pd.read_csv(tmp_path / "reports" / "runs.csv")
    assert len(runs_csv) == 1
    assert runs_csv.iloc[0]["stage"] == "a"

