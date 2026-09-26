from __future__ import annotations

import json

import pytest
from _fake_onnx import build_stage_onnx, logit, uniform_image
from PIL import Image

from inference import cli
from inference.session import clear_cache


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    build_stage_onnx(tmp_path / "stage_a.onnx", prob_bias=logit(0.9), cam_mode="constant")
    build_stage_onnx(tmp_path / "stage_b.onnx", prob_bias=logit(0.95), cam_mode="constant")
    cfg = {
        "data_version": "vtest",
        "stage_a": {"run_id": "a", "onnx_path": "stage_a.onnx", "threshold_tA": 0.5},
        "stage_b": {"run_id": "b", "onnx_path": "stage_b.onnx", "threshold_tB": 0.5},
        "preprocess": {"jpeg_roundtrip": False},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("FLOODML_MODEL_DIR", str(tmp_path))
    clear_cache()
    yield tmp_path
    clear_cache()


@pytest.fixture
def frames_dir(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(3):
        Image.fromarray(uniform_image(120 + i)).save(d / f"{i:03d}.jpg")
    (d / "not_an_image.txt").write_text("skip me")
    return d


def test_cli_text_output(model_dir, frames_dir, capsys):
    rc = cli.run([str(frames_dir), "--camera-id", "cam1"])
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert "raw=" in lines[0] and "smoothed=" in lines[0] and "moved=" in lines[0]


def test_cli_json_output(model_dir, frames_dir, capsys):
    rc = cli.run([str(frames_dir), "--camera-id", "cam1", "--json"])
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert "status" in row and "raw_status" in row and "moved" in row
        assert row["heatmap_png"] is None


def test_cli_save_heatmaps(model_dir, frames_dir, tmp_path):
    out_dir = tmp_path / "heatmaps"
    rc = cli.run([str(frames_dir), "--camera-id", "cam1", "--save-heatmaps", str(out_dir)])
    assert rc == 0
    saved = sorted(out_dir.glob("*.png"))
    assert len(saved) == 3


def test_cli_smoothing_reaches_flooded_after_n(model_dir, frames_dir, capsys):
    cli.run([str(frames_dir), "--camera-id", "cam1", "--smooth-n", "3", "--json"])
    rows = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [r["raw_status"] for r in rows] == ["flooded"] * 3
    assert rows[0]["smoothed_status"] == "wet"
    assert rows[1]["smoothed_status"] == "wet"
    assert rows[2]["smoothed_status"] == "flooded"


def test_cli_blocklist_suppresses_flooded(model_dir, frames_dir, capsys):
    cli.run([str(frames_dir), "--camera-id", "cam1", "--smooth-n", "1",
              "--blocklist", "cam1", "--json"])
    rows = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert all(r["smoothed_status"] != "flooded" for r in rows)

