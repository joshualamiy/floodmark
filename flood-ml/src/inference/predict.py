# predict() / predict_batch(): the backend contract, see docs/INFERENCE_API.md
from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np

from .heatmap import heatmap_display, make_heatmap_png
from .preprocess import preprocess, to_pil
from .session import load_models

WET_NOTE = "water detected below flood alert threshold"


@dataclass
class Prediction:
    status: str
    confidence: float
    stage_a_probs: dict
    stage_b_probs: dict
    stage_probabilities: dict
    heatmap_png: bytes | None
    thresholds: dict
    model_version: dict
    note: str | None = None
    heatmap_status: str = "disabled"
    heatmap_note: str | None = None
    heatmap_score: float | None = None
    raw_heatmap_png: bytes | None = None

    def to_dict(self, include_heatmap: bool = False) -> dict:
        d = {
            "status": self.status,
            "confidence": self.confidence,
            "stage_a_probs": self.stage_a_probs,
            "stage_b_probs": self.stage_b_probs,
            "stage_probabilities": self.stage_probabilities,
            "thresholds": self.thresholds,
            "model_version": self.model_version,
            "note": self.note,
            "heatmap_png": None,
            "heatmap_status": self.heatmap_status,
            "heatmap_note": self.heatmap_note,
            "heatmap_score": self.heatmap_score,
            "raw_heatmap_png": None,
        }
        if include_heatmap and self.heatmap_png is not None:
            d["heatmap_png"] = base64.b64encode(self.heatmap_png).decode("ascii")
        if include_heatmap and self.raw_heatmap_png is not None:
            d["raw_heatmap_png"] = base64.b64encode(self.raw_heatmap_png).decode("ascii")
        return d


# dry if pA < tA, flooded if pB >= tB, else wet (= possible flooding)
def _status_and_probs(pa: float, pb: float, ta: float, tb: float):
    if pa < ta:
        status = "dry"
    elif pb >= tb:
        status = "flooded"
    else:
        status = "wet"
    sp = {"dry": 1.0 - pa, "wet": pa * (1.0 - pb), "flooded": pa * pb}
    return status, sp, sp[status]


def _run_stage(sess, x: np.ndarray):
    cam, prob = sess.run(["cam", "prob"], {"image": x})
    return prob[:, 0], cam


def predict(image, models=None, heatmap: bool = True, camera_id: str | None = None,
            *, raw_heatmap: bool = False) -> Prediction:
    return predict_batch([image], models=models, heatmap=heatmap,
                         camera_id=camera_id, raw_heatmap=raw_heatmap)[0]


def predict_batch(
    images, models=None, heatmap: bool = True, camera_id: str | None = None,
    *, raw_heatmap: bool = False,
) -> list[Prediction]:
    del camera_id
    if len(images) == 0:
        return []
    m = models or load_models()
    cfg = m.config
    ta = float(cfg["stage_a"]["threshold_tA"])
    tb = float(cfg["stage_b"]["threshold_tB"])
    preproc_cfg = cfg.get("preprocess", {})
    do_jpeg = preproc_cfg.get("jpeg_roundtrip", True)
    mode = preproc_cfg.get("mode", "crop")
    size = int(preproc_cfg.get("size", cfg.get("input", {}).get("size", 224)))

    frames = [to_pil(img) for img in images]
    arrs, geoms = [], []
    for f in frames:
        arr, geom = preprocess(f, mode=mode, size=size, do_jpeg_roundtrip=do_jpeg)
        arrs.append(arr)
        geoms.append(geom)
    x = np.stack(arrs).astype(np.float32)

    pa, _cam_a = _run_stage(m.sess_a, x)
    pb, cam_b = _run_stage(m.sess_b, x)

    model_version = {
        "data_version": cfg.get("data_version"),
        "stage_a_run_id": cfg["stage_a"]["run_id"],
        "stage_b_run_id": cfg["stage_b"]["run_id"],
    }
    thresholds = {"tA": ta, "tB": tb}

    out = []
    for i, frame in enumerate(frames):
        status, sp, conf = _status_and_probs(float(pa[i]), float(pb[i]), ta, tb)
        note = WET_NOTE if status == "wet" else None
        display = heatmap_display(float(pa[i]), float(pb[i]), ta, cam_b[i])
        hm = make_heatmap_png(frame, cam_b[i], geoms[i],
                              strength=display["strength"]) if heatmap else None
        raw = make_heatmap_png(frame, cam_b[i], geoms[i]) if raw_heatmap else None
        out.append(Prediction(
            status=status,
            confidence=float(conf),
            stage_a_probs={"dry": float(1.0 - pa[i]), "wet": float(pa[i])},
            stage_b_probs={"not_flooded": float(1.0 - pb[i]), "flooded": float(pb[i])},
            stage_probabilities={k: float(v) for k, v in sp.items()},
            heatmap_png=hm,
            thresholds=thresholds,
            model_version=model_version,
            note=note,
            heatmap_status=display["state"] if heatmap else "disabled",
            heatmap_note=display["note"] if heatmap else None,
            heatmap_score=display["score"] if heatmap else None,
            raw_heatmap_png=raw,
        ))
    return out

