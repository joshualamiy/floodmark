"""Multi-crop test-time augmentation on the CURRENT (crop-224) shipped ONNX
models -- no retraining. Takes left/center/right 224 crops of the 256-short-
side frame and reports max pA, max pB across the three, so an off-center
flood (improve_v2 problem 1) has a better chance of landing inside a crop.
Cheap fallback: same models, ~3x inference cost, no new training run.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SHORT_SIDE = 256
CROP_SIZE = 224


def three_crop_boxes(w: int, h: int, size: int = CROP_SIZE) -> list[tuple[int, int, int, int]]:
    """Left/center/right (or top/center/bottom) boxes along the longer axis of
    a size-or-bigger frame. Collapses to one box if the frame is <= size on
    that axis (dedup keeps them in left-to-right / top-to-bottom order).
    """
    boxes = []
    if w >= h:
        offsets = sorted({0, max(0, (w - size) // 2), max(0, w - size)})
        for x in offsets:
            boxes.append((x, max(0, (h - size) // 2), x + size, max(0, (h - size) // 2) + size))
    else:
        offsets = sorted({0, max(0, (h - size) // 2), max(0, h - size)})
        for y in offsets:
            boxes.append((max(0, (w - size) // 2), y, max(0, (w - size) // 2) + size, y + size))
    return boxes


def three_crops(img: Image.Image, short_side: int = SHORT_SIDE, size: int = CROP_SIZE) -> list[np.ndarray]:
    w, h = img.size
    if min(w, h) > short_side:
        scale = short_side / min(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    w2, h2 = img.size
    crops = []
    for box in three_crop_boxes(w2, h2, size):
        canvas = Image.new("RGB", (size, size))
        canvas.paste(img.crop(box), (0, 0))
        crops.append(np.asarray(canvas, dtype=np.float32))
    return crops


def perturb_tta_flip_rate(sess_a, sess_b, paths, ta: float, tb: float, size: int = CROP_SIZE) -> dict:
    """Perturbation robustness for the multi-crop TTA candidate: applies
    eval.shortcut._perturb (imported, not edited) to EACH of the 3 crops,
    then takes the max pA/pB across crops, same as the unperturbed case.
    """
    from eval.shortcut import _perturb
    from train.pipeline_eval import status_from_probs

    out = {}
    for kind in ("dark", "label_box"):
        pa_out, pb_out = [], []
        for p in paths:
            img = Image.open(p).convert("RGB")
            crops = three_crops(img, size=size)
            x = np.stack([_perturb(c, kind) for c in crops]).astype(np.float32)
            _cam_a, prob_a = sess_a.run(["cam", "prob"], {"image": x})
            _cam_b, prob_b = sess_b.run(["cam", "prob"], {"image": x})
            pa_out.append(float(prob_a[:, 0].max()))
            pb_out.append(float(prob_b[:, 0].max()))
        status = status_from_probs(np.array(pa_out), np.array(pb_out), ta, tb)
        out[kind] = {"n": len(paths), "flipped_to_dry": int(np.sum(status == "dry")),
                     "still_flooded": int(np.sum(status == "flooded"))}
    return out


def predict_tta_max(sess_a, sess_b, paths, size: int = CROP_SIZE, batch: int = 32) -> dict[str, np.ndarray]:
    """Runs both stages over every crop of every path and returns per-image
    max pA / max pB (the max, not the mean, since we want ANY crop that sees
    the flood to be able to trigger it).
    """
    pa_out, pb_out = [], []
    for p in paths:
        img = Image.open(p).convert("RGB")
        crops = three_crops(img, size=size)
        x = np.stack(crops).astype(np.float32)
        _cam_a, prob_a = sess_a.run(["cam", "prob"], {"image": x})
        _cam_b, prob_b = sess_b.run(["cam", "prob"], {"image": x})
        pa_out.append(float(prob_a[:, 0].max()))
        pb_out.append(float(prob_b[:, 0].max()))
    return {"pA": np.array(pa_out), "pB": np.array(pb_out)}


if __name__ == "__main__":
    import argparse
    import json

    import onnxruntime as ort
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--split", default="val")
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--out-csv", default="reports/eval/tta_val_preds.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    rows = df[df["split"] == args.split].reset_index(drop=True)
    prov = ["CPUExecutionProvider"]
    sess_a = ort.InferenceSession(f"{args.models_root}/stage_a.onnx", providers=prov)
    sess_b = ort.InferenceSession(f"{args.models_root}/stage_b.onnx", providers=prov)
    r = predict_tta_max(sess_a, sess_b, rows["path"])
    rows["pA_tta"], rows["pB_tta"] = r["pA"], r["pB"]
    from pathlib import Path

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_csv, index=False)
    print(json.dumps({"n": len(rows), "out_csv": args.out_csv}, indent=2))
