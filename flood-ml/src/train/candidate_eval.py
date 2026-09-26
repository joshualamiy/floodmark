# val-only candidate checks: perturbation flips + live false alarms
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from train.data import make_dataset
from train.pipeline_eval import predict_probs, status_from_probs


def _val_images_array(rows: pd.DataFrame, img_size: int, mode: str) -> np.ndarray:
    fake = rows.copy()
    fake["label_bin"] = 0.0
    fake["sample_weight"] = 1.0
    ds = make_dataset(
        fake, stage="a", variant=None, training=False, batch_size=len(rows), img_size=img_size, mode=mode,
    )
    images, _labels, _weights = next(iter(ds))
    return images.numpy()


def perturb_flip_rate(stage_a_model, stage_b_model, flooded_rows, img_size, mode, ta, tb) -> dict:
    from eval.shortcut import _perturb

    base = _val_images_array(flooded_rows, img_size, mode)
    out = {}
    for kind in ("dark", "label_box"):
        pert = np.zeros_like(base)
        for i, img in enumerate(base):
            if img_size != 224:
                small = np.asarray(
                    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR),
                    dtype=np.float32,
                )
                p = _perturb(small, kind)
                pert[i] = np.asarray(
                    Image.fromarray(np.clip(p, 0, 255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR),
                    dtype=np.float32,
                )
            else:
                pert[i] = _perturb(img, kind)
        pA = stage_a_model.predict(pert, verbose=0).reshape(-1)
        pB = stage_b_model.predict(pert, verbose=0).reshape(-1)
        status = status_from_probs(pA, pB, ta, tb)
        out[kind] = {
            "n": len(flooded_rows),
            "flipped_to_dry": int(np.sum(status == "dry")),
            "still_flooded": int(np.sum(status == "flooded")),
        }
    return out


def live_false_alarms(
    stage_a_model, stage_b_model, img_size, mode, ta, tb,
    frames_csv="data/ga511/frames.csv",
    cam_splits_json="data/processed/ga511_camera_splits.json",
    ga511_root="data/ga511",
    max_frames: int | None = None,
    seed: int = 0,
) -> dict:
    f = pd.read_csv(frames_csv)
    f = f[f["dead_reason"].isna() & f["path"].notna()].copy()
    splits = json.loads(Path(cam_splits_json).read_text())
    f["cam_split"] = f["camera_id"].astype(str).map(splits).fillna("unmapped")
    seen = f[f["cam_split"].isin(["train", "val"])].reset_index(drop=True)
    if max_frames and len(seen) > max_frames:
        seen = seen.sample(max_frames, random_state=seed).reset_index(drop=True)

    rows = pd.DataFrame({"path": [f"{ga511_root}/{p}" for p in seen["path"]]})
    pA = predict_probs(stage_a_model, rows, img_size, mode=mode)
    pB = predict_probs(stage_b_model, rows, img_size, mode=mode)
    status = status_from_probs(pA, pB, ta, tb)
    n = len(rows)
    return {
        "n": n,
        "n_train_cams": int((seen["cam_split"] == "train").sum()),
        "n_val_cams": int((seen["cam_split"] == "val").sum()),
        "flooded": int(np.sum(status == "flooded")),
        "wet": int(np.sum(status == "wet")),
        "flooded_rate": float(np.mean(status == "flooded")) if n else float("nan"),
        "wet_rate": float(np.mean(status == "wet")) if n else float("nan"),
    }


if __name__ == "__main__":
    import argparse

    from train.data import load_manifest, select_all_rows
    from train.pipeline_eval import load_checkpoint, run_pipeline_selection

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-a-run", required=True)
    ap.add_argument("--stage-b-run", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--mode", default="crop")
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--live-max-frames", type=int, default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    sel = run_pipeline_selection(
        args.stage_a_run, {args.label: args.stage_b_run}, img_size=args.img_size, mode=args.mode,
    )
    report = sel["variants"][args.label]["report"]
    ta, tb = report["tA"], report["tB"]

    df = load_manifest()
    val_rows = select_all_rows(df, split="val")
    flooded_rows = val_rows[val_rows["label"] == "flooded"].reset_index(drop=True)

    stage_a_model = load_checkpoint(args.stage_a_run)
    stage_b_model = load_checkpoint(args.stage_b_run)

    perturb = perturb_flip_rate(stage_a_model, stage_b_model, flooded_rows, args.img_size, args.mode, ta, tb)
    live = live_false_alarms(
        stage_a_model, stage_b_model, args.img_size, args.mode, ta, tb, max_frames=args.live_max_frames,
    )

    out = {
        "label": args.label, "mode": args.mode, "img_size": args.img_size,
        "stage_a_run": args.stage_a_run, "stage_b_run": args.stage_b_run,
        "pipeline_report": report, "perturb": perturb, "live_false_alarms": live,
    }
    print(json.dumps(out, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))

