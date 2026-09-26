"""Run the pipeline over a folder of frames.

Usage: PYTHONPATH=src ../my_env/bin/python -m inference.cli <folder> [--camera-id ID]
       [--smooth-n 3] [--json] [--save-heatmaps DIR] [--blocklist 11372,...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .camera_move import CameraMoveDetector
from .predict import predict
from .preprocess import to_pil
from .smoothing import TemporalSmoother

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def iter_frames(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--camera-id", default=None)
    ap.add_argument("--smooth-n", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--save-heatmaps", type=Path, default=None)
    ap.add_argument("--blocklist", default="")
    args = ap.parse_args(argv)

    blocklist = [b for b in args.blocklist.split(",") if b]
    smoother = TemporalSmoother(n=args.smooth_n, blocklist=blocklist)
    mover = CameraMoveDetector()
    camera_id = args.camera_id or "cli"

    if args.save_heatmaps:
        args.save_heatmaps.mkdir(parents=True, exist_ok=True)

    for path in iter_frames(args.folder):
        raw_frame = to_pil(str(path))
        moved, move_score = mover.update(camera_id, raw_frame)

        pred = predict(raw_frame, camera_id=camera_id, heatmap=bool(args.save_heatmaps))
        if moved:
            mover_skip_note = "camera moved: frame skipped, not counted toward smoothing"
            smoother.skip(camera_id)
            smoothed_status, smoothed_note = "skipped", mover_skip_note
        else:
            smoothed = smoother.update(camera_id, pred)
            smoothed_status, smoothed_note = smoothed.status, smoothed.note

        if args.save_heatmaps and pred.heatmap_png is not None:
            out = args.save_heatmaps / f"{path.stem}_heatmap.png"
            out.write_bytes(pred.heatmap_png)

        if args.json:
            row = pred.to_dict()
            row.update({
                "file": str(path),
                "raw_status": pred.status,
                "smoothed_status": smoothed_status,
                "smoothed_note": smoothed_note,
                "pA": pred.stage_a_probs["wet"],
                "pB": pred.stage_b_probs["flooded"],
                "moved": moved,
                "move_score": move_score,
            })
            print(json.dumps(row))
        else:
            print(
                f"{path.name}  raw={pred.status:8s} smoothed={smoothed_status:8s} "
                f"conf={pred.confidence:.3f} pA={pred.stage_a_probs['wet']:.3f} "
                f"pB={pred.stage_b_probs['flooded']:.3f} moved={moved}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
