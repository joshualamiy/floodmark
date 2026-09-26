# water model eval (test needs explicit opt-in)
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inference.water import letterbox_image, load_water_model, restore_probability
from train.water_data import annotated_rows, original_pair, read_manifest
from train.water_metrics import mask_counts, summarize


def evaluate(model_dir, manifest, split="val", allow_test=False):
    if split not in ("val", "test"):
        raise ValueError("Only validation or independent held-out evaluation is supported")
    if split == "test" and not allow_test:
        raise ValueError("Test evaluation is reserved for the independent Phase4 reviewer")
    model = load_water_model(model_dir)
    if model is None or not model.config.get("frozen"):
        raise ValueError("A frozen water artifact is required")
    path = Path(model_dir) / "water.onnx"
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != model.config["onnx_sha256"]:
        raise ValueError("Frozen model checksum mismatch")
    if hashlib.sha256(Path(manifest).read_bytes()).hexdigest() != model.config["manifest_sha256"]:
        raise ValueError("Manifest changed since model freeze")
    rows = annotated_rows(read_manifest(manifest), split)
    if not rows:
        raise ValueError("No supported pixel annotations")
    records = []
    for row in rows:
        image, truth = original_pair(row)
        pixels, _, geometry = letterbox_image(image, model.config["input_size"])
        probability = model.session.run(["water_prob"], {"image": pixels[None]})[0][0, ..., 0]
        prediction = restore_probability(probability, geometry) >= model.config["threshold"]
        records.append({"source": row["source"], "orig_path": row["orig_path"],
                        **mask_counts(prediction, truth)})
    return {"split": split, "geometry": "original_frame_pixels",
            "threshold": model.config["threshold"], "model_version": model.config["model_version"],
            "road": False, "metrics": summarize(records),
            "empty_empty_metric": 1.0, "per_image": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/water")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = evaluate(args.model_dir, args.manifest, args.split, args.allow_test)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()

