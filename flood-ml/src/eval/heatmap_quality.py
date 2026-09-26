"""Validation-only raw CAM localization and occlusion diagnostics.

Run from flood-ml with PYTHONPATH=src ../my_env/bin/python -B -m eval.heatmap_quality.
Outputs stay in reports/heatmap_quality/ (ignored); models and labels are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw

from inference.preprocess import preprocess, to_pil

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)
LOGIT_EPS = 1e-6
LIMITATIONS = [
    ("CAM localizes positive evidence for the Stage B flood score, not water segmentation. "
    "Water-mask agreement is a diagnostic, not segmentation accuracy or a flood decision."),
    ("Only existing validation mask rows are used. No training, test evaluation, threshold "
    "selection, label changes, synthetic water masks, or confidence gating is performed."),
    ("Masked validation coverage is source-biased; FRED frames share sequences/locations. "
    "Per-image averages are descriptive, not independent confidence estimates."),
    ("Mask classes use nearest-neighbor geometry, including native mask-to-image scaling. "
    "Padding is excluded from localization denominators and measured separately."),
    ("Peak-in-water averages all exactly tied maxima. A flat positive CAM scores the water "
    "area baseline; zero CAM localization and occlusion are unavailable, never successes."),
    ("Mean-filled square occlusion can create out-of-distribution boundaries and remove "
    "context. High, low, and random squares have identical area; random squares can overlap "
    "high squares. Positive high-minus-random drop is limited faithfulness evidence."),
    ("Occlusion occurs after deployed preprocessing without a second JPEG pass. Logits "
    "are reconstructed from probability clipped to [1e-6, 1-1e-6]; saturation truncates "
    "drops. The model's pre-sigmoid tensor is not available."),
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(row):
    text = "\0".join(str(row.get(k, "")) for k in ("source", "orig_path", "mask_path"))
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def row_seed(seed, key, purpose):
    digest = hashlib.sha256(f"{seed}:{purpose}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def read_validation_rows(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "source", "orig_path", "mask_path", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("manifest is missing required columns")
        rows = [r for r in reader if r["split"] == "val"]
    masked = [r for r in rows if r["mask_path"].strip()]
    ids = [stable_key(r) for r in masked]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate validation image/mask rows")
    return masked, {"validation_rows": len(rows), "validation_mask_rows": len(masked),
                    "validation_without_mask": len(rows) - len(masked),
                    "mask_rows_by_source": dict(sorted(Counter(r["source"] for r in masked).items()))}


def select_per_source(rows, count, seed, purpose):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    result = []
    for source in sorted(groups):
        ordered = sorted(groups[source], key=lambda r: (row_seed(seed, stable_key(r), purpose),
                                                       stable_key(r)))
        result.extend(ordered if count is None else ordered[:count])
    return result


def preprocess_options(config):
    options = config.get("preprocess", {})
    return {"mode": options.get("mode", "crop"),
            "size": int(options.get("size", config.get("input", {}).get("size", 224))),
            "do_jpeg_roundtrip": options.get("jpeg_roundtrip", True)}


def load_water_mask(path, source, restricted=False):
    with Image.open(path) as image:
        mask = np.asarray(image).copy()
        if source == "flood_area_segmentation" and not restricted:
            return np.asarray(image.convert("L")) > 127
    if mask.ndim == 3 and np.all(mask == mask[..., :1]):
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError("mask must contain scalar class IDs")
    if source == "fred":
        if not np.isin(mask, [0, 1, 2]).all():
            raise ValueError("unexpected FRED mask classes")
        return mask == 2
    if source in {"roadway_flooding", "flood_area_segmentation", "flood_master_test"}:
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("expected binary 0/1 water mask")
        return mask == 1
    raise ValueError(f"unsupported mask semantics: {source}")


def align_mask(mask, geom):
    """Map native full-frame classes into the exact inference input geometry."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not all(mask.shape):
        raise ValueError("mask must be nonempty and 2D")
    original = tuple(geom["orig_size"])
    mh, mw = mask.shape
    ow, oh = original
    if abs(mw / mh - ow / oh) > 1 / mh + 1 / oh:
        raise ValueError("mask/image aspect mismatch; unknown registration")
    nearest = Image.Resampling.NEAREST
    native = Image.fromarray(mask.astype(np.uint8)).resize(original, nearest)
    valid = Image.new("L", original, 1)
    size = int(geom["size"])

    def forward(image):
        if geom["mode"] == "crop":
            resized = image.resize(tuple(geom["resized_size"]), nearest)
            return resized.crop(tuple(geom["crop_box_resized"]))
        if geom["mode"] == "squash":
            return image.resize((size, size), nearest)
        if geom["mode"] == "letterbox":
            left, top, right, bottom = geom["content_box_canvas"]
            canvas = Image.new("L", (size, size))
            canvas.paste(image.resize((right - left, bottom - top), nearest), (left, top))
            return canvas
        raise ValueError("unknown preprocessing geometry")

    return np.asarray(forward(native), dtype=bool), np.asarray(forward(valid), dtype=bool)


def upsample_cam(cam, shape):
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or not all(cam.shape):
        raise ValueError("CAM must be nonempty and 2D")
    if not np.isfinite(cam).all() or (cam < 0).any():
        raise ValueError("named raw CAM must be finite and nonnegative")
    image = Image.fromarray(cam).resize(tuple(shape[::-1]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float64)


def cam_metrics(cam, water, valid=None):
    water = np.asarray(water, dtype=bool)
    valid = np.ones_like(water) if valid is None else np.asarray(valid, dtype=bool)
    if water.ndim != 2 or water.shape != valid.shape or not valid.any():
        raise ValueError("water and valid masks must match with some real content")
    water = water & valid
    heat = upsample_cam(cam, water.shape)
    total = float(heat.sum())
    content = float(heat[valid].sum())
    area = float(water.sum() / valid.sum())
    result = {"available": False, "reason": None, "water_area_fraction": area,
              "water_pixels": int(water.sum()), "valid_pixels": int(valid.sum()),
              "raw_cam_sum": float(np.asarray(cam, dtype=np.float64).sum()),
              "padding_energy_fraction": float(heat[~valid].sum() / total) if total else None,
              "water_energy_fraction": None, "energy_minus_area": None,
              "energy_over_area": None, "peak_in_water": None, "peak_tie_pixels": None}
    if content == 0:
        result["reason"] = "zero_cam" if total == 0 else "no_content_energy"
        return result
    energy = float(heat[water].sum() / content)
    peaks = valid & (heat == heat[valid].max())
    result.update(available=True, water_energy_fraction=energy, energy_minus_area=energy - area,
                  energy_over_area=energy / area if area else None,
                  peak_in_water=float(water[peaks].mean()), peak_tie_pixels=int(peaks.sum()))
    return result


def safe_logit(probability):
    probability = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("expected finite probabilities in [0,1]")
    clipped = np.clip(probability, LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(clipped) - np.log1p(-clipped)


def occlusion_boxes(cam, valid, fraction, rng, random_controls):
    if not 0 < fraction < 1 or random_controls < 1:
        raise ValueError("occlusion needs 0 < area fraction < 1 and random controls >= 1")
    valid = np.asarray(valid, dtype=bool)
    ys, xs = np.where(valid)
    if not len(ys):
        raise ValueError("no valid content")
    left, top, right, bottom = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    if not valid[top:bottom, left:right].all():
        raise ValueError("occlusion requires rectangular real-image content")
    side = min(right - left, bottom - top, max(1, round(math.sqrt(valid.sum() * fraction))))
    heat = upsample_cam(cam, valid.shape)[top:bottom, left:right]
    integral = np.pad(heat, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    scores = (integral[side:, side:] - integral[:-side, side:]
              - integral[side:, :-side] + integral[:-side, :-side])

    def box(index):
        y, x = np.unravel_index(int(index), scores.shape)
        return [int(left + x), int(top + y), int(left + x + side), int(top + y + side)]

    high = rng.choice(np.flatnonzero(scores == scores.max()))
    low = rng.choice(np.flatnonzero(scores == scores.min()))
    random = rng.integers(0, scores.size, size=random_controls)
    return [box(high), box(low)] + [box(i) for i in random]


def occlusion_diagnostic(score, image, cam, valid, baseline_probability, *, seed,
                         fraction=0.10, random_controls=5):
    metrics = cam_metrics(cam, np.zeros_like(valid), valid)
    heat = upsample_cam(cam, valid.shape)
    if not metrics["available"]:
        return {"available": False, "reason": metrics["reason"]}
    if np.ptp(heat[valid]) == 0:
        return {"available": False, "reason": "flat_cam"}
    boxes = occlusion_boxes(cam, valid, fraction, np.random.default_rng(seed), random_controls)
    fill = np.asarray(image)[valid].mean(axis=0, dtype=np.float64).astype(np.float32)
    batch = np.repeat(np.asarray(image, dtype=np.float32)[None], len(boxes), axis=0)
    for altered, (left, top, right, bottom) in zip(batch, boxes):
        altered[top:bottom, left:right] = fill
    probabilities = np.asarray(score(batch), dtype=np.float64).reshape(-1)
    if len(probabilities) != len(boxes):
        raise ValueError("model returned unexpected probability batch")
    drops = safe_logit(baseline_probability) - safe_logit(probabilities)
    probability_drops = baseline_probability - probabilities
    high = boxes[0]
    area = (high[2] - high[0]) * (high[3] - high[1])
    overlaps = [max(0, min(high[2], b[2]) - max(high[0], b[0]))
                * max(0, min(high[3], b[3]) - max(high[1], b[1])) / area for b in boxes[2:]]
    def clipped(p):
        return (np.asarray(p) < LOGIT_EPS) | (np.asarray(p) > 1 - LOGIT_EPS)

    return {"available": True, "reason": None, "occluded_pixels": int(area),
            "actual_area_fraction": float(area / valid.sum()), "boxes_high_low_random": boxes,
            "fill_rgb": fill.tolist(), "baseline_probability": float(baseline_probability),
            "baseline_logit_clipped": bool(clipped(baseline_probability)),
            "perturbed_logit_clipped_count": int(clipped(probabilities).sum()),
            "high_logit_drop": float(drops[0]), "low_logit_drop": float(drops[1]),
            "random_logit_drops": drops[2:].tolist(),
            "random_logit_drop_mean": float(drops[2:].mean()),
            "high_minus_random_logit_drop": float(drops[0] - drops[2:].mean()),
            "high_minus_low_logit_drop": float(drops[0] - drops[1]),
            "high_probability_drop": float(probability_drops[0]),
            "random_probability_drop_mean": float(probability_drops[2:].mean()),
            "random_overlap_with_high_mean": float(np.mean(overlaps))}


class StageB:
    def __init__(self, model_dir):
        import onnxruntime as ort

        self.config_path = Path(model_dir) / "config.json"
        self.config = json.loads(self.config_path.read_text())
        self.model_path = Path(model_dir) / self.config["stage_b"]["onnx_path"]
        self.hashes = {"config": sha256_file(self.config_path),
                       "stage_b_onnx": sha256_file(self.model_path)}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(self.model_path), options,
                                            providers=["CPUExecutionProvider"])
        contract = self.config.get("onnx", {})
        self.input_name = contract.get("input_name", "image")
        outputs = contract.get("output_names", {})
        self.cam_name = outputs.get("cam", "cam")
        self.prob_name = outputs.get("prob", "prob")
        if not {self.cam_name, self.prob_name}.issubset(o.name for o in self.session.get_outputs()):
            raise ValueError("deployed model lacks named CAM/probability outputs")

    def predict(self, image):
        cam, probability = self.session.run([self.cam_name, self.prob_name],
                                            {self.input_name: image[None]})
        probability = float(np.asarray(probability).reshape(-1)[0])
        safe_logit(probability)
        return probability, cam[0]

    def score(self, batch):
        return self.session.run([self.prob_name], {self.input_name: batch})[0].reshape(-1)

    def verify_unchanged(self):
        current = {"config": sha256_file(self.config_path),
                   "stage_b_onnx": sha256_file(self.model_path)}
        if self.hashes != current:
            raise RuntimeError("model/config changed during evaluation; rerun")


def segmentation_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    if prediction.shape != truth.shape or truth.ndim != 2:
        raise ValueError("segmentation and baseline mask shapes must match")
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    union = tp + fp + fn
    return {"available": True, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": tp / union if union else None,
            "dice": 2 * tp / (2 * tp + fp + fn) if union else None,
            "empty_truth": tp + fn == 0, "both_empty": union == 0}


class WaterSegmentation:
    def __init__(self, directory):
        from inference.water import load_water_model

        self.directory = Path(directory)
        self.model = load_water_model(self.directory)
        self.hashes = {name: sha256_file(self.directory / name)
                       for name in ("water.onnx", "config.json")} if self.model else {}

    def evaluate(self, image, mask):
        from inference.water import letterbox_image, restore_probability

        size = int(self.model.config["input_size"])
        array, _, geom = letterbox_image(image, size)
        probability = self.model.session.run(["water_prob"], {"image": array[None]})[0]
        if (probability.shape != (1, size, size, 1) or not np.isfinite(probability).all()
                or ((probability < 0) | (probability > 1)).any()):
            raise ValueError("invalid segmentation probability map")
        probability = restore_probability(probability[0, ..., 0], geom)
        truth = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
            image.size, Image.Resampling.NEAREST), dtype=bool)
        return segmentation_metrics(probability >= float(self.model.config["threshold"]), truth)

    def metadata(self):
        if not self.model:
            return {"available": False, "reason": "artifact_not_available"}
        if any(sha256_file(self.directory / name) != digest for name, digest in self.hashes.items()):
            raise RuntimeError("water model/config changed during evaluation; rerun")
        return {"available": True, "model_version": self.model.config.get("model_version"),
                "threshold": self.model.config["threshold"], "threshold_tuned_here": False,
                "input_size": self.model.config["input_size"], "fingerprints": self.hashes,
                "comparison": "Full original frame; baseline mask nearest-neighbor resize; "
                "inference.water letterbox + probability restoration, then configured threshold. "
                "Independent of CAM crop and flood probabilities. Both-empty IoU/Dice are null; "
                "empty agreements counted separately. No road-extent claim."}


def segmentation_summary(records):
    rows = [r["segmentation"] for r in records if r.get("segmentation", {}).get("available")]
    counts = {key: sum(r[key] for r in rows) for key in ("tp", "fp", "fn", "tn")}
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    union = tp + fp + fn
    return {"n": len(rows), "counts": counts, "micro_iou": tp / union if union else None,
            "micro_dice": 2 * tp / (2 * tp + fp + fn) if union else None,
            "macro_iou": stats(r["iou"] for r in rows), "macro_dice": stats(r["dice"] for r in rows),
            "water_present_macro_iou": stats(r["iou"] for r in rows if not r["empty_truth"]),
            "water_present_macro_dice": stats(r["dice"] for r in rows if not r["empty_truth"]),
            "empty_truth": sum(r["empty_truth"] for r in rows),
            "both_empty": sum(r["both_empty"] for r in rows)}


def data_versions(config, manifest):
    version_file = Path(manifest).parent / "VERSION"
    manifest_version = version_file.read_text().strip() if version_file.is_file() else None
    if manifest_version and manifest_version.startswith("data_version ="):
        manifest_version = manifest_version.split("=", 1)[1].strip()
    deployed_version = config.get("data_version")
    return {"deployed_model": deployed_version, "manifest_sidecar": manifest_version,
            "match": deployed_version == manifest_version
            if deployed_version is not None and manifest_version is not None else None}


def stats(values):
    values = [float(v) for v in values if v is not None]
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None}


def summarize(records):
    available = [r["cam"] for r in records if r.get("cam", {}).get("available")]
    water = [r for r in available if r["water_pixels"] > 0]
    occlusions = [r["occlusion"] for r in records if r.get("occlusion") is not None]
    measured = [r for r in occlusions if r["available"]]
    reasons = Counter(r.get("error") or r.get("cam", {}).get("reason")
                      for r in records if not r.get("cam", {}).get("available"))
    return {"n": len(records), "labels": dict(sorted(Counter(r["label"] for r in records).items())),
            "groups": len({r["group"] for r in records}),
            "cam_available": len(available), "cam_unavailable_reasons": dict(sorted(reasons.items())),
            "water_present": sum(r.get("cam", {}).get("water_pixels", 0) > 0 for r in records),
            "water_present_cam_available": len(water),
            "water_absent_cam_available": len(available) - len(water),
            "all_available": {key: stats(r[key] for r in available) for key in
                              ("water_area_fraction", "water_energy_fraction", "peak_in_water",
                               "padding_energy_fraction")},
            "water_present_available": {key: stats(r[key] for r in water) for key in
                                        ("water_area_fraction", "water_energy_fraction",
                                         "energy_minus_area", "energy_over_area", "peak_in_water")},
            "occlusion": {"selected": len(occlusions), "available": len(measured),
                          "unavailable_reasons": dict(sorted(Counter(
                              r["reason"] for r in occlusions if not r["available"]).items())),
                          "baseline_logit_clipped": sum(r["baseline_logit_clipped"] for r in measured),
                          "high_drop_exceeds_random_fraction": stats(
                              r["high_minus_random_logit_drop"] > 0 for r in measured)["mean"],
                          "perturbed_logit_clipped": sum(
                              r["perturbed_logit_clipped_count"] for r in measured),
                          **{key: stats(r[key] for r in measured) for key in
                             ("high_logit_drop", "low_logit_drop", "random_logit_drop_mean",
                              "high_minus_random_logit_drop", "high_minus_low_logit_drop")}}}


def contact_tile(image, water, cam, record):
    base = np.clip(image, 0, 255).astype(np.uint8)
    heat = upsample_cam(cam, water.shape)
    heat = heat / heat.max() if heat.max() else heat
    tint = np.asarray([255, 65, 0], dtype=float)
    overlay = np.clip(base * (1 - .65 * heat[..., None])
                      + tint * (.65 * heat[..., None]), 0, 255).astype(np.uint8)
    mask_rgb = np.where(water[..., None], np.array([30, 195, 225]), base * .45).astype(np.uint8)
    size = base.shape[0]
    tile = Image.new("RGB", (3 * size, size + 44), "white")
    for index, pixels in enumerate((base, mask_rgb, overlay)):
        tile.paste(Image.fromarray(pixels), (index * size, 44))
    draw = ImageDraw.Draw(tile)
    draw.text((5, 2), f"{record['id']} {record['source']} {record['label']} "
              f"pB={record['stage_b_probability']:.5f}", fill="black")
    draw.text((5, 19), "Model view | baseline water mask (cyan) | raw CAM (display / max)",
              fill="black")
    return tile


def markdown_report(report):
    def fmt(value):
        return "unavailable" if value is None else f"{value:.4f}"

    lines = ["# Validation CAM quality", "", "Raw Stage B CAM; no confidence gating.", "",
             (f"Validation mask rows: {report['coverage']['validation_mask_rows']}; "
             f"evaluated: {len(report['records'])}. Seed: {report['settings']['seed']}."), "",
             ("Water-present rows with available CAM only in the energy/peak columns; "
             "availability denominator includes every selected row."), "",
             ("| Source | Rows | CAM available | Water + CAM | Mean water area | "
             "Mean water energy | Mean energy - area | Mean peak-in-water |"),
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for source, summary in report["by_source"].items():
        water = summary["water_present_available"]
        values = [fmt(water[k]["mean"]) for k in ("water_area_fraction", "water_energy_fraction",
                                                 "energy_minus_area", "peak_in_water")]
        lines.append(f"| {source} | {summary['n']} | {summary['cam_available']} | "
                     f"{summary['water_present_cam_available']} | " + " | ".join(values) + " |")
    lines.extend(["", "## Equal-area occlusion", "",
                  ("Positive high-minus-random logit drop favors the high-CAM square. "
                  "Random controls are uniform square locations, not water masks."), "",
                  ("| Source | Selected | Available | Clipped baseline | Mean high drop | "
                  "Mean random drop | Mean high - random | High > random fraction |"),
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for source, summary in report["by_source"].items():
        occ = summary["occlusion"]
        values = [fmt(occ[k]["mean"]) for k in ("high_logit_drop", "random_logit_drop_mean",
                                               "high_minus_random_logit_drop")]
        lines.append(f"| {source} | {occ['selected']} | {occ['available']} | "
                     f"{occ['baseline_logit_clipped']} | " + " | ".join(values)
                     + f" | {fmt(occ['high_drop_exceeds_random_fraction'])} |")
    if report["water_model"].get("available"):
        lines.extend(["", "## Separate water segmentation", "",
                      report["water_model"]["comparison"], "",
                      "| Source | Available | Micro IoU | Micro Dice | Both empty |",
                      "|---|---:|---:|---:|---:|"])
        for source, value in report["segmentation_by_source"].items():
            lines.append(f"| {source} | {value['n']} | {fmt(value['micro_iou'])} | "
                         f"{fmt(value['micro_dice'])} | {value['both_empty']} |")
    lines.extend(["", "## Data versions", "", "```json",
                  json.dumps(report["data_versions"], indent=2), "```", "",
                  ("A version mismatch means this report uses the current validation manifest, "
                  "not necessarily the model's original selection snapshot.")])
    lines.extend(["", "## Availability", "", "```json",
                  json.dumps({s: {"cam": r["cam_unavailable_reasons"],
                                   "occlusion": r["occlusion"]["unavailable_reasons"]}
                              for s, r in report["by_source"].items()}, indent=2), "```", "",
                  "## Reproduction", "", "```json", json.dumps(report["settings"], indent=2),
                  "```", "", ("Hashes, source/label breakdowns, individual metrics, and "
                  "occlusion rectangles are in metrics.json; no image bytes or paths are embedded."),
                  "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in LIMITATIONS)
    return "\n".join(lines) + "\n"


def run_report(args):
    output = Path(args.output_dir).resolve()
    if output.parent != (ROOT / "reports").resolve() or not output.name.startswith("heatmap_quality"):
        raise ValueError("output-dir must be a reports/heatmap_quality* directory")
    output.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(output / "run.log", mode="w")
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    try:
        model = StageB(args.model_dir)
        options = preprocess_options(model.config)
        versions = data_versions(model.config, args.manifest)
        water_model = WaterSegmentation(args.water_model) if args.water_model else None
        rows, coverage = read_validation_rows(args.manifest)
        rows = select_per_source(rows, args.max_per_source or None, args.seed, "localization")
        occlusion_ids = {stable_key(r) for r in select_per_source(
            rows, args.occlusion_per_source, args.seed, "occlusion-sample")}
        sheet_rows = select_per_source(rows, None, args.seed, "contact-sheet")
        sheet_groups = defaultdict(list)
        for row in sheet_rows:
            sheet_groups[row["source"]].append(stable_key(row))
        interleaved = [group[i] for i in range(max(map(len, sheet_groups.values()), default=0))
                       for group in sheet_groups.values() if i < len(group)]
        sheet_ids = set(interleaved[:args.contact_sheet])
        records, tiles = [], []
        data_digest = hashlib.sha256()
        for row in rows:
            key = stable_key(row)
            group = row.get("group_id") or key
            record = {"id": key, "source": row["source"], "label": row["label"],
                      "group": hashlib.sha256(group.encode()).hexdigest()[:16], "error": None,
                      "occlusion": {"available": False, "reason": "input_unavailable"}
                      if key in occlusion_ids else None}
            try:
                image_path = ROOT / row["orig_path"]
                mask_path = ROOT / row["mask_path"]
                if not row["orig_path"].strip():
                    raise ValueError("original image is required")
                image = to_pil(image_path)
                mask = load_water_mask(mask_path, row["source"],
                                       row.get("restricted", "").lower() in {"true", "1"})
                data_digest.update((key + sha256_file(image_path) + sha256_file(mask_path)).encode())
                array, geom = preprocess(image, **options)
                water, valid = align_mask(mask, geom)
                probability, cam = model.predict(array)
                record.update(stage_b_probability=probability, cam_shape=list(cam.shape),
                              image_size=list(image.size), mask_size=list(mask.shape[::-1]),
                              cam=cam_metrics(cam, water, valid))
                if key in occlusion_ids:
                    record["occlusion"] = occlusion_diagnostic(
                        model.score, array, cam, valid, probability,
                        seed=row_seed(args.seed, key, "occlusion-controls"),
                        fraction=args.occlusion_fraction, random_controls=args.random_controls)
                if water_model and water_model.model:
                    try:
                        record["segmentation"] = water_model.evaluate(image, mask)
                    except (OSError, ValueError) as error:
                        record["segmentation"] = {"available": False, "reason": type(error).__name__}
                        LOG.exception("unavailable segmentation %s", key)
                if key in sheet_ids:
                    tiles.append(contact_tile(array, water, cam, record))
            except (OSError, ValueError) as error:
                record["error"] = type(error).__name__
                LOG.exception("unavailable row %s", key)
            records.append(record)
            LOG.info("row %d/%d %s", len(records), len(rows), key)
        model.verify_unchanged()
        import onnxruntime as ort

        report = {"schema_version": 1, "split": "val", "stage": "B", "raw_cam": True,
                  "confidence_gated": False, "coverage": coverage, "data_versions": versions,
                  "water_model": water_model.metadata() if water_model else
                  {"available": False, "reason": "not_requested"},
                  "segmentation_overall": segmentation_summary(records),
                  "segmentation_by_source": {},
                  "settings": {"seed": args.seed, "max_per_source": args.max_per_source,
                               "occlusion_per_source": args.occlusion_per_source,
                               "occlusion_fraction": args.occlusion_fraction,
                               "random_controls": args.random_controls,
                               "contact_sheet": args.contact_sheet,
                               "preprocess": options, "logit_clip_epsilon": LOGIT_EPS,
                               "onnx_execution": "CPU, sequential, one thread"},
                  "fingerprints": {**model.hashes, "evaluated_data": data_digest.hexdigest(),
                                   "evaluator": sha256_file(__file__),
                                   "preprocess": sha256_file(ROOT / "src/inference/preprocess.py"),
                                   "validation_selection": hashlib.sha256(json.dumps(
                                       rows, sort_keys=True).encode()).hexdigest()},
                  "versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "pillow": PIL.__version__, "onnxruntime": ort.__version__},
                  "overall": summarize(records), "by_source": {}, "by_source_label": {},
                  "records": records, "limitations": LIMITATIONS}
        for source in sorted({r["source"] for r in records}):
            subset = [r for r in records if r["source"] == source]
            report["by_source"][source] = summarize(subset)
            report["segmentation_by_source"][source] = segmentation_summary(subset)
            report["by_source_label"][source] = {
                label: summarize([r for r in subset if r["label"] == label])
                for label in sorted({r["label"] for r in subset})}
        (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (output / "report.md").write_text(markdown_report(report))
        if tiles:
            sheet = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles)))
            y = 0
            for tile in tiles:
                sheet.paste(tile, (0, y))
                y += tile.height
            sheet.save(output / "contact_sheet.jpg", quality=90)
        return report
    finally:
        handler.close()
        LOG.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifest.csv")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/heatmap_quality")
    parser.add_argument("--water-model", type=Path, default=None,
                        help="optional separate water model directory (config.json + water.onnx)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-source", type=int, default=0, help="0 evaluates all validation masks")
    parser.add_argument("--occlusion-per-source", type=int, default=16)
    parser.add_argument("--random-controls", type=int, default=5)
    parser.add_argument("--occlusion-fraction", type=float, default=0.10)
    parser.add_argument("--contact-sheet", type=int, default=8, help="0 disables local image output")
    args = parser.parse_args()
    if (min(args.max_per_source, args.occlusion_per_source, args.contact_sheet) < 0
            or args.random_controls < 1 or not 0 < args.occlusion_fraction < 1):
        parser.error("sample counts must be nonnegative; controls >= 1; fraction must be in (0,1)")
    report = run_report(args)
    print(f"Evaluated {len(report['records'])} validation mask rows; "
          f"CAM available {report['overall']['cam_available']}. Reports: {args.output_dir}")


if __name__ == "__main__":
    main()
