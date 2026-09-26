"""Binary pixel metrics, independent of classification."""
from collections import defaultdict

import numpy as np


def mask_counts(prediction, truth, valid=None):
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    valid = np.ones_like(truth, bool) if valid is None else np.asarray(valid, bool)
    return {
        "tp": int(np.count_nonzero(prediction & truth & valid)),
        "fp": int(np.count_nonzero(prediction & ~truth & valid)),
        "fn": int(np.count_nonzero(~prediction & truth & valid)),
        "tn": int(np.count_nonzero(~prediction & ~truth & valid)),
    }


def scores(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
    }


def summarize(records):
    result = {}
    groups = defaultdict(list)
    for record in records:
        groups[record["source"]].append(record)
        groups["overall"].append(record)
    for source, group in groups.items():
        total = {k: sum(r[k] for r in group) for k in ("tp", "fp", "fn", "tn")}
        per_image = [scores(r) for r in group]
        result[source] = {
            "n": len(group), **total, **scores(total),
            "mean_image_iou": float(np.mean([r["iou"] for r in per_image])),
            "mean_image_dice": float(np.mean([r["dice"] for r in per_image])),
            "empty_truth_images": sum(r["tp"] + r["fn"] == 0 for r in group),
        }
    return result


def canvas_metrics(probabilities, targets, rows, threshold):
    return summarize([
        {"source": row["source"], **mask_counts(
            p[..., 0] >= threshold, target[..., 0], target[..., 1])}
        for p, target, row in zip(probabilities, targets, rows)
    ])
