# water segmentation data (train/val masks only)
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from inference.water import letterbox_image

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("fred", "roadway_flooding")


def resolve_path(path, root=ROOT):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def audit_splits(rows, relevant_sources=None):
    for field in ("group_id", "dup_cluster", "orig_path", "mask_path", "phash"):
        assignments = defaultdict(set)
        relevant = set()
        for row in rows:
            value = row.get(field, "")
            if value:
                key = (row["source"], value) if field == "group_id" else value
                assignments[key].add(row["split"])
                if relevant_sources is None or row["source"] in relevant_sources:
                    relevant.add(key)
        conflicts = [key for key, splits in assignments.items() if len(splits) > 1 and key in relevant]
        if conflicts:
            raise ValueError(f"Cross-split {field} leakage: {len(conflicts)} identities")


def read_manifest(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    audit_splits(rows, relevant_sources=SOURCES)
    return rows


def annotated_rows(rows, split):
    return [dict(r) for r in rows if r["split"] == split
            and r["source"] in SOURCES and r.get("mask_path") and r.get("orig_path")]


def capped_rows(rows, per_source=192, per_group=96, seed=42):
    selected = []
    for source in SOURCES:
        groups = defaultdict(list)
        for row in rows:
            if row["source"] == source:
                groups[row["group_id"]].append(row)
        for group in groups:
            groups[group].sort(key=lambda r: hashlib.sha256(
                f"{seed}:{r['orig_path']}".encode()).hexdigest())
            groups[group] = groups[group][:per_group]
        source_rows = []
        while any(groups.values()) and len(source_rows) < per_source:
            for group in sorted(groups):
                if groups[group] and len(source_rows) < per_source:
                    source_rows.append(groups[group].pop(0))
        selected.extend(source_rows)
    return selected


def original_pair(row, root=ROOT):
    if not row.get("mask_path"):
        raise ValueError("Pixel mask required; unannotated frames are excluded")
    with Image.open(resolve_path(row["orig_path"], root)) as image:
        image = image.convert("RGB")
    with Image.open(resolve_path(row["mask_path"], root)) as mask_image:
        values = np.asarray(mask_image)
        mask_size = mask_image.size
    if values.ndim == 3:
        if not np.all(values == values[..., :1]):
            raise ValueError("Unexpected multichannel annotation")
        values = values[..., 0]
    allowed = {0, 1, 2} if row["source"] == "fred" else {0, 1}
    if row["source"] not in SOURCES or not set(np.unique(values)).issubset(allowed):
        raise ValueError(f"Unknown mask semantics for {row['source']}")
    w, h = image.size
    mw, mh = mask_size
    if abs(w / h - mw / mh) > 0.01:
        raise ValueError("Mask/frame aspect mismatch; crop alignment must be reviewed")
    water = values == (2 if row["source"] == "fred" else 1)
    mask = Image.fromarray(water.astype(np.uint8)).resize(
        image.size, Image.Resampling.NEAREST
    )
    return image, np.asarray(mask, np.uint8)


def load_arrays(rows, size=320, root=ROOT):
    images, targets = [], []
    for row in rows:
        image, mask = original_pair(row, root)
        pixels, valid, geometry = letterbox_image(image, size)
        left, top, right, bottom = geometry["box"]
        target = np.zeros((size, size), np.float32)
        target[top:bottom, left:right] = np.asarray(Image.fromarray(mask).resize(
            (right - left, bottom - top), Image.Resampling.NEAREST
        ))
        images.append(pixels.astype(np.uint8))
        targets.append(np.stack((target, valid), axis=-1))
    if not images:
        raise ValueError("No pixel-annotated rows available")
    return np.stack(images), np.stack(targets)


def counts(rows):
    return dict(Counter(r["source"] for r in rows))

