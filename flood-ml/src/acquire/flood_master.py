"""Flood Master Database (FMD): Google Drive listing, integrity checks, and
cross-referencing against our public dataset downloads.

RESTRICTED DATA. The Flood Master Database was created by the AIIA Lab,
Aristotle University of Thessaloniki (AUTH), and is licensed to this team for
non-commercial research only. See docs/FLOOD_MASTER.md for the license
summary and credit line. This module must never print, log, or write raw
pixel data, dataset contact information, or full annotation contents -- only
paths, counts, and aggregate statistics, all of which stay under
data/restricted/ (gitignored).

Usage (run from flood-ml/, with src/ on PYTHONPATH):

    PYTHONPATH=src ../my_env/bin/python -m acquire.flood_master list-drive
    PYTHONPATH=src ../my_env/bin/python -m acquire.flood_master download-drive
    PYTHONPATH=src ../my_env/bin/python -m acquire.flood_master diff-drive
    PYTHONPATH=src ../my_env/bin/python -m acquire.flood_master verify
    PYTHONPATH=src ../my_env/bin/python -m acquire.flood_master build-index

Each subcommand is idempotent and safe to re-run. Long output goes to
logs/jobs/flood_master_<subcommand>.log; the console only gets a summary.

Contract for Phase 2 (data prep): `data/restricted/flood_master/index.csv`
has one row per FMD annotation. For train/val rows, `local_image_path` is
the path of the public image the FMD mask applies to (under data/raw/).
Phase 2 should join its per-source manifest rows to this index on
`local_image_path`; where a match exists, drop the public-dataset row's own
mask in favor of `fmd_mask_path` (the FMD "cleaned" mask), and keep exactly
one manifest row for that image. Rows with match_method="unresolved" have no
usable local image and should be skipped. Test rows (`fmd_split=="test"`)
have no `public_mask_path"` (FMD test masks have no public counterpart);
their `local_image_path` is the FMD frame itself, and `group_id` must be
kept intact as a single group per video when splitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger("acquire.flood_master")

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1JDGK1DQUzRNqFsir0ptQfeSMQF9kMV9Q"

FMD_ROOT = Path("data/restricted/flood_master")
FMD_DRIVE_ROOT = Path("data/restricted/flood_master_drive")
RAW_ROOT = Path("data/raw")
LOG_DIR = Path("logs/jobs")

TRAIN_CSV = FMD_ROOT / "train" / "train.csv"
VAL_CSV = FMD_ROOT / "val" / "val.csv"
TEST_CSV = FMD_ROOT / "test" / "test.csv"

MAX_DRIVE_DOWNLOAD_BYTES = 5 * 1024**3  # 5 GB guardrail from the brief

# Public raw folders used to resolve FMD train/val "Image path" values and to
# check that FMD test frames don't duplicate anything we already downloaded.
FAS_IMAGE_DIR = RAW_ROOT / "flood_area_segmentation" / "Image"
FAS_MASK_DIR = RAW_ROOT / "flood_area_segmentation" / "Mask"
ROADWAY_ROOT = RAW_ROOT / "roadway_flooding"
WATER_V1_JPEG = RAW_ROOT / "water_segmentation" / "water_v1" / "water_v1" / "JPEGImages"
WATER_V1_ANN = RAW_ROOT / "water_segmentation" / "water_v1" / "water_v1" / "Annotations"
WATER_V2_JPEG = RAW_ROOT / "water_segmentation" / "water_v2" / "water_v2" / "JPEGImages"
WATER_V2_ANN = RAW_ROOT / "water_segmentation" / "water_v2" / "water_v2" / "Annotations"
FRED_ROOT = RAW_ROOT / "fred"

PHASH_HAMMING_THRESHOLD = 6


def _setup_file_logger(name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"flood_master_{name}.log"
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger("acquire.flood_master")
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    return log_path


def read_fmd_csv(path: Path) -> list[dict[str, str]]:
    """Read a train/val/test CSV (BOM-safe), stripping stray whitespace."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [{k: v.strip() for k, v in row.items()} for row in csv.DictReader(fh)]


# --------------------------------------------------------------------------
# Path resolution: FMD row -> local file
# --------------------------------------------------------------------------


def resolve_train_val_annotation(split: str, row: dict[str, str]) -> Path:
    """Resolve a train/val row's "Annotation path" to a file on disk.

    Most rows point into data/restricted/flood_master/<split>/annotations/.
    Roadway Flooding rows instead reuse the original dataset's own label
    path (data/raw/roadway_flooding/Dataset/labels/...), per readme.txt.
    """
    ann = row["Annotation path"]
    if ann.startswith("Dataset/labels/"):
        return ROADWAY_ROOT / ann
    return FMD_ROOT / split / ann


def resolve_train_val_public_image(row: dict[str, str]) -> Path | None:
    """Resolve a train/val row's "Image path" to our local public download.

    Returns None when we don't have that image locally (this happens for
    the "flooding_pixabay" subset of the Water Dataset source, see the
    report: that subset isn't in our Kaggle water_v1/water_v2 mirrors).
    """
    source = row["Source"]
    img = row["Image path"]
    if source == "Flood Area Segmentation":
        p = FAS_IMAGE_DIR.parent / img  # img already starts with "Image/"
        return p if p.exists() else None
    if source == "Roadway Flooding Image Dataset":
        p = ROADWAY_ROOT / img
        return p if p.exists() else None
    if source == "Water Dataset":
        marker = "JPEGImages/"
        idx = img.find(marker)
        if idx < 0:
            return None
        suffix = img[idx + len(marker) :]
        for jpeg_dir in (WATER_V2_JPEG, WATER_V1_JPEG):
            p = jpeg_dir / suffix
            if p.exists():
                return p
        return None
    return None


def resolve_train_val_public_mask(row: dict[str, str], public_image: Path | None) -> Path | None:
    """Resolve the *public* mask/label that shipped with the source dataset
    (as opposed to FMD's own cleaned mask), for the IoU comparison.

    Roadway Flooding has no separate public mask to compare: FMD literally
    reuses that dataset's own label path as its "annotation path" (same
    file), so IoU there is 1.0 by construction and not informative.
    """
    source = row["Source"]
    if public_image is None:
        return None
    if source == "Flood Area Segmentation":
        p = FAS_MASK_DIR / f"{public_image.stem}.png"
        return p if p.exists() else None
    if source == "Water Dataset":
        # Swap .../JPEGImages/... for .../Annotations/... at the same depth.
        parts = list(public_image.parts)
        try:
            i = parts.index("JPEGImages")
        except ValueError:
            return None
        parts[i] = "Annotations"
        p = Path(*parts).with_suffix(".png")
        return p if p.exists() else None
    return None


def resolve_test_rgb(row: dict[str, str]) -> Path:
    sub = "greek_test" if row["Source"] == "greek video" else "italian_test"
    return FMD_ROOT / "test" / sub / "rgb" / row["Image path"]


def resolve_test_annotation(row: dict[str, str]) -> Path:
    return FMD_ROOT / "test" / row["Annotation path"]


def test_group_id(row: dict[str, str]) -> str:
    return "fmd_greek_video" if row["Source"] == "greek video" else "fmd_italian_video"


# --------------------------------------------------------------------------
# Drive listing / download / diff
# --------------------------------------------------------------------------


def list_drive_folder(url: str = DRIVE_FOLDER_URL) -> list[dict[str, str]]:
    """List every file in the shared Drive folder (metadata only, no bytes).

    gdown 6.4.0's download_folder(skip_download=True) recurses the whole
    folder tree in one call and returns every file, including subfolders;
    there is no 50-file cap in this version (verified empirically: it
    returned all 4320 files in the FMD folder in one call). If a future
    gdown version reintroduces pagination, this will need `remaining_ok`
    or manual paging; check the installed version's
    `gdown.download_folder` signature first.
    """
    import gdown

    result = gdown.download_folder(url, skip_download=True, quiet=True)
    if result is None:
        raise RuntimeError(
            "gdown returned no listing for the Drive folder. Either the folder "
            "is inaccessible with the current account/cookies, or gdown's API "
            "changed. Do not attempt to bypass any access gate; report this."
        )
    return [{"id": f.id, "path": f.path} for f in result]


def download_drive_folder(
    url: str = DRIVE_FOLDER_URL, output: Path = FMD_DRIVE_ROOT, max_bytes: int = MAX_DRIVE_DOWNLOAD_BYTES
) -> Path:
    """Download the whole Drive folder to `output`, keeping its structure.

    Caller should already know (via list_drive_folder + local file count)
    that this is a reasonable size. Google Drive enforces a per-file /
    per-folder download quota for heavily-accessed shared links ("Cannot
    retrieve the public link of the file ... may have had many accesses").
    That quota is not something we can or should work around (no bypassing
    access gates); if it triggers, this function will raise/log failures
    per file and the caller should retry later.
    """
    import gdown

    output.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(url, output=str(output), quiet=False, use_cookies=False)
    return output


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def diff_drive_vs_local(
    drive_root: Path = FMD_DRIVE_ROOT, local_root: Path = FMD_ROOT
) -> dict[str, Any]:
    """Byte-level (md5) diff of whatever has actually been downloaded under
    `drive_root` against the matching path in `local_root`. Only covers
    files present in `drive_root` -- run `download_drive_folder` first for
    full coverage; a partial download (e.g. stopped by Drive's quota) still
    yields a partial, honestly-labeled diff.
    """
    report: dict[str, Any] = {"checked": 0, "identical": 0, "different": [], "no_local_counterpart": []}
    if not drive_root.exists():
        report["note"] = f"{drive_root} does not exist; nothing downloaded to diff."
        return report
    for path in drive_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(drive_root)
        local = local_root / rel
        report["checked"] += 1
        if not local.exists():
            report["no_local_counterpart"].append(str(rel))
            continue
        if _md5(path) == _md5(local):
            report["identical"] += 1
        else:
            report["different"].append(str(rel))
    return report


# --------------------------------------------------------------------------
# Integrity checks (task 2)
# --------------------------------------------------------------------------


def _mask_unique_values(path: Path) -> tuple[int, ...]:
    arr = np.array(Image.open(path))
    return tuple(int(v) for v in np.unique(arr))


def verify_integrity() -> dict[str, Any]:
    report: dict[str, Any] = {"train": {}, "val": {}, "test": {}, "errors": []}

    for split, csv_path in (("train", TRAIN_CSV), ("val", VAL_CSV)):
        rows = read_fmd_csv(csv_path)
        missing_ann: list[str] = []
        decode_errors: list[str] = []
        unique_by_source: dict[str, set[tuple[int, ...]]] = defaultdict(set)
        water_frac_by_source: dict[str, list[float]] = defaultdict(list)
        size_mismatches: list[dict[str, Any]] = []
        for row in rows:
            ann = resolve_train_val_annotation(split, row)
            if not ann.exists():
                missing_ann.append(str(ann))
                continue
            try:
                im = Image.open(ann)
                arr = np.array(im)
            except Exception as exc:  # noqa: BLE001 - want to record any decode failure
                decode_errors.append(f"{ann}: {exc}")
                continue
            uv = tuple(int(v) for v in np.unique(arr))
            unique_by_source[row["Source"]].add(uv)
            water_frac_by_source[row["Source"]].append(float((arr > 0).mean()))

            public_image = resolve_train_val_public_image(row)
            if public_image is not None:
                with Image.open(public_image) as pim:
                    if pim.size != im.size:
                        size_mismatches.append(
                            {
                                "source": row["Source"],
                                "image_size": list(pim.size),
                                "mask_size": list(im.size),
                                "mask": str(ann.name),
                            }
                        )
        report[split] = {
            "n_rows": len(rows),
            "n_missing_annotation": len(missing_ann),
            "missing_annotation_sample": missing_ann[:10],
            "n_decode_errors": len(decode_errors),
            "decode_error_sample": decode_errors[:10],
            "unique_mask_values_by_source": {k: sorted(v) for k, v in unique_by_source.items()},
            "water_fraction_stats_by_source": {
                k: {
                    "mean": float(np.mean(v)),
                    "median": float(np.median(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "n": len(v),
                }
                for k, v in water_frac_by_source.items()
            },
            "n_image_mask_size_mismatches": len(size_mismatches),
            "size_mismatch_sample": size_mismatches[:10],
        }

    # test split
    rows = read_fmd_csv(TEST_CSV)
    missing_rgb: list[str] = []
    missing_ann: list[str] = []
    decode_errors = []
    unique_by_source = defaultdict(set)
    frame_ids: dict[str, list[int]] = defaultdict(list)
    resolutions: dict[str, set[tuple[int, int]]] = defaultdict(set)
    water_whole: dict[str, list[float]] = defaultdict(list)
    water_bottom60: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        rgb = resolve_test_rgb(row)
        ann = resolve_test_annotation(row)
        if not rgb.exists():
            missing_rgb.append(str(rgb))
        else:
            try:
                with Image.open(rgb) as im:
                    im.verify()
            except Exception as exc:  # noqa: BLE001
                decode_errors.append(f"{rgb}: {exc}")
        if not ann.exists():
            missing_ann.append(str(ann))
            continue
        try:
            arr = np.array(Image.open(ann))
        except Exception as exc:  # noqa: BLE001
            decode_errors.append(f"{ann}: {exc}")
            continue
        src = row["Source"]
        uv = tuple(int(v) for v in np.unique(arr))
        unique_by_source[src].add(uv)
        resolutions[src].add((arr.shape[1], arr.shape[0]))
        water_whole[src].append(float((arr > 0).mean()))
        h = arr.shape[0]
        bottom = arr[int(h * 0.4) :, :]
        water_bottom60[src].append(float((bottom > 0).mean()))
        m = re.match(r"frame(\d+)\.jpg", row["Image path"])
        if m:
            frame_ids[src].append(int(m.group(1)))

    def _stats(values: list[float]) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": len(values),
        }

    frame_range = {}
    for src, ids in frame_ids.items():
        ids_sorted = sorted(ids)
        frame_range[src] = {
            "n_frames": len(ids_sorted),
            "min_frame_id": ids_sorted[0],
            "max_frame_id": ids_sorted[-1],
            "contiguous": bool(np.all(np.diff(ids_sorted) == 1)),
        }

    report["test"] = {
        "n_rows": len(rows),
        "n_missing_rgb": len(missing_rgb),
        "missing_rgb_sample": missing_rgb[:10],
        "n_missing_annotation": len(missing_ann),
        "n_decode_errors": len(decode_errors),
        "decode_error_sample": decode_errors[:10],
        "unique_mask_values_by_source": {k: sorted(v) for k, v in unique_by_source.items()},
        "resolutions_by_source": {k: sorted(v) for k, v in resolutions.items()},
        "water_fraction_whole_frame_by_source": {k: _stats(v) for k, v in water_whole.items()},
        "water_fraction_bottom60_by_source": {k: _stats(v) for k, v in water_bottom60.items()},
        "frame_id_range_by_source": frame_range,
    }
    return report


# --------------------------------------------------------------------------
# Index building (tasks 3-5): resolve, dedup, IoU, phash overlap
# --------------------------------------------------------------------------


@dataclass
class IndexRow:
    fmd_split: str
    source: str
    fmd_image_path: str
    local_image_path: str
    fmd_mask_path: str
    public_mask_path: str
    match_method: str
    phash: str
    group_id: str
    restricted: bool = True


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two boolean masks, resizing b to a's shape if needed."""
    if mask_a.shape != mask_b.shape:
        img_b = Image.fromarray((mask_b.astype(np.uint8)) * 255)
        img_b = img_b.resize((mask_a.shape[1], mask_a.shape[0]), Image.NEAREST)
        mask_b = np.array(img_b) > 127
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def _fas_public_mask_bool(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def _water_public_mask_bool(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr > 127


def build_public_image_corpus() -> list[Path]:
    """Every still image under our public raw downloads, for the phash
    overlap check against FMD's Greek/Italian test frames. FRED is included
    (front camera only); tinycamml has no labeled roadway images (per
    Subagent 1's brief) so it's skipped.
    """
    paths: list[Path] = []
    if FAS_IMAGE_DIR.exists():
        paths.extend(sorted(FAS_IMAGE_DIR.glob("*.jpg")))
    images_dir = ROADWAY_ROOT / "Dataset" / "images"
    if images_dir.exists():
        paths.extend(sorted(images_dir.glob("*.jpg")))
    for jpeg_dir in (WATER_V1_JPEG, WATER_V2_JPEG):
        if jpeg_dir.exists():
            paths.extend(sorted(jpeg_dir.rglob("*.jpg")))
    for sub in ("dry", "flooded"):
        front = FRED_ROOT / sub / "KITTI-style"
        if front.exists():
            paths.extend(sorted(front.rglob("front-imgs/*.png")))
    return paths


def _phash_bytes(h: imagehash.ImageHash) -> int:
    # imagehash stores a boolean array; pack to an int for fast Hamming ops.
    return int("".join("1" if b else "0" for b in h.hash.flatten()), 2)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _band_keys(value: int, n_bands: int = 8, band_bits: int = 8) -> list[tuple[int, int]]:
    return [(i, (value >> (i * band_bits)) & 0xFF) for i in range(n_bands)]


def compute_phash_index(
    paths: list[Path], log_every: int = 2000
) -> dict[str, int]:
    """phash (as packed int) for every path that decodes; skips failures."""
    out: dict[str, int] = {}
    for i, p in enumerate(paths):
        if log_every and i % log_every == 0:
            logger.info("phash progress: %d/%d", i, len(paths))
        try:
            with Image.open(p) as im:
                out[str(p)] = _phash_bytes(imagehash.phash(im))
        except Exception as exc:  # noqa: BLE001
            logger.warning("phash failed for %s: %s", p, exc)
    return out


def bucketed_overlap_search(
    query_hashes: dict[str, int],
    corpus_hashes: dict[str, int],
    threshold: int = PHASH_HAMMING_THRESHOLD,
) -> dict[str, list[tuple[str, int]]]:
    """For each query hash, find corpus hashes within Hamming `threshold`,
    without O(n*m) pairwise comparison.

    Uses 8-band LSH (8 bytes x 8 bits over a 64-bit phash): by pigeonhole,
    any pair with Hamming distance <= 6 must match exactly in at least
    8 - 6 = 2 of the 8 bands, so indexing by exact-byte-per-band and only
    verifying true Hamming distance among band-sharing candidates is
    lossless for threshold <= 7 while checking a small candidate set.
    """
    band_index: dict[tuple[int, int], list[str]] = defaultdict(list)
    for path, h in corpus_hashes.items():
        for band in _band_keys(h):
            band_index[band].append(path)

    results: dict[str, list[tuple[str, int]]] = {}
    for qpath, qh in query_hashes.items():
        candidates: set[str] = set()
        for band in _band_keys(qh):
            candidates.update(band_index.get(band, []))
        hits = []
        for cpath in candidates:
            d = _hamming(qh, corpus_hashes[cpath])
            if d <= threshold:
                hits.append((cpath, d))
        if hits:
            results[qpath] = sorted(hits, key=lambda x: x[1])
    return results


def build_index(compute_test_overlap: bool = True) -> tuple[list[IndexRow], dict[str, Any]]:
    rows: list[IndexRow] = []
    stats: dict[str, Any] = {"resolved": defaultdict(lambda: defaultdict(int)), "iou": {}, "test_overlap": {}}

    iou_values: dict[str, list[float]] = defaultdict(list)

    for split, csv_path in (("train", TRAIN_CSV), ("val", VAL_CSV)):
        for row in read_fmd_csv(csv_path):
            source = row["Source"]
            ann = resolve_train_val_annotation(split, row)
            public_image = resolve_train_val_public_image(row)
            public_mask = resolve_train_val_public_mask(row, public_image)

            if public_image is None:
                match_method = "unresolved"
                stats["resolved"][split][f"{source}:unresolved"] += 1
                local_image_path = ""
                group_id = row["Image path"]
                phash_hex = ""
            else:
                match_method = "filename_only"
                stats["resolved"][split][f"{source}:resolved"] += 1
                local_image_path = str(public_image)
                group_id = str(public_image)
                try:
                    with Image.open(public_image) as im:
                        phash_hex = str(imagehash.phash(im))
                except Exception:  # noqa: BLE001
                    phash_hex = ""

                # IoU against the source dataset's own public mask, where we
                # have one and the FMD mask decodes (source != Roadway,
                # which reuses the same file and is IoU=1.0 by construction).
                if public_mask is not None and ann.exists() and source != "Roadway Flooding Image Dataset":
                    try:
                        fmd_bool = np.array(Image.open(ann)) > 0
                        if source == "Flood Area Segmentation":
                            pub_bool = _fas_public_mask_bool(public_mask)
                        else:
                            pub_bool = _water_public_mask_bool(public_mask)
                        iou_values[source].append(_iou(fmd_bool, pub_bool))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("IoU failed for %s: %s", ann, exc)

            rows.append(
                IndexRow(
                    fmd_split=split,
                    source=source,
                    fmd_image_path=row["Image path"],
                    local_image_path=local_image_path,
                    fmd_mask_path=str(ann) if ann.exists() else "",
                    public_mask_path=str(public_mask) if public_mask else "",
                    match_method=match_method,
                    phash=phash_hex,
                    group_id=group_id,
                    restricted=True,
                )
            )

    # test rows
    test_rows = read_fmd_csv(TEST_CSV)
    test_phashes: dict[str, int] = {}
    for row in test_rows:
        rgb = resolve_test_rgb(row)
        ann = resolve_test_annotation(row)
        group_id = test_group_id(row)
        phash_hex = ""
        if rgb.exists():
            try:
                with Image.open(rgb) as im:
                    h = imagehash.phash(im)
                phash_hex = str(h)
                test_phashes[str(rgb)] = _phash_bytes(h)
            except Exception as exc:  # noqa: BLE001
                logger.warning("phash failed for %s: %s", rgb, exc)
        rows.append(
            IndexRow(
                fmd_split="test",
                source=row["Source"],
                fmd_image_path=row["Image path"],
                local_image_path=str(rgb) if rgb.exists() else "",
                fmd_mask_path=str(ann) if ann.exists() else "",
                public_mask_path="",
                match_method="filename_only" if rgb.exists() else "unresolved",
                phash=phash_hex,
                group_id=group_id,
                restricted=True,
            )
        )

    for source, values in iou_values.items():
        arr = np.array(values)
        stats["iou"][source] = {
            "n": len(arr),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "n_below_0.8": int((arr < 0.8).sum()),
        }

    if compute_test_overlap:
        logger.info("Building public image phash corpus for overlap check...")
        corpus_paths = build_public_image_corpus()
        logger.info("Corpus size: %d public images", len(corpus_paths))
        corpus_hashes = compute_phash_index(corpus_paths)
        overlap = bucketed_overlap_search(test_phashes, corpus_hashes)
        stats["test_overlap"] = {
            "n_test_frames_checked": len(test_phashes),
            "n_public_images_checked": len(corpus_hashes),
            "n_overlaps_found": len(overlap),
            "sample": {k: v[:3] for k, v in list(overlap.items())[:20]},
        }

    return rows, stats


def write_index_csv(rows: list[IndexRow], path: Path = FMD_ROOT / "index.csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fmd_split",
        "source",
        "fmd_image_path",
        "local_image_path",
        "fmd_mask_path",
        "public_mask_path",
        "match_method",
        "phash",
        "group_id",
        "restricted",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "fmd_split": r.fmd_split,
                    "source": r.source,
                    "fmd_image_path": r.fmd_image_path,
                    "local_image_path": r.local_image_path,
                    "fmd_mask_path": r.fmd_mask_path,
                    "public_mask_path": r.public_mask_path,
                    "match_method": r.match_method,
                    "phash": r.phash,
                    "group_id": r.group_id,
                    "restricted": r.restricted,
                }
            )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _cmd_list_drive(_args: argparse.Namespace) -> None:
    log_path = _setup_file_logger("list_drive")
    files = list_drive_folder()
    out_path = FMD_ROOT.parent / "flood_master_drive_listing.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(files, indent=0))
    print(f"Drive folder has {len(files)} files. Listing written to {out_path}. Log: {log_path}")


def _cmd_download_drive(args: argparse.Namespace) -> None:
    log_path = _setup_file_logger("download_drive")
    print(f"Downloading Drive folder to {FMD_DRIVE_ROOT} (log: {log_path}) ...")
    download_drive_folder(max_bytes=args.max_bytes)
    print("Download attempt finished (check the log for any per-file failures).")


def _cmd_diff_drive(_args: argparse.Namespace) -> None:
    log_path = _setup_file_logger("diff_drive")
    report = diff_drive_vs_local()
    out_path = FMD_ROOT.parent / "flood_master_drive_diff.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(
        f"Diff: checked={report['checked']} identical={report['identical']} "
        f"different={len(report['different'])} no_local_counterpart={len(report['no_local_counterpart'])}"
    )
    print(f"Full report: {out_path}. Log: {log_path}")


def _cmd_verify(_args: argparse.Namespace) -> None:
    log_path = _setup_file_logger("verify")
    report = verify_integrity()
    out_path = FMD_ROOT / "verify_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(
        f"train rows={report['train']['n_rows']} missing_ann={report['train']['n_missing_annotation']} | "
        f"val rows={report['val']['n_rows']} missing_ann={report['val']['n_missing_annotation']} | "
        f"test rows={report['test']['n_rows']} missing_rgb={report['test']['n_missing_rgb']}"
    )
    print(f"Full report: {out_path}. Log: {log_path}")


def _cmd_build_index(args: argparse.Namespace) -> None:
    log_path = _setup_file_logger("build_index")
    t0 = time.time()
    rows, stats = build_index(compute_test_overlap=not args.no_overlap)
    write_index_csv(rows)
    out_path = FMD_ROOT / "index_stats.json"
    out_path.write_text(json.dumps(stats, indent=2, default=int))
    print(f"Wrote {len(rows)} rows to {FMD_ROOT / 'index.csv'} in {time.time() - t0:.1f}s")
    print(f"Stats: {out_path}. Log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-drive", help="List the Drive folder's contents (no download).")

    p_dl = sub.add_parser("download-drive", help="Download the whole Drive folder.")
    p_dl.add_argument("--max-bytes", type=int, default=MAX_DRIVE_DOWNLOAD_BYTES)

    sub.add_parser("diff-drive", help="md5-diff downloaded Drive files against the local copy.")
    sub.add_parser("verify", help="Integrity checks: CSV rows, mask decode, mask semantics.")

    p_idx = sub.add_parser("build-index", help="Resolve/dedup/IoU/overlap; writes index.csv.")
    p_idx.add_argument(
        "--no-overlap", action="store_true", help="Skip the public-corpus phash overlap check (faster)."
    )

    args = parser.parse_args()
    {
        "list-drive": _cmd_list_drive,
        "download-drive": _cmd_download_drive,
        "diff-drive": _cmd_diff_drive,
        "verify": _cmd_verify,
        "build-index": _cmd_build_index,
    }[args.command](args)


if __name__ == "__main__":
    main()
