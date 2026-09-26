# builds data/processed/manifest.csv from every source
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

from prep.common import (
    CAMERA_SPLITS_PATH,
    GA511_ROOT,
    IMAGES_ROOT,
    LOGS_JOBS,
    MANIFEST_COLUMNS,
    MANIFEST_PATH,
    PROCESSED_ROOT,
    SPLITS_REPORT_PATH,
    VERSION_PATH,
    data_version_string,
    save_processed_image,
    setup_job_logger,
    stable_hash_fraction,
)
from prep.dedup import cluster_phashes, thin_sequence
from prep.mask_rules import (
    assign_label_for_source,
    nysdot_crop,
    water_frac_bottom_band,
    water_frac_fred,
    water_frac_grayscale_mask,
)
from prep.reports import append_pipeline_report_sections, write_class_counts_report
from prep.sources import (
    INCLUDE_ALLEYFLOODNET,
    iter_alleyfloodnet,
    iter_eu_flood_2013,
    iter_flood_area_segmentation,  # noqa: F401
    iter_fmd_test,
    iter_fred,
    iter_ga511,
    iter_iowa_rwis,
    iter_nysdot,
    iter_roadway_flooding,
    load_labels_csv,
)
from prep.splits import assert_disjoint, ga511_camera_splits, greedy_group_stratified_split

log = setup_job_logger("build_manifest")

LEGACY_SOURCES = ("roadway_flooding", "fred", "nysdot_road_surface", "flood_master_test")
# separate camera networks never share photos, skip cross-source dedup between them
CAMERA_NETWORK_SOURCES = frozenset({"ga511", "iowa_rwis", "nysdot_road_surface"})
MANIFEST_BACKUP_PATH = PROCESSED_ROOT / "manifest_v1-703f0040.csv"
MANIFEST_BACKUP_PATH_V4 = PROCESSED_ROOT / "manifest_v1-7251bbd2.csv"
IOWA_RWIS_SPLIT_SALT = "iowa-rwis-mixed-split-v1"

FRED_FORCED_LOCATION_SPLITS = {
    "cambogan": "test",
    "pullenvale": "val",
    "dairycreek": "train",
    "holmview": "train",
    "mountcotton": "train",
}

NYSDOT_FORCED_CAMERA_SPLITS = {
    "I_495_at_Veterans_Memorial_Hwy(Exit_57)__Westbound__Skyline_1877": "test",
    "I_495_at_Terry_Road_(Exits_59_58)__Westbound__Skyline_1878": "val",
}


CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"


def gather_rows(smoke: int | None = None, include_alleyfloodnet: bool = INCLUDE_ALLEYFLOODNET) -> list[dict]:
    t0 = time.time()
    rows: list[dict] = []
    builders = [
        ("roadway_flooding", iter_roadway_flooding),
        ("fred", iter_fred),
        ("nysdot_road_surface", iter_nysdot),
        ("flood_master_test", iter_fmd_test),
        ("iowa_rwis", iter_iowa_rwis),
        ("eu_flood_2013", iter_eu_flood_2013),
        ("alleyfloodnet", lambda: iter_alleyfloodnet(include=include_alleyfloodnet)),
    ]
    for name, fn in builders:
        r = fn()
        if smoke:
            r = r[:smoke]
        log.info("gathered %d raw rows from %s", len(r), name)
        rows.extend(r)

    user_labels = load_labels_csv(GA511_ROOT / "labels.csv")
    ai_review_labels = load_labels_csv(GA511_ROOT / "ai_review_labels.csv")
    existing_camera_splits: dict[str, str] = {}
    if CAMERA_SPLITS_PATH.exists():
        existing_camera_splits = json.loads(CAMERA_SPLITS_PATH.read_text())
    ga_rows = iter_ga511(GA511_ROOT / "frames.csv", user_labels, ai_review_labels, existing_camera_splits)
    if smoke:
        ga_rows = ga_rows[:smoke]
    log.info("gathered %d raw rows from ga511", len(ga_rows))
    rows.extend(ga_rows)

    log.info("gather_rows: %d total rows in %.1fs", len(rows), time.time() - t0)
    return rows


def compute_labels(rows: list[dict]) -> None:
    t0 = time.time()
    n_by_outcome: Counter = Counter()
    for r in rows:
        if r.get("label"):
            r["water_frac_road"] = float("nan")
            continue
        mask_kind = r.get("mask_kind")
        mask_path = r.get("mask_path")
        if not mask_kind or not mask_path:
            r["_drop"] = True
            n_by_outcome["no_mask_no_label"] += 1
            continue
        try:
            if mask_kind == "fred":
                wf = water_frac_fred(mask_path)
            elif mask_kind == "bottom_band01":
                wf = water_frac_bottom_band(mask_path)
            elif mask_kind == "bottom_band_grayscale":
                wf = water_frac_grayscale_mask(mask_path)
            else:
                raise ValueError(f"unknown mask_kind {mask_kind}")
        except Exception as e:  # noqa: BLE001
            log.warning("mask decode failed for %s (%s): %s", r["orig_path"], mask_path, e)
            r["_drop"] = True
            n_by_outcome["mask_decode_error"] += 1
            continue

        r["water_frac_road"] = wf
        label = assign_label_for_source(wf, r["source"])
        if label is None:
            r["_drop"] = True
            n_by_outcome["below_threshold_or_unverified_bucket_excluded"] += 1
        else:
            r["label"] = label
            n_by_outcome[f"labeled_{label}"] += 1

    log.info("compute_labels outcomes: %s (%.1fs)", dict(n_by_outcome), time.time() - t0)


CLIP_FILTERED_SOURCES = frozenset({"roadway_flooding", "nysdot_road_surface", "flood_master_test"})


def apply_clip_filter(rows: list[dict], skip: bool = False) -> dict:
    stats: dict[str, dict] = defaultdict(lambda: {"kept": 0, "removed": 0, "removed_paths": []})
    passthrough = [
        r for r in rows
        if r["source"] not in CLIP_FILTERED_SOURCES and r["source"] != "ga511" and not r.get("_drop")
    ]
    for r in passthrough:
        stats[r["source"]]["kept"] += 1

    candidates = [r for r in rows if r["source"] in CLIP_FILTERED_SOURCES and not r.get("_drop")]
    if skip or not candidates:
        for r in candidates:
            stats[r["source"]]["kept"] += 1
        return dict(stats)

    from prep.clip_filter import load_clip, score_batch

    t0 = time.time()
    bundle = load_clip(CLIP_MODEL, CLIP_PRETRAINED)
    log.info("CLIP model loaded in %.1fs", time.time() - t0)

    batch_size = 256
    for i in range(0, len(candidates), batch_size):
        chunk = candidates[i:i + batch_size]
        paths = [r["orig_path"] for r in chunk]
        results = score_batch(paths, bundle=bundle)
        for r, (is_road, pos, neg) in zip(chunk, results):
            s = stats[r["source"]]
            if is_road:
                s["kept"] += 1
            else:
                s["removed"] += 1
                if len(s["removed_paths"]) < 40:
                    s["removed_paths"].append(r["orig_path"])
                r["_drop"] = True
                r["_clip_reason"] = f"pos={pos:.3f} neg={neg:.3f}"
        if (i // batch_size) % 5 == 0:
            log.info("CLIP scored %d/%d", i + len(chunk), len(candidates))

    log.info("CLIP filter done in %.1fs: %s", time.time() - t0,
              {k: {"kept": v["kept"], "removed": v["removed"]} for k, v in stats.items()})
    return {k: v for k, v in stats.items()}


def resize_copy_and_hash(rows: list[dict]) -> None:
    t0 = time.time()
    counters_by_source: Counter = Counter()
    for r in rows:
        if r.get("_drop"):
            continue
        src = r["source"]
        try:
            img = Image.open(r["orig_path"]).convert("RGB")
            if r.get("needs_nysdot_crop"):
                img = nysdot_crop(img)
            counters_by_source[src] += 1
            dest_name = f"{src}_{counters_by_source[src]:06d}.jpg"
            dest = IMAGES_ROOT / src / dest_name
            w, h = save_processed_image(img, dest)
            r["path"] = str(dest)
            r["width"] = w
            r["height"] = h
            with Image.open(dest) as saved:
                r["phash"] = str(imagehash.phash(saved))
        except Exception as e:  # noqa: BLE001
            log.warning("resize/hash failed for %s: %s", r["orig_path"], e)
            r["_drop"] = True
    log.info("resize_copy_and_hash: processed %d rows in %.1fs", sum(counters_by_source.values()), time.time() - t0)


def thin_and_dedup(rows: list[dict]) -> dict:
    alive = [r for r in rows if not r.get("_drop")]

    thinned_out = 0
    thinning_by_group: dict[str, dict[str, int]] = {}
    by_seq: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(alive):
        if r["source"] in ("fred", "flood_master_test"):
            by_seq[(r["source"], r["group_id"])].append(i)

    keep_mask = [True] * len(alive)
    for (source, group_id), idxs in by_seq.items():
        idxs_sorted = sorted(idxs, key=lambda i: alive[i]["orig_path"])
        items = [(i, alive[i].get("phash")) for i in idxs_sorted]
        kept = set(thin_sequence(items))
        for i in idxs_sorted:
            if i not in kept:
                keep_mask[i] = False
                thinned_out += 1
        thinning_by_group[f"{source}:{group_id}"] = {
            "before": len(idxs_sorted),
            "after": len(kept),
        }

    alive = [r for r, keep in zip(alive, keep_mask) if keep]
    log.info("temporal thinning removed %d frames from FRED/FMD-test sequences", thinned_out)
    log.info("thinning by group: %s", thinning_by_group)

    phashes = [r.get("phash") for r in alive]
    cluster_ids = cluster_phashes(phashes)
    for r, cid in zip(alive, cluster_ids):
        r["dup_cluster"] = f"dc{cid:07d}"

    clusters: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(alive):
        clusters[r["dup_cluster"]].append(i)

    cross_source_collapsed = 0
    keep_final = [True] * len(alive)
    for idxs in clusters.values():
        sources_in_cluster = {alive[i]["source"] for i in idxs}
        if len(sources_in_cluster) <= 1:
            continue
        if sources_in_cluster <= CAMERA_NETWORK_SOURCES:
            continue
        def score(i):
            r = alive[i]
            return (
                0 if r.get("mask_path") else 1,
                0 if r.get("forced_split") else 1,
                0 if r["source"] in LEGACY_SOURCES or r["source"] == "ga511" else 1,
                r["source"],
                r["orig_path"],
            )
        idxs_sorted = sorted(idxs, key=score)
        winner = idxs_sorted[0]
        for i in idxs_sorted[1:]:
            keep_final[i] = False
            cross_source_collapsed += 1
        log.info("cross-source dup cluster %s collapsed %s -> kept %s",
                  alive[winner]["dup_cluster"], sources_in_cluster, alive[winner]["source"])

    alive = [r for r, keep in zip(alive, keep_final) if keep]
    log.info("cross-source dedup collapsed %d duplicate rows", cross_source_collapsed)

    for r in alive:
        if r["source"] in ("roadway_flooding", "flood_area_segmentation", "alleyfloodnet") and not r.get("group_id"):
            r["group_id"] = r["dup_cluster"]

    return {
        "thinned_frames": thinned_out,
        "thinning_rule": "keep a frame if phash Hamming > 6 from the last *kept* frame, "
                          "or if 15 consecutive frames have been dropped (resample at least "
                          "every 16th frame even during a long static stretch)",
        "thinning_by_group": thinning_by_group,
        "cross_source_duplicates_collapsed": cross_source_collapsed,
        "n_after": len(alive),
    }, alive


def iowa_rwis_forced_splits(ext_rows: list[dict], n_test: int = 6, n_val: int = 6) -> dict[str, str]:
    labels_by_cam: dict[str, set] = defaultdict(set)
    for r in ext_rows:
        if r["source"] == "iowa_rwis" and r.get("label"):
            labels_by_cam[r["camera_id"]].add(r["label"])
    mixed = sorted(cam for cam, labs in labels_by_cam.items() if {"wet", "dry"} <= labs)
    ranked = sorted(mixed, key=lambda c: stable_hash_fraction(c, IOWA_RWIS_SPLIT_SALT))
    forced: dict[str, str] = {}
    for cam in ranked[:n_test]:
        forced[cam] = "test"
    for cam in ranked[n_test:n_test + n_val]:
        forced[cam] = "val"
    return forced


def legacy_forced_splits(ext_rows: list[dict], old_manifest_path: Path | None = None) -> dict[str, str]:
    if old_manifest_path is None:
        old_manifest_path = MANIFEST_BACKUP_PATH if MANIFEST_BACKUP_PATH.exists() else MANIFEST_PATH
    if not old_manifest_path.exists():
        return {}
    old = pd.read_csv(old_manifest_path)
    old = old[old["source"].isin(LEGACY_SOURCES)]
    split_by_orig = dict(zip(old["orig_path"], old["split"]))

    priority = {"test": 0, "val": 1, "train": 2}
    votes: dict[str, Counter] = defaultdict(Counter)
    for r in ext_rows:
        if r["source"] not in LEGACY_SOURCES:
            continue
        old_split = split_by_orig.get(r["orig_path"])
        if old_split:
            votes[r["group_id"]][old_split] += 1

    forced: dict[str, str] = {}
    conflicts = []
    for gid, counter in votes.items():
        if len(counter) > 1:
            conflicts.append((gid, dict(counter)))
        forced[gid] = min(counter, key=lambda s: (priority[s], -counter[s]))
    if conflicts:
        log.warning("legacy_forced_splits: %d group(s) merged across old splits: %s", len(conflicts), conflicts)
    return forced


def assign_splits(rows: list[dict]) -> tuple[dict, list[dict]]:
    ga_rows = [r for r in rows if r["source"] == "ga511"]
    ext_rows = [r for r in rows if r["source"] != "ga511"]

    existing_splits: dict[str, str] = {}
    if CAMERA_SPLITS_PATH.exists():
        existing_splits = json.loads(CAMERA_SPLITS_PATH.read_text())

    reviewed_positive = {
        r["camera_id"] for r in ga_rows
        if r.get("label") in ("wet", "flooded") and r.get("label_source") in ("manual", "ai_review")
    }
    camera_ids = sorted({r["camera_id"] for r in ga_rows})
    cam_splits = ga511_camera_splits(camera_ids, existing_splits, reviewed_positive)
    CAMERA_SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAMERA_SPLITS_PATH.write_text(json.dumps(cam_splits, indent=2, sort_keys=True))
    for r in ga_rows:
        r["split"] = cam_splits[r["camera_id"]]

    n_weak_precip_test_dropped = sum(
        1 for r in ga_rows if r["split"] == "test" and r.get("label_source") == "weak_precip"
    )
    ga_rows = [
        r for r in ga_rows
        if not (r["split"] == "test" and r.get("label_source") == "weak_precip")
    ]
    if n_weak_precip_test_dropped:
        log.info(
            "dropped %d ga511 weak_precip row(s) that hashed into the test split "
            "(test uses manual/ai_review labels only)",
            n_weak_precip_test_dropped,
        )

    iowa_forced = iowa_rwis_forced_splits(ext_rows)
    # pin old rows to their old split so the test set stays comparable
    legacy_forced = legacy_forced_splits(ext_rows)
    forced: dict[str, str] = {}
    for r in ext_rows:
        if r["group_id"] in legacy_forced:
            forced[r["group_id"]] = legacy_forced[r["group_id"]]
        elif r.get("forced_split"):
            forced[r["group_id"]] = r["forced_split"]
        elif r["source"] == "fred" and r["group_id"] in FRED_FORCED_LOCATION_SPLITS:
            forced[r["group_id"]] = FRED_FORCED_LOCATION_SPLITS[r["group_id"]]
        elif r["source"] == "nysdot_road_surface" and r["group_id"] in NYSDOT_FORCED_CAMERA_SPLITS:
            forced[r["group_id"]] = NYSDOT_FORCED_CAMERA_SPLITS[r["group_id"]]
        elif r["source"] == "iowa_rwis" and r["group_id"] in iowa_forced:
            forced[r["group_id"]] = iowa_forced[r["group_id"]]

    group_labels: dict[str, Counter] = defaultdict(Counter)
    for r in ext_rows:
        if r.get("label"):
            group_labels[r["group_id"]][r["label"]] += 1

    group_splits = greedy_group_stratified_split(group_labels, forced=forced)
    for r in ext_rows:
        r["split"] = group_splits.get(r["group_id"], "train")

    all_rows = ga_rows + ext_rows
    violations = assert_disjoint(ga_rows, keys=("camera_id",))
    violations += assert_disjoint(ext_rows, keys=("group_id", "dup_cluster"))
    if violations:
        for v in violations:
            log.error("split disjointness violation: %s", v)
        raise AssertionError(f"{len(violations)} split-disjointness violations; see logs/jobs/build_manifest.log")

    write_splits_report(all_rows, cam_splits, group_splits, forced)

    stats = {
        "ga511_camera_splits": Counter(cam_splits.values()),
        "external_group_splits": Counter(group_splits.values()),
        "ga511_weak_precip_test_dropped": n_weak_precip_test_dropped,
    }
    return stats, all_rows


def write_splits_report(
    rows: list[dict],
    cam_splits: dict[str, str],
    group_splits: dict[str, str],
    forced_groups: dict[str, str],
) -> None:
    rows_per_split = Counter(r["split"] for r in rows)
    n_cameras_per_split = Counter(cam_splits.values())
    n_groups_per_split = Counter(group_splits.values())

    dup_clusters_per_split: dict[str, set] = defaultdict(set)
    for r in rows:
        dc = r.get("dup_cluster")
        if dc:
            dup_clusters_per_split[r["split"]].add(dc)

    report = {
        "assertion": "passed: no group_id, camera_id, or dup_cluster spans two splits",
        "n_rows_total": len(rows),
        "rows_per_split": dict(rows_per_split),
        "ga511_cameras_per_split": dict(n_cameras_per_split),
        "external_groups_per_split": dict(n_groups_per_split),
        "dup_clusters_per_split": {k: len(v) for k, v in dup_clusters_per_split.items()},
        "forced_groups": forced_groups,
    }
    SPLITS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))
    log.info("wrote %s", SPLITS_REPORT_PATH)


def assert_legacy_test_rows_unchanged(old_manifest_path: Path, new_rows: list[dict]) -> dict:
    if not old_manifest_path.exists():
        return {"skipped": "no prior manifest.csv to compare against"}
    old = pd.read_csv(old_manifest_path)
    old_legacy = old[old["source"].isin(LEGACY_SOURCES)]

    new_by_orig: dict[str, list[dict]] = defaultdict(list)
    for r in new_rows:
        new_by_orig[r.get("orig_path")].append(r)

    mismatches: list[str] = []
    renumbered: list[str] = []
    for _, old_r in old_legacy.iterrows():
        cands = new_by_orig.get(old_r["orig_path"])
        if not cands:
            mismatches.append(f"missing from rebuild: orig_path={old_r['orig_path']!r} (old path={old_r['path']!r})")
            continue
        new_r = cands[0]
        if new_r.get("label") != old_r["label"] or new_r.get("split") != old_r["split"]:
            mismatches.append(
                f"changed: orig_path={old_r['orig_path']!r} "
                f"old(label={old_r['label']!r}, split={old_r['split']!r}) -> "
                f"new(label={new_r.get('label')!r}, split={new_r.get('split')!r})"
            )
        elif new_r.get("path") != old_r["path"]:
            renumbered.append(f"orig_path={old_r['orig_path']!r}: path {old_r['path']!r} -> {new_r.get('path')!r}")

    result = {
        "n_old_legacy_rows_checked": len(old_legacy),
        "n_old_legacy_test_rows_checked": int((old_legacy["split"] == "test").sum()),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:50],
        "n_path_renumbered_same_label_split": len(renumbered),
        "path_renumbered_examples": renumbered[:10],
    }
    if mismatches:
        raise AssertionError(
            f"{len(mismatches)} legacy row mismatch(es) against {old_manifest_path}; "
            f"first few: {mismatches[:5]}"
        )
    log.info("assert_legacy_test_rows_unchanged: %s", result)
    return result


def assert_v4_rebuild_invariants(reference_manifest_path: Path, new_rows: list[dict]) -> dict:
    # every non-ga511 row + ga511 test rows must match the pre-v4 manifest
    if not reference_manifest_path.exists():
        return {"skipped": f"no {reference_manifest_path} to compare against"}
    old = pd.read_csv(reference_manifest_path)

    new_by_orig: dict[str, list[dict]] = defaultdict(list)
    for r in new_rows:
        new_by_orig[r.get("orig_path")].append(r)

    mismatches: list[str] = []
    missing: list[str] = []

    non_ga = old[old["source"] != "ga511"]
    for _, old_r in non_ga.iterrows():
        cands = new_by_orig.get(old_r["orig_path"])
        if not cands:
            missing.append(f"non-ga511 row missing from rebuild: orig_path={old_r['orig_path']!r}")
            continue
        new_r = cands[0]
        if new_r.get("label") != old_r["label"] or new_r.get("split") != old_r["split"]:
            mismatches.append(
                f"non-ga511 changed: orig_path={old_r['orig_path']!r} "
                f"old(label={old_r['label']!r}, split={old_r['split']!r}) -> "
                f"new(label={new_r.get('label')!r}, split={new_r.get('split')!r})"
            )

    ga_test_old = old[(old["source"] == "ga511") & (old["split"] == "test")]
    for _, old_r in ga_test_old.iterrows():
        cands = new_by_orig.get(old_r["orig_path"])
        if not cands:
            missing.append(f"ga511 test row missing from rebuild: orig_path={old_r['orig_path']!r}")
            continue
        new_r = cands[0]
        if new_r.get("label") != old_r["label"] or new_r.get("split") != "test":
            mismatches.append(
                f"ga511 test row changed: orig_path={old_r['orig_path']!r} "
                f"old(label={old_r['label']!r}) -> new(label={new_r.get('label')!r}, split={new_r.get('split')!r})"
            )

    n_non_ga_old, n_non_ga_new = len(non_ga), sum(1 for r in new_rows if r["source"] != "ga511")
    if n_non_ga_new != n_non_ga_old:
        mismatches.append(f"non-ga511 row count changed: old={n_non_ga_old} new={n_non_ga_new}")

    n_ga_test_old = len(ga_test_old)
    n_ga_test_new = sum(1 for r in new_rows if r["source"] == "ga511" and r.get("split") == "test")
    if n_ga_test_new != n_ga_test_old:
        mismatches.append(f"ga511 test row count changed (nothing new may enter test): "
                           f"old={n_ga_test_old} new={n_ga_test_new}")

    result = {
        "n_non_ga511_rows_checked": int(n_non_ga_old),
        "n_ga511_test_rows_checked": int(n_ga_test_old),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:50],
        "n_missing": len(missing),
        "missing": missing[:50],
    }
    log.info("assert_v4_rebuild_invariants: %s", json.dumps(result, indent=2, default=str))
    if mismatches or missing:
        raise AssertionError(
            f"{len(mismatches)} mismatch(es), {len(missing)} missing row(s) vs {reference_manifest_path}; "
            f"first few: {(mismatches + missing)[:5]}"
        )
    return result


def report_ga511_daytime(pre_dedup_rows: list[dict], final_rows: list[dict], date_str: str = "2026-09-26") -> dict:
    from datetime import date as _date

    from prep.common import ga511_is_daytime, ga511_local_time

    target_date = _date.fromisoformat(date_str)

    def _is_target(r: dict) -> bool:
        if r.get("source") != "ga511":
            return False
        dt = ga511_local_time(r.get("orig_path"))
        if dt is None or dt.date() != target_date:
            return False
        return bool(ga511_is_daytime(r.get("orig_path")))

    pre_labeled = {r["orig_path"] for r in pre_dedup_rows if _is_target(r) and r.get("label")}
    final_by_path = {r["orig_path"]: r for r in final_rows if _is_target(r)}
    final_paths = set(final_by_path.keys())

    by_split: Counter = Counter()
    by_split_label_source: dict[str, Counter] = defaultdict(Counter)
    for r in final_by_path.values():
        split = r.get("split") or "unknown"
        by_split[split] += 1
        by_split_label_source[split][r.get("label_source") or "none"] += 1

    return {
        "date": date_str, "window": "08:00-19:00 America/New_York",
        "n_labeled_pre_dedup": len(pre_labeled),
        "n_final": len(final_paths),
        "n_thinned_or_deduped": len(pre_labeled - final_paths),
        "by_split": dict(by_split),
        "by_split_label_source": {k: dict(v) for k, v in by_split_label_source.items()},
    }


def backup_old_manifest(old_manifest_path: Path = MANIFEST_PATH, backup_path: Path = MANIFEST_BACKUP_PATH) -> str:
    if not old_manifest_path.exists():
        return "skipped: no manifest.csv to back up"
    if backup_path.exists():
        return f"skipped: {backup_path} already exists"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old_manifest_path, backup_path)
    log.info("backed up %s -> %s", old_manifest_path, backup_path)
    return f"copied to {backup_path}"


def write_manifest(rows: list[dict]) -> None:
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["path"]):
            out = {c: r.get(c, "") for c in MANIFEST_COLUMNS}
            if isinstance(out.get("restricted"), bool):
                out["restricted"] = str(out["restricted"])
            wf = out.get("water_frac_road")
            if wf is None or (isinstance(wf, float) and math.isnan(wf)):
                out["water_frac_road"] = ""
            w.writerow(out)
    VERSION_PATH.write_text(data_version_string() + "\n")
    log.info("wrote manifest with %d rows to %s", len(rows), MANIFEST_PATH)


def run(smoke: int | None = None, skip_clip: bool = False, include_alleyfloodnet: bool = INCLUDE_ALLEYFLOODNET) -> dict:
    t0 = time.time()
    rows = gather_rows(smoke=smoke, include_alleyfloodnet=include_alleyfloodnet)
    compute_labels(rows)
    pre_dedup_snapshot = [dict(r) for r in rows]  # for the daytime thinned/deduped count, below
    clip_stats = apply_clip_filter(rows, skip=skip_clip)
    resize_copy_and_hash(rows)
    dedup_stats, alive = thin_and_dedup(rows)
    split_stats, alive = assign_splits(alive)

    reference_manifest = MANIFEST_BACKUP_PATH if MANIFEST_BACKUP_PATH.exists() else MANIFEST_PATH
    legacy_check = assert_legacy_test_rows_unchanged(reference_manifest, alive)
    # back up pre-v4 manifest, then check the rebuild against it
    v4_backup_note = backup_old_manifest(MANIFEST_PATH, MANIFEST_BACKUP_PATH_V4)
    v4_check = assert_v4_rebuild_invariants(MANIFEST_BACKUP_PATH_V4, alive)
    daytime_report = report_ga511_daytime(pre_dedup_snapshot, alive)
    backup_note = backup_old_manifest()
    write_manifest(alive)

    class_counts_md = Path("reports/class_counts.md")
    write_class_counts_report(alive, class_counts_md, Path("reports/figures/class_counts.png"))
    append_pipeline_report_sections(class_counts_md, clip_stats=clip_stats, dedup_stats=dedup_stats)
    log.info("wrote %s", class_counts_md)

    summary = {
        "n_raw": len(rows),
        "n_final": len(alive),
        "clip": clip_stats,
        "dedup": dedup_stats,
        "splits": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in split_stats.items()},
        "legacy_test_rows_check": legacy_check,
        "v4_rebuild_invariants_check": v4_check,
        "ga511_daytime_2026_09_26": daytime_report,
        "manifest_backup": backup_note,
        "manifest_backup_v4": v4_backup_note,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (LOGS_JOBS / "build_manifest_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("build_manifest.run summary: %s", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=int, default=None, help="limit each source to N raw rows for a fast test run")
    ap.add_argument("--skip-clip", action="store_true", help="skip the CLIP road-scene filter (faster smoke runs)")
    ap.add_argument("--skip-alleyfloodnet", action="store_true", help="drop the alleyfloodnet source entirely")
    ap.add_argument("--daemon", action="store_true", help="fork a detached background process and return immediately")
    args = ap.parse_args()

    if args.daemon:
        import subprocess
        LOGS_JOBS.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_JOBS / "build_manifest_daemon.log"
        cmd = [sys.executable, "-m", "prep.build_manifest"]
        if args.smoke:
            cmd += ["--smoke", str(args.smoke)]
        if args.skip_clip:
            cmd += ["--skip-clip"]
        if args.skip_alleyfloodnet:
            cmd += ["--skip-alleyfloodnet"]
        with open(log_path, "ab") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=logf,
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                start_new_session=True,
            )
        pid_path = LOGS_JOBS / "build_manifest.pid"
        pid_path.write_text(str(proc.pid))
        print(f"started detached build_manifest pid={proc.pid}, log={log_path}")
        return

    summary = run(smoke=args.smoke, skip_clip=args.skip_clip, include_alleyfloodnet=not args.skip_alleyfloodnet)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

