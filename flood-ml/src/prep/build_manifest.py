"""End-to-end, deterministic, re-runnable manifest build.

    cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m prep.build_manifest [--smoke N] [--skip-clip]

Pipeline:
  1. gather raw rows from every source (prep.sources)
  2. compute water_frac_road + label for mask-backed rows (prep.mask_rules)
  3. CLIP road-scene filter on external sources (prep.clip_filter)
  4. resize to data/processed/images/<source>/..., compute phash + width/height
  5. temporal thinning of FRED sequences + FMD test videos, then global
     cross-source phash dedup (prep.dedup)
  6. group-aware split assignment (prep.splits), with an assertion that no
     group_id/camera_id/dup_cluster spans two splits
  7. write data/processed/manifest.csv + data/processed/VERSION

Long, per-image progress goes to logs/jobs/build_manifest.log; the console
only prints stage summaries. Designed to be started detached
(`--daemon`) since a full run over ~8k images (mask decode + CLIP + resize)
can take several minutes.
"""
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
    iter_flood_area_segmentation,  # noqa: F401 - kept importable/documented; excluded below, see EXCLUDED note
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

# v3: sources gathered before this retrain pass. EVERY row of theirs (any
# split) must keep its split across a rebuild -- checked programmatically by
# assert_legacy_test_rows_unchanged (test rows) and enforced by
# legacy_forced_splits (all rows) before the manifest is overwritten. ga511
# is deliberately excluded: its frames.csv keeps growing and test-camera
# frames still need manual labels, as documented in the v3 brief.
LEGACY_SOURCES = ("roadway_flooding", "fred", "nysdot_road_surface", "flood_master_test")
MANIFEST_BACKUP_PATH = PROCESSED_ROOT / "manifest_v1-703f0040.csv"
IOWA_RWIS_SPLIT_SALT = "iowa-rwis-mixed-split-v1"

# Orchestrator ruling (val was unusable for threshold tuning with 0 FRED
# rows): of the 4 FRED locations with BOTH dry and flooded sequences
# (cambogan, dairycreek, holmview, pullenvale), force ~2 to train, 1 to val,
# 1 to test -- a whole location (all its sequences) per split, so val and
# test each get real FRED dry/flooded rows instead of relying on the greedy
# stratified split (which was starving val of FRED entirely). The one
# flooded-only location (mountcotton) goes to train, per the same ruling.
# Locations not listed here (e.g. a newly-arrived one) fall back to the
# greedy stratified split.
FRED_FORCED_LOCATION_SPLITS = {
    "cambogan": "test",
    "pullenvale": "val",
    "dairycreek": "train",
    "holmview": "train",
    "mountcotton": "train",
}

# Orchestrator ruling: NYSDOT (only 4 camera groups total, each with a good
# dry/wet mix) had 0 test rows under the greedy stratified split. Force one
# camera to test and one to val (each with both dry and wet rows) so both
# splits get real wet-road negatives for Stage A/B threshold tuning; the
# remaining 2 cameras go to train.
NYSDOT_FORCED_CAMERA_SPLITS = {
    "I_495_at_Veterans_Memorial_Hwy(Exit_57)__Westbound__Skyline_1877": "test",
    "I_495_at_Terry_Road_(Exits_59_58)__Westbound__Skyline_1878": "val",
}

# EXCLUDED (Phase 2 finding, overrides PLAN.md/DATASETS.md): Flood Area
# Segmentation and the Flood Master Database's ITALIAN test video turned out,
# on visual spot-check (12 random images for Flood Area Segmentation, 24
# spread across the whole Italian video, see docs/phase_reports/phase2_prep.md),
# to be elevated drone/news-b-roll footage looking down on flooded towns and
# fields -- not the ground-level DOT-camera view this classifier targets.
# CLIP's generic prompts only weakly discriminate this specific "drone
# hovering over a flooded town" case (mean pos-neg margin for Flood Area
# Segmentation was -0.017, max +0.057 across all 290 images; even the
# CLIP-preferred top-20 are unambiguously aerial by eye), so a per-image CLIP
# threshold would either keep obviously-aerial images or exclude ~everything.
# Excluded wholesale rather than row-by-row.
#
# The Flood Master GREEK test video was RE-VERIFIED this pass (the previous
# worker had lumped it in with the Italian video sight-unseen): 20 frames
# spread across its full range are a fixed, static-framing elevated camera
# over a flooded street with submerged cars, not a drone shot. It is a real
# flooded-road scene and is gathered below via `iter_fmd_test()` (which
# defaults to Greek only).

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
        # flood_area_segmentation is intentionally NOT gathered -- see the
        # EXCLUDED note above. iter_fmd_test() defaults to the Greek video
        # only (the Italian video is excluded the same way).
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
    ga_rows = iter_ga511(GA511_ROOT / "frames.csv", user_labels, ai_review_labels)
    if smoke:
        ga_rows = ga_rows[:smoke]
    log.info("gathered %d raw rows from ga511", len(ga_rows))
    rows.extend(ga_rows)

    log.info("gather_rows: %d total rows in %.1fs", len(rows), time.time() - t0)
    return rows


def compute_labels(rows: list[dict]) -> None:
    """Mutates rows in place: fills `water_frac_road` and `label` (where not
    already known, e.g. FRED dry / NYSDOT) from each row's mask. Rows that
    end up with no usable label are marked `_drop=True` rather than removed
    here, so counts stay easy to report.
    """
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
        except Exception as e:  # noqa: BLE001 - log and drop, never crash the build
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



# Sources that mix view types within a single Kaggle/photo-collection download
# and so benefit from the CLIP road-scene filter. FRED is deliberately NOT
# here: every FRED frame is a single vehicle-mounted forward dashcam feed by
# construction (there is no "wrong view type" contamination to remove), and
# CLIP filtering it was found to strip real flooded/dry road-path frames --
# on inspection, ~98% of FRED's 352 CLIP-removed frames were from ONE
# location (Mount-Cotton) where the road is a paved park driveway/parking
# area next to grass; CLIP's kept and removed samples from that same
# location look visually indistinguishable (pos/neg scores both ~0.24-0.31,
# right at the decision boundary) -- this is the dashcam's actual road, not
# an aerial/indoor/river shot, so filtering it serves no purpose and only
# throws away scarce flooded-road data. See docs/phase_reports/phase2_prep.md.
CLIP_FILTERED_SOURCES = frozenset({"roadway_flooding", "nysdot_road_surface", "flood_master_test"})


def apply_clip_filter(rows: list[dict], skip: bool = False) -> dict:
    """Filters external (non-ga511) rows through the CLIP road-scene check.
    Only applied to `CLIP_FILTERED_SOURCES` (mixed-view-type photo
    collections) -- FRED is a single coherent dashcam sequence and is passed
    through untouched (see the module-level note on `CLIP_FILTERED_SOURCES`).
    Returns a dict of per-source removed/kept counts and a sample of removed
    paths (for the report grids). Mutates `_drop` on rejected rows.
    """
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
    """For every surviving row: apply the NYSDOT crop if needed, resize to
    short-side 256 JPEG q95 under data/processed/images/<source>/, and record
    phash + width/height. Mutates `path`, `width`, `height`, `phash`.
    """
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
    """Temporal thinning within FRED/FMD-test sequences, then global
    cross-source phash clustering. Cross-source duplicate clusters collapse
    to a single surviving row; within-source clusters keep every row but
    share `dup_cluster` (and therefore must land in the same split).
    """
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
        # Cross-source duplicate: keep exactly one representative row.
        # Preference: has a mask_path > forced_split (test videos) >
        # legacy source (so a v3 addition can never bump an old source's
        # row, which would silently break the old-test-rows-unchanged
        # guarantee) > deterministic order.
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

    # Still-image sets (no natural grouping) use dup_cluster as their split group.
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
    """v3 brief: val AND test must each get several iowa_rwis cameras with
    both wet and dry frames. Deterministically hash-ranks the cameras that
    have both labels and pins the top `n_test` to test, the next `n_val` to
    val; everything else (including non-mixed iowa_rwis cameras) falls
    through to the normal greedy stratified split.
    """
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
    """v3 bug found and fixed: `greedy_group_stratified_split` recomputes
    every group's split FROM SCRATCH over the whole external pool, so simply
    adding thousands of new v3 groups (eu_flood_2013 uploaders, alleyfloodnet
    dup clusters) reshuffled OLD groups' assignments too, including
    roadway_flooding's, whose group_id (a dup_cluster id) is also just a
    renumbered id, not stable content identity -- a first real run of this
    rebuild flipped 66 old test rows before this existed.

    Looks up each legacy-source row's split by `orig_path` (the one truly
    stable identity) in the pre-rebuild manifest, then pins its NEW group_id
    to that split, so `greedy_group_stratified_split` only ever decides
    genuinely new groups. If a brand-new image happens to pHash-bridge two
    previously-separate old dup_clusters that had DIFFERENT old splits (only
    possible via a new image, since old clustering only grows monotonically,
    never splits), resolves test > val > train -- test-set identity is the
    hard requirement -- and logs it.
    """
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

    # --- 511GA: persisted, stable per-camera split ---
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

    # The 511GA test split uses only manual/ai_review labels (per the Phase 2
    # brief): a camera can still hash into "test" while its only usable label
    # so far is a weak_precip likely_dry frame, which must NOT leak into the
    # test set. Drop those rows entirely (they stay unlabeled/excluded, same
    # as any other frame with no usable label) rather than writing them.
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

    # --- external: forced groups, then greedy stratified split ---
    iowa_forced = iowa_rwis_forced_splits(ext_rows)
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
    # 511GA's authoritative grouping key is camera_id (its split is a stable
    # hash of camera_id, deliberately independent of anything else). Two
    # different cameras' frames can occasionally share a phash-based
    # dup_cluster by coincidence (generic-looking dark highway scenes at low
    # detail), which must NOT force a camera to move split -- that would
    # break the "a camera's split never changes" guarantee for a reason
    # (an unrelated camera's frame) that has nothing to do with the camera
    # itself. So dup_cluster disjointness is only enforced for external rows,
    # where dup_cluster is the sole grouping key for still-image sources.
    # (After cross-source dedup collapsing, no dup_cluster spans both a
    # ga511 and an external row -- collapsing already reduced any such
    # cluster to a single surviving row.)
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
    """Writes data/processed/splits_report.json: row counts per split, the
    disjointness assertion result (always "passed" here -- `assign_splits`
    raises before this is called if it isn't), and group/camera/dup_cluster
    counts, so the split assignment is auditable without re-running the
    build.
    """
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
    """v3 brief: existing sources and splits must not change. Checks EVERY
    row of a source that existed before this pass (LEGACY_SOURCES -- ga511 is
    exempt, see its constant docstring), not just test rows, since the brief
    says "every existing row keeps its split." Compares by `orig_path` (the
    true stable identity across a rebuild -- Phase 4 found that 511GA's
    numbered `path` values shift when frames.csv grows, see
    reports/EVALUATION.md), and raises if any old legacy row's label or split
    changed, or if it disappeared. A `path` renumbering with the same
    label/split is reported, not raised (expected for ga511 only; a legacy
    source renumbering would itself be a red flag, so it's included in the
    report for a human to check). Test rows get their own counted subset in
    the result, since that's the literal invariant the brief names.
    """
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


def backup_old_manifest(old_manifest_path: Path = MANIFEST_PATH, backup_path: Path = MANIFEST_BACKUP_PATH) -> str:
    """Copies the pre-rebuild manifest.csv to a version-tagged backup, once
    (never overwrites an existing backup -- a re-run in the same session
    must not clobber it with an already-rebuilt manifest).
    """
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
    clip_stats = apply_clip_filter(rows, skip=skip_clip)
    resize_copy_and_hash(rows)
    dedup_stats, alive = thin_and_dedup(rows)
    split_stats, alive = assign_splits(alive)

    # Prefer the durable pre-v3 backup (data/processed/manifest_v1-703f0040.csv,
    # sha256-verified by the orchestrator) as the reference: it stays a valid
    # comparison point even after a successful rebuild overwrites manifest.csv
    # itself, unlike comparing against the live file.
    reference_manifest = MANIFEST_BACKUP_PATH if MANIFEST_BACKUP_PATH.exists() else MANIFEST_PATH
    legacy_check = assert_legacy_test_rows_unchanged(reference_manifest, alive)
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
        "manifest_backup": backup_note,
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
