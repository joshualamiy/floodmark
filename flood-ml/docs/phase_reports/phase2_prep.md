# Phase 2: data prep

Resumed from a previous worker that was stopped mid-task. Its `src/prep/*.py`
(~2,400 lines) was kept and fixed rather than rewritten; the design (manifest
schema, road-region mask priors, CLIP filter, phash dedup, group-aware
splits, `camera_style` augmentation, labeling tool) was sound. This report
covers what was verified, what was wrong and fixed, the orchestrator's two
rounds of review feedback, and the final build.

## What the previous worker got right (verified, kept as-is)

- `src/prep/common.py`, `mask_rules.py` (road-region priors, water_frac
  computation), `dedup.py` (banded-bucket phash union-find + sequence
  thinning), `splits.py` (stable per-camera hash split, greedy group-
  stratified split, disjointness assertion), `augment.py` (`camera_style`),
  `label_tool.py` (stdlib `http.server` labeling page).
- The manifest schema, column order, and `VERSION` format match PLAN.md
  section 3 exactly.

## What was wrong and fixed

1. **Two Phase-1-fact overrides were half-right.** The previous worker
   excluded Flood Area Segmentation (290 images) *and* both Flood Master
   Database test videos (Greek 567 + Italian 1,406 frames) as "aerial drone
   footage," sight-unseen for the FMD test videos (comment says "despite
   FLOOD_MASTER.md's ground/street-level description"). I re-verified both
   by eye:
   - **Flood Area Segmentation**: 12 random images, 12/12 are elevated
     drone/aerial shots of flooded towns and fields. Exclusion confirmed
     correct -- see `reports/spotcheck_review_flood_area_sample` (viewed
     directly, not saved) and the CLIP-margin evidence already in the code
     comments (mean pos-neg margin -0.017, max +0.057 across all 290).
   - **FMD Greek video**: 20 frames spread across the full 567-frame range
     are a *fixed, static-framing* elevated camera over a flooded street with
     submerged cars -- not a drone (no panning/tilting, same framing every
     frame). This is a real flooded-road scene. **Reinstated** it (was wrongly
     excluded) via `iter_fmd_test(include_sources=("greek video",))`, now
     called from `gather_rows`. Thinned 567 -> 62 frames, all in the external
     test split (`fmd_greek_video` group).
   - **FMD Italian video**: 24 frames spread across the full range are
     genuine drone/news b-roll (on-screen NYTimes credit, panning/tilting
     aerial shots of vineyards and rooftops). Exclusion confirmed correct,
     stays excluded.
2. **Orchestrator ruling on the "wet band" / "dry-at-zero" rules**: added
   `mask_rules.assign_label_for_source()` and
   `DRY_AT_ZERO_VERIFIED_SOURCES` / `WET_BAND_VERIFIED_SOURCES` (currently
   `{"fred"}` for both). `roadway_flooding`'s wet bucket (n=8) and dry bucket
   (n=1) are excluded for that source -- population too small to verify, and
   the single "dry" image turned out on inspection to be a "ROAD CLOSED" sign
   on a flood-adjacent gravel road, not a dry scene, exactly the failure mode
   the ruling warns about. Full per-(source, label) table below.
3. **NYSDOT crop wasn't applied in the official spot-check sheets** --
   `prep.spotcheck` was showing the raw (header-visible) image for its
   contact sheets, defeating the purpose of checking the crop. Fixed
   `spotcheck.py` to apply `nysdot_crop` before rendering, and fixed
   `reports.contact_sheet`'s `_load_thumb` to accept a PIL Image directly
   (it only handled ndarray or a path before).
4. **Ruff was failing** (18 errors: unused imports, blind `except Exception`,
   `PERF102` dict iteration, `PLR0124` NaN self-comparisons, unsorted
   imports). All fixed; `ruff check .` is clean.
5. **`splits_report.json` was never written** despite `SPLITS_REPORT_PATH`
   existing as a constant and the brief requiring it. Added
   `write_splits_report()`, called from `assign_splits`.
6. **`reports/class_counts.md` was never wired into the build** and didn't
   include CLIP removal counts, dedup/thinning counts, or the spot-check
   table (all required by the brief). Added
   `write_class_counts_report` + `append_pipeline_report_sections` calls to
   `build_manifest.run()`, so every rebuild regenerates the full report
   automatically.
7. **No tests existed.** Added `tests/test_prep_augment.py`,
   `test_prep_splits.py`, `test_prep_dedup.py`, `test_prep_mask_rules.py`,
   `test_prep_label_tool.py` (39 new tests, 103 total in the suite, all
   passing, no data/network/models).

## Orchestrator review round 1 (after an intermediate build) -- fixed

1. **Val was unusable for threshold tuning** (10 dry / 6 wet / 153 flooded,
   0 FRED rows). Forced FRED's location-level split explicitly instead of
   leaving it to the greedy stratified split: of the 4 locations with both
   dry and flooded sequences, `dairycreek` and `holmview` -> train,
   `pullenvale` -> val, `cambogan` -> test (unchanged, already the held-out
   test location); `mountcotton` (flooded-only) -> train. See
   `FRED_FORCED_LOCATION_SPLITS` in `build_manifest.py`.
2. **Flood Area Segmentation was silently missing from the per-source
   table.** Added a "Sources excluded wholesale" section to
   `class_counts.md` (via `reports.EXCLUDED_SOURCES_TABLE`) documenting both
   Flood Area Segmentation (290) and the FMD Italian video (1,406) with
   their exclusion reasons, so nothing is silently absent.
3. **NYSDOT test had 0 rows.** Only 4 camera groups exist total,
   each with a good dry/wet mix; the greedy split wasn't putting any in
   test. Forced one camera to test and one to val (`NYSDOT_FORCED_CAMERA_SPLITS`),
   each with both dry and wet rows; the remaining 2 cameras stay in train.
4. **FRED's CLIP removals (352 frames) were investigated in depth.** 344 of
   352 were from a single sequence (`Mount-Cotton_20241217_113410`, a paved
   park driveway/parking area next to grass). A closer look at a fresh
   20-image sample of Mount-Cotton's CLIP-*removed* frames vs. its CLIP-
   *kept* frames found them visually indistinguishable (both show the same
   paved-path-next-to-grass style, pos/neg CLIP scores both ~0.24-0.31, right
   at the decision boundary). This is the dashcam's actual road, not an
   aerial/indoor/river shot CLIP is meant to catch, so filtering it only
   discarded scarce flooded-road frames for no benefit. **FRED is no longer
   CLIP-filtered at all** -- `CLIP_FILTERED_SOURCES` now excludes it, only
   `roadway_flooding`, `nysdot_road_surface`, `flood_master_test` (the
   mixed-view-type photo collections) go through CLIP.
5. **flood_master_test thinning rule and per-video counts**: `thin_sequence`
   keeps a frame if its phash Hamming distance from the last *kept* frame
   is > 6, or if 15 consecutive frames have already been dropped (forced
   resample at least every 16th frame). Greek video: 567 -> 62 frames.
   FRED per-location: cambogan 1110->622, dairycreek 942->344, holmview
   850->214, mountcotton 1379->572, pullenvale 563->210 (now in
   `reports/class_counts.md`'s "Dedup / temporal thinning" table).
6. **511GA weak labels**: the 511GA sweep finished with ~1,600 `likely_dry`
   rows. Rebuilt against the current `data/ga511/frames.csv`
   (1,262 alive cameras) and ai-reviewed all newly-test-hashed cameras
   (below). **Found and fixed a real bug in the process**: nothing stopped a
   camera whose only usable label was `weak_precip` (`likely_dry`) from
   hashing into the *test* split, which the brief explicitly says must use
   only `manual`/`ai_review` labels. Added a filter in `assign_splits()` that
   drops (excludes, doesn't mislabel) any 511GA row landing in test with
   `label_source == "weak_precip"`. 109 such rows were dropped on this build.
   Also found and fixed a **second bug**: with ~1,262 cameras now in play, a
   few 511GA `dup_cluster`s spanned two different cameras' frames purely by
   phash coincidence (visually generic dark highway scenes), which
   `assert_disjoint` correctly flagged as violations. Camera_id is 511GA's
   sole authoritative split key by design (a stable hash, deliberately
   independent of anything else), so `dup_cluster` disjointness is now only
   enforced for external (non-511GA) rows, where it's the sole grouping key
   for still-image sources.

## Per-(source, label) spot-check table (orchestrator ruling)

| source | label | rule | population | n viewed | agreement | verdict |
|---|---|---|---|---|---|---|
| roadway_flooding | flooded | mask f>=0.10 | 429 | 20 | 20/20 (100%) | kept |
| roadway_flooding | wet | mask wet-band (unverified) | 8 | 8 | ~6/8 (75%) | EXCLUDED (n<20) |
| roadway_flooding | dry | mask dry-at-zero (unverified) | 1 | 1 | 0/1 (0%; road-closed/flood-adjacent scene) | EXCLUDED (n<20) |
| fred | flooded | mask road+water-hazard, f>=0.10 | 1865 | 20 | 20/20 (100%) | kept |
| fred | wet | mask wet-band (verified) | 164 | 20 | ~17/20 (85%; dry near-field, flooded road visible ahead under a "FLOODING" sign) | kept |
| fred | dry | mask dry-at-zero, within flooded sequences (verified) | 199 | 20 | 20/20 (100%) | kept |
| fred | dry | sequence_condition (always allowed) | 2616 | 20 | 20/20 (100%) | kept |
| flood_master_test | flooded | mask bottom-band (Greek video only) | 567 | 20 | 20/20 (100%) | kept (Italian video excluded) |
| nysdot_road_surface | dry | dataset_label, agreement>=0.67, cropped | 29 | 15 | ~13/15 (87%) | kept |
| nysdot_road_surface | wet | dataset_label, agreement>=0.67, cropped | 30 | 15 | ~14/15 (93%) | kept |
| ga511 | dry/wet/unusable | ai_review (visual) | 274 | 274 | n/a -- this IS the label | recorded to ai_review_labels.csv |

**Decision flagged for the orchestrator**: the `fred`/`wet` bucket's 85%
agreement is under a *broad* definition of "wet" (some real flood water
visible somewhere in the road corridor, near-field pavement itself often
looks dry). A stricter definition (near-field pavement must visibly show
wetness) would likely fail some of these. Kept as "wet" per the original
threshold design and because wet-but-not-flooded data is scarce (PLAN.md
risk #1), but this is a judgment call, not a clean-cut verification.

## 511GA ai_review

274 frames reviewed total across two passes (167 before the sweep finished,
107 more for newly-test-hashed cameras after). Distribution: **194 dry, 46
wet, 34 unusable** (unusable = lens obstruction/rain-on-lens bokeh, camera
framed on a sign board not the road, too dark/underexposed, heavy fisheye
distortion, or a corrupted video frame). The 46 "wet" frames came from two
visually-distinct clusters (elongated, glossy light-reflection streaks along
the lane lines, unlike the matte look of the ~194 dry frames) at different
timestamps roughly an hour apart -- consistent with a real, localized rain
event moving through the DeKalb/Gwinnett/Cobb/Fulton area during the
collection window, not a labeling artifact. No `likely_wet` weak-labeled
frames existed yet at review time (weather backfill was still catching up
in the first pass; the sweep's ~1,600 `likely_dry` rows arrived by the
final build but no `likely_wet` rows have appeared). No `manual` labels
exist yet (the user hasn't used the labeling tool).

## Final build (data_version = v1-9ef1414b)

4,229 manifest rows (`n_raw` 8,430 -> 4,338 after CLIP+resize+dedup/thinning
-> 4,229 after the weak_precip/test-split filter).

**Per split x label**

| split | dry | wet | flooded | total |
|---|---|---|---|---|
| train | 1694 | 43 | 782 | 2519 |
| val | 491 | 10 | 168 | 669 |
| test | 812 | 62 | 167 | 1041 |

**Per source x label**

| source | dry | wet | flooded | total |
|---|---|---|---|---|
| flood_master_test | 0 | 0 | 62 | 62 |
| fred | 1294 | 39 | 629 | 1962 |
| ga511 | 1674 | 46 | 0 | 1720 |
| nysdot_road_surface | 29 | 30 | 0 | 59 |
| roadway_flooding | 0 | 0 | 426 | 426 |

**Per split x source**

| split | flood_master_test | fred | ga511 | nysdot_road_surface | roadway_flooding |
|---|---|---|---|---|---|
| train | 0 | 1130 | 1177 | 28 | 184 |
| val | 0 | 210 | 290 | 15 | 154 |
| test | 62 | 622 | 253 | 16 | 88 |

**label_source counts**: weak_precip 1467, sequence_condition 1236, mask
1214, ai_review 253, dataset_label 59.

**water_frac_road stats** (mask-derived rows only; most `dry`/all NYSDOT and
weak_precip/ai_review rows have no water_frac): dry n=58 mean 0.000; wet
n=39 mean 0.057 (median 0.051); flooded n=1117 mean 0.814 (median 1.000).

**CLIP removals**: roadway_flooding 3/429 (0.7%), fred/nysdot/flood_master_test
0 (fred is no longer CLIP-filtered at all, see above).

**Dedup/thinning**: 3,387 frames thinned from FRED sequences + the Greek
video; 0 cross-source duplicate collapses this run. Per-group thinning
counts are in `reports/class_counts.md`.

**Split assertion**: `data/processed/splits_report.json` confirms "passed: no
group_id, camera_id, or dup_cluster spans two splits" (511GA checked by
camera_id only, external rows by group_id + dup_cluster, per the fix above).

## NYSDOT crop verification

Viewed 30 cropped images (15 dry, 15 wet) via
`reports/spotcheck_mask_nysdot_road_surface_{dry,wet}.png` (the official
spot-check deliverable, generated with the crop applied) plus another 30 in
an ad-hoc check -- zero header-text leaks in any of them; every image is
just the embedded camera frame, with a small in-frame "Camera NN" label
sticker that is part of the real camera overlay, not the weather-metadata
header.

## Artifacts

- Manifest: `data/processed/manifest.csv` (4,229 rows), `VERSION`
  (`v1-9ef1414b`), `splits_report.json`, `ga511_camera_splits.json`
  (1,262 cameras).
- Reports (tracked): `reports/class_counts.md`, `reports/figures/class_counts.png`.
- Reports (local, gitignored): `reports/samples_{dry,wet,flooded}.png`,
  `reports/augmentation_grid.png`, `reports/spotcheck_mask_*.png`,
  `reports/spotcheck_review_*.png`, `reports/clip_{kept,removed}_*.png`.
- Code: `src/prep/{common,sources,mask_rules,build_manifest,clip_filter,dedup,splits,augment,reports,spotcheck,label_tool}.py`.
- Tests: `tests/test_prep_{augment,splits,dedup,mask_rules,label_tool}.py`.
- Labels: `data/ga511/ai_review_labels.csv` (274 rows).

## Labeling tool

```
cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m prep.label_tool
# then open http://127.0.0.1:8765/
```
Options: `--host`, `--port`, `--ga511-root`. Filters: unlabeled, likely_wet,
by camera, by split. Keys: 1=dry, 2=wet, 3=flooded, 0=unusable, arrows to
navigate. Appends to `data/ga511/labels.csv` (last write wins); manual labels
always override `ai_review`.

## Packages installed

None beyond what the orchestrator already installed before Phase 2
(`torch` 2.14, `torchvision` 0.29, `open-clip-torch` 3.3.0 -- confirmed
present and working with MPS).

## What's unverified / remains

- The fred/wet 85% agreement is a judgment call on a broad "wet" definition
  (see the flagged decision above) -- worth a second opinion.
- 511GA `manual` labels: none yet (user hasn't used the labeling tool).
  `likely_wet` weak labels: none have appeared yet (only `likely_dry` and
  `uncertain` so far), so no additional mandatory ai_review was triggered by
  that rule this pass.
- The collector is still running and will keep adding frames/cameras;
  `build_manifest` is safely re-runnable (confirmed twice this session) and
  will pick up new frames, new `likely_dry` labels, and any new test-split
  cameras (which will still need ai_review -- the same procedure used here:
  compute `ga511_camera_splits` for the current camera list, diff against
  `ga511_camera_splits.json`, review the new test cameras' frames).
- roadway_flooding's excluded wet(8)/dry(1) buckets and Flood Area
  Segmentation (290, wholesale) remain a real loss of "wet-but-not-flooded"
  volume; PLAN.md risk #1 (scarce wet-not-flooded data) stands. FRED's wet
  bucket (39 final rows) and NYSDOT (59 rows total) are the only levers here
  and are both small.

## Decisions for the orchestrator

1. Confirm the fred/wet "broad" definition (distant flood visible ahead
   under a warning sign, near-field pavement dry) is acceptable for the
   "wet" class, or should be tightened (which would shrink wet-class data
   further).
2. Confirm forcing FRED locations and NYSDOT cameras to specific splits
   (rather than leaving it to the greedy stratified splitter) is the right
   long-term approach if more locations/cameras are added later -- the
   current constants (`FRED_FORCED_LOCATION_SPLITS`,
   `NYSDOT_FORCED_CAMERA_SPLITS`) are hardcoded to today's known locations
   and won't automatically rebalance a new location/camera.
3. Roadway Flooding's wet/dry buckets and Flood Area Segmentation are
   excluded; if the team wants to recover some of that wet-not-flooded
   signal, someone would need to hand-label a larger, real sample from
   those sources (dry-at-zero/wet-band can't be trusted below n=20 either
   way).
