# v3 retrain: iowa_rwis + eu_flood_2013 + alleyfloodnet, not_flooded label

Data version: `v1-7251bbd2` (was `v1-703f0040`). Backup of the pre-v3
manifest: `data/processed/manifest_v1-703f0040.csv` (sha256 prefix
`703f0040`, made by the orchestrator before this build finished; my own
`backup_old_manifest()` detects it and skips, never overwrites it).

## Part 1: manifest

### New sources added

- **iowa_rwis**: `src/prep/sources.py::iter_iowa_rwis`. Only the user's
  manual labels count (`data/othercams/iowa_rwis/labels.csv`, last write
  wins); weak (precip-based) labels are never used as a label, only carried
  through as `weak_label` for reference. 340 labeled frames, all usable (0
  `unusable`): 226 dry, 114 wet -- matches `docs/phase_reports/othercams.md`
  exactly.
- **eu_flood_2013**: `iter_eu_flood_2013`. Drops the 275 unlicensed
  "pollution" rows (`license` contains "unverified") and the 1 remaining
  `label == "unknown"` row, leaving 3,434 licensed flooded/not_flooded rows.
  See the road-filter section below for the CLIP threshold.
- **alleyfloodnet**: `iter_alleyfloodnet`, gated by `INCLUDE_ALLEYFLOODNET`
  (module constant, `prep/sources.py`) and a `--skip-alleyfloodnet` CLI flag
  on `build_manifest`. No natural grouping key in the source data, so
  `group_id=None` at gather time; `build_manifest.thin_and_dedup` fills it
  from the post-dedup pHash `dup_cluster` (same rule already used for
  roadway_flooding), giving group-aware splits without a new mechanism.
- floodimg: skipped entirely, per the brief.

### EU road-filter threshold

`road_score` is CLIP `best_positive - best_negative` margin (same convention
as `prep/clip_filter.score_batch`'s `is_road = pos > neg`), already computed
per-row in `data/raw/eu_flood_2013/candidates.csv` at acquisition time.
**Threshold: keep `road_score > 0.0`.**

Chosen by viewing 3 seeded contact sheets (local only, not committed):
36 images in the ambiguous band `[-0.06, +0.06]`, 24 at `road_score <= -0.10`,
24 at `road_score >= +0.10`.
- `<= -0.10`: uniformly pure river/lake/nature/aerial-map/gym-interior shots
  with no road in frame -- correctly excluded at any reasonable threshold.
- `>= +0.10`: uniformly real ground-level flooded street/building scenes --
  correctly kept.
- The `[-0.06, +0.06]` band is genuinely mixed (river-adjacent plazas,
  bridges, courtyards on both sides of 0), so no threshold in that band is
  clearly "correct" by eye; 0.0 is the same decision boundary the rest of
  this pipeline already uses for "is this a road scene," and it exactly
  reproduces the acquisition-time report's own numbers (61.0% of flooded
  rows, 72.2% of not_flooded rows score `> 0`), so I kept it rather than
  hand-picking a different cut with no stronger visual justification.

Removed: **1,304 of 3,434** licensed rows (37.9%) -- 1,213 flooded, 91
not_flooded. Kept: 2,130 (1,894 flooded, 236 not_flooded). Contact sheets:
`reports/eu_flood_near_zero_contact_sheet.png`,
`reports/eu_flood_very_negative_contact_sheet.png`,
`reports/eu_flood_very_positive_contact_sheet.png` (local only, gitignored).

### Cross-source dedup

Ran the existing global pHash dedup (`prep.dedup.cluster_phashes`, Hamming
<= 6) over every surviving row from every source combined, same as before.
**Cross-source duplicate clusters collapsed: 0** -- no collision found
between eu_flood_2013/alleyfloodnet and roadway_flooding, or with anything
else. (Temporal thinning within FRED/FMD-test sequences still removed 3,260
frames, unrelated to the new sources -- see `reports/class_counts.md`.)

### Bug found and fixed: old splits reshuffled by adding new external groups

`greedy_group_stratified_split` recomputes every group's split from scratch
over the WHOLE external pool each run (it has no persistence, unlike
`ga511_camera_splits`). Simply adding ~1,600 new external groups (eu_flood_2013
uploaders, alleyfloodnet dup clusters, iowa_rwis cameras) into that same
shared greedy pass reshuffled old groups too -- including roadway_flooding's,
whose `group_id` (a `dup_cluster` id) is itself just a renumbered id, not
stable content identity. **A first real build run flipped 66 old legacy rows**
(caught by my own `assert_legacy_test_rows_unchanged`, which raised instead of
silently writing a bad manifest).

Fixed with `prep.build_manifest.legacy_forced_splits`: looks up every
pre-v3-source row's split by `orig_path` (the true stable identity) in the
backup manifest, and pins its NEW `group_id` to that split before the greedy
pass runs, so `greedy_group_stratified_split` only ever decides genuinely new
groups. A merged-cluster edge case (a brand-new image pHash-bridging two
previously-separate old dup_clusters with different old splits) resolves
`test > val > train` and is logged; the real build hit **0** such conflicts.
Regression-tested in `tests/test_prep_build_manifest.py`.

### Old-test-rows-unchanged assertion: PASSED

`assert_legacy_test_rows_unchanged` (compares by `orig_path` against the
backup manifest): **2,472 legacy rows checked (782 of them test), 0
mismatches, 0 path-renumbering surprises.** ga511 is exempt by design (its
`frames.csv` keeps growing and test-camera frames still need manual labels);
its stable per-camera split file (`data/processed/ga511_camera_splits.json`)
was untouched in content for every previously-known camera (spot-checked:
1,243 old cameras, 0 disagreements; 38 genuinely new cameras, all landing
exactly where the pure deterministic hash rule alone would put them, so no
sample-size-dependent drift).

### New manifest counts

`data_version v1-7251bbd2`, 9,337 rows (up from 4,297).

| split | dry | wet | flooded | not_flooded | total |
|---|---|---|---|---|---|
| train | 3,050 | 92 | 2,530 | 521 | 6,193 |
| val | 826 | 22 | 542 | 112 | 1,502 |
| test | 957 | 31 | 542 | 112 | 1,642 |

| source | dry | wet | flooded | not_flooded | total |
|---|---|---|---|---|---|
| alleyfloodnet | 0 | 0 | 601 | 509 | 1,110 |
| eu_flood_2013 | 0 | 0 | 1,894 | 236 | 2,130 |
| flood_master_test | 0 | 0 | 62 | 0 | 62 |
| fred | 1,294 | 0 | 631 | 0 | 1,925 |
| ga511 | 3,284 | 1 | 0 | 0 | 3,285 |
| iowa_rwis | 226 | 114 | 0 | 0 | 340 |
| nysdot_road_surface | 29 | 30 | 0 | 0 | 59 |
| roadway_flooding | 0 | 0 | 426 | 0 | 426 |

Per split x source (train/val/test): alleyfloodnet 774/169/167,
eu_flood_2013 1494/317/319, flood_master_test 0/0/62, fred 1104/206/615,
ga511 2366/593/326, iowa_rwis 244/48/48, nysdot_road_surface 28/15/16,
roadway_flooding 183/154/89. Full tables (plus label_source counts, dedup/
thinning, spot-check table): `reports/class_counts.md`,
`reports/figures/class_counts.png`.

**iowa_rwis split**: stable salted hash per camera
(`build_manifest.iowa_rwis_forced_splits`, salt
`"iowa-rwis-mixed-split-v1"`), same mechanism as ga511's camera splits.
Deterministically ranks the 41 cameras with BOTH wet and dry frames by a
salted hash and pins the top 6 to test, the next 6 to val (documented in the
function's own docstring); everything else falls through to the normal
greedy stratified split. Result: val and test both landed with real wet+dry
mixed-camera coverage (val 48 rows, test 48 rows from iowa_rwis).

### Not_flooded label (Part 2)

- `train.data.select_stage_rows`: Stage A now excludes `label == "not_flooded"`
  rows entirely (their dry/wet state is unknown). Stage B "mixed" keeps them
  as negatives (`label_bin=0`, alongside dry and wet), never wet-upweighted.
- `train.data.select_all_rows` (new): every row regardless of label, for
  callers (pipeline eval) that need not_flooded present even though Stage A
  itself excludes it.
- `train.pipeline_eval.evaluate_pipeline`: excludes not_flooded from Stage A
  AUC/accuracy/threshold search and from the dry/wet/flooded confusion table;
  reports its false-flood rate (`false_alarm_not_flooded_rate`) as its own
  field, same shape as the true-dry/true-wet rates. Also added
  `stage_a_wet_recall` (now measurable) and `per_source_breakdown` (per-source
  slice at the already-tuned tA/tB).
- `train.candidate_eval`: switched its "get all val rows" calls to
  `select_all_rows` (was silently reusing a Stage-A selection hack that would
  now drop not_flooded rows).
- Tests: `tests/test_train_data.py` (Stage A exclusion, Stage B mixed
  inclusion, no wet-upweight leak), `tests/test_train_pipeline_eval.py`
  (Stage A unaffected by an adversarial not_flooded pA, false-flood rate
  reported, per-source slice, fixed-threshold mode for the baseline row).

## Part 3: retrain and select

*(filled in once training/selection finishes -- see the end of this report
for the live status as of the last update.)*
