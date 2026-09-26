# Phase 1 / Subagent 2: Flood Master Database

Scope: `data/restricted/**`, `src/acquire/flood_master.py`, `docs/FLOOD_MASTER.md`.
No restricted content appears below -- only paths, counts, and stats.

## 1. Drive vs. local diff

Listed the shared Drive folder with `gdown.download_folder(url, skip_download=True)`
(gdown 6.4.0 has no 50-file cap in this version -- it recursed the whole tree
and returned all files in one call, so `remaining_ok` wasn't needed/available).

- **Drive: 4320 files. Local copy: 4320 files. Identical relative paths on
  both sides -- 0 additions, 0 removals.** Breakdown: `train/annotations`
  282, `val/annotations` 87, `test/greek_test` 1134 (567 rgb + 567
  annotations), `test/italian_test` 2812 (1406 + 1406), plus
  `readme.txt`, `sources.csv`, `train.csv`, `val.csv`, `test.csv`.
- Attempted a full byte-level (md5) diff by downloading the whole folder to
  `data/restricted/flood_master_drive/` (structure kept). **Google Drive's
  shared-file download quota kicked in after 54 files** ("Cannot retrieve
  the public link of the file ... may have had many accesses"). I confirmed
  this wasn't specific to already-touched files by testing two previously
  untouched files (`sources.csv`, one val annotation) -- both failed
  immediately with the same error, so the whole folder is currently
  quota-blocked for further downloads today. This is Google's own
  abuse-prevention limit on a heavily-shared link, not something to work
  around (no bypassing access gates per the rules).
  - **The 54 files that did download before the quota hit are 100% md5-identical
    to the local copy** (0 diffs).
  - I deleted the partial `flood_master_drive/` directory afterward since it
    added no information beyond "identical" and isn't needed going forward.
  - `src/acquire/flood_master.py download-drive` / `diff-drive` are ready to
    re-run for a full diff once the quota resets (typically ~24h) or from a
    different network/account. **Decision for orchestrator:** is a full
    byte-diff worth retrying later, or is name/count parity + the 54-file
    sample (both 100% clean) sufficient given the signed license + AUTH's
    written confirmation already cover provenance?

## 2. Integrity

All read via `PYTHONPATH=src ../my_env/bin/python -m acquire.flood_master verify`
(`data/restricted/flood_master/verify_report.json`, gitignored).

- **train.csv**: 416 rows, 0 missing annotations, 0 decode errors.
- **val.csv**: 132 rows, 0 missing annotations, 0 decode errors.
- **test.csv**: 1973 rows, 0 missing RGB frames, 0 missing annotations, 0 decode errors.
- **Mask semantics**: every mask (train/val annotations, Roadway's own
  labels, and test `frame<X>Ids.png`) decodes to single-channel uint8 with
  exactly the value set `{0, 1}`. Interpreting `1 = water` (background/road
  = 0) is consistent with readme.txt ("we normalized the binary masks in
  the {0,1} values") and with water-fraction sanity checks below.
- **Water fraction stats** (fraction of pixels == 1):
  - Train: FAS mean 0.41 (median 0.37, range 0.04-0.88, n=217); Roadway mean
    0.45 (median 0.46, range 0.05-0.92, n=134); Water Dataset mean 0.40
    (median 0.40, range 0.07-0.86, n=65).
  - Val: FAS mean 0.40 (n=72); Roadway mean 0.47 (n=45); Water Dataset mean
    0.44 (n=15).
  - Test whole-frame: Greek mean 0.65 / median 0.67 (n=567); Italian mean
    0.36 / median 0.28 (n=1406).
  - Test bottom-60%: Greek mean 0.64 / median 0.69; Italian mean 0.44 /
    median 0.35.
  - Both test videos: resolution 1280x720 throughout, frame ids contiguous
    (Greek 8326-8892, i.e. a 567-frame excerpt of a longer video; Italian
    0-1405, i.e. the entire 1406-frame video), matching readme.txt.
- **Data-quality finding**: 7 of 416+132 = 548 Flood Area Segmentation train/val
  rows (1.3%) have an Image/Mask pixel-dimension mismatch (e.g. mask
  `14Ids.png` is 1024x630 while `Image/14.jpg` is 1900x1425, and vice versa
  for `15Ids.png`/`Image/15.jpg`). I resized each mismatched mask to its
  image's size (nearest-neighbor) and computed IoU against that source's own
  public mask anyway: **IoU came back 0.95-0.99 for all 7**, i.e. content
  still matches correctly once resized -- this looks like an image/mask
  resolution inconsistency in the Kaggle "flood-area-segmentation" mirror
  (different resize pipeline per file), not a mislabeled/swapped pair.
  Flagged for awareness; no action needed unless Phase 2's mask loader
  assumes image and mask share pixel dimensions (it should resize either way).
  No such mismatches in Roadway Flooding or Water Dataset rows.

## 3. Resolution of train/val images to public downloads

Match rule (documented in the module docstring): Flood Area Segmentation and
Roadway Flooding rows resolve deterministically by concatenating the CSV's
`Image path` onto the known local dataset root (single candidate, so
`match_method=filename_only`); Water Dataset rows strip the
`WaterDataset/train_images/JPEGImages/` prefix and look for the remaining
subpath under our `water_v2` then `water_v1` Kaggle mirrors (checked for
v1/v2 filename collisions across all 80 Water Dataset rows -- **none
found**, so there was never an ambiguous case needing pixel/md5
disambiguation).

Resolved / unresolved counts (train + val combined):

| Source | Resolved | Unresolved | Total |
|---|---|---|---|
| Flood Area Segmentation | 289 | 0 | 289 |
| Roadway Flooding Image Dataset | 179 | 0 | 179 |
| Water Dataset | 30 | 50 | 80 |
| **Total** | **498** | **50** | **548** |

**The 50 unresolved rows are all in a `flooding_pixabay` subfolder** that
FMD's `Image path` values reference (e.g.
`WaterDataset/train_images/JPEGImages/flooding_pixabay/114492.jpg`) but that
does not exist anywhere under our local `water_v1`/`water_v2` Kaggle mirror
(checked both `JPEGImages` trees exhaustively -- no folder named
`*pixabay*` and none of the sampled filenames exist anywhere in
`data/raw`). `sources.csv` lists the Water Dataset's origin as "V-Floodnet"
via a *different* Google Drive link than our Kaggle download, so this
still-image subset likely lives there and simply wasn't part of the Kaggle
mirror Subagent 1 downloaded. **Decision for orchestrator**: this is
Subagent 1's territory (`data/raw/**`); flagging so they can decide whether
to pull that subset from V-FloodNet's own Drive folder. Until then, these 50
rows have an FMD mask but no image and are marked `match_method=unresolved`
with an empty `local_image_path` in `index.csv`.

## 4. Dedup and IoU

- **Train/val dedup**: `index.csv` is the dedup marker -- for every row with
  `match_method=filename_only`, `local_image_path` points at the exact
  public image Phase 2 will also see in its own per-source manifest.
  Phase 2 should join on `local_image_path` and keep one row, preferring
  `fmd_mask_path`. Documented as an explicit contract in both
  `docs/FLOOD_MASTER.md` and the module docstring.
- **IoU (FMD cleaned mask vs. the source dataset's own public mask)**:
  - Flood Area Segmentation: n=289, mean 0.989, median 0.992, min 0.952, **0
    rows below 0.8**.
  - Water Dataset: n=30, mean/median/min all **1.0** (the Kaggle mirror's
    Annotations are already pure black/white, so thresholding at 127
    reproduces FMD's own {0,1} normalization exactly).
  - Roadway Flooding: not computed -- FMD's "annotation path" for this
    source *is* the original dataset's label file (same file on disk), so
    IoU is 1.0 by construction and not informative.
- **Test-frame overlap check**: built a phash (imagehash.phash) corpus of
  all 16,966 public still images we have (Flood Area Segmentation 290,
  Roadway Flooding 441, Water Dataset v1+v2 10,893, FRED front-camera
  dry+flooded 5,342; TinyCamML has no labeled images per Subagent 1's
  brief, excluded). Used 8-band LSH bucketing (8 bytes of the 64-bit phash,
  each as an exact-match bucket -- by pigeonhole this is lossless for
  Hamming <= 7, so <= 6 is covered) instead of an O(n*m) pairwise compare
  over 1973 x 16966 ~= 33M pairs. **Result: 0 of the 1973 Greek/Italian test
  frames match any public image within Hamming <= 6.** No leakage between
  FMD's test videos and our other public downloads. Corpus build + search
  took ~155s.

## 5. Greek/Italian test videos (one group each)

| Video | Frames | Frame id range | Resolution | Contiguous |
|---|---|---|---|---|
| Greek (YouTube) | 567 | 8326-8892 | 1280x720 | yes |
| Italian (NYTimes) | 1406 | 0-1405 (whole video) | 1280x720 | yes |

`index.csv` sets `group_id=fmd_greek_video` / `fmd_italian_video` for every
row from that video, per PLAN.md's group-aware split rule (§3: "The Flood
Master Greek and Italian videos each form one group ... by default").

## 6. Files created / packages installed

- `src/acquire/flood_master.py` -- CLI with `list-drive`, `download-drive`,
  `diff-drive`, `verify`, `build-index` subcommands. Ruff-clean.
- `docs/FLOOD_MASTER.md` -- public-safe structure/counts/mask-semantics/
  license summary + credit line, citing the official AIIA Lab source page
  (found via web search, confirmed by fetching the page):
  <https://aiia.csd.auth.gr/flood-master-database/>. Includes the two
  papers the AIIA Lab asks to be cited (found on that same page).
- `data/restricted/flood_master/index.csv` (2521 rows: 416 train + 132 val +
  1973 test) and `index_stats.json`, `verify_report.json` (all gitignored
  under `data/`).
- `tests/test_flood_master.py` -- 11 unit tests for the pure logic (phash
  Hamming/bucketing, IoU, path resolution), no network/data/models. All
  pass. `ruff check` clean on all new files.
- **No extra packages installed** -- gdown 6.4.0, imagehash, pillow, numpy,
  pandas, opencv-python-headless were already present as stated in the brief.

## What failed / remains unverified

- Full byte-level (md5) diff of the Drive folder vs. local copy is
  **incomplete** (54/4320 files, all identical) because Google's shared-link
  download quota blocked further downloads. Path/name/count parity is 100%
  confirmed for all 4320 files, which is strong evidence the copies match,
  but this isn't a full guarantee. Retry `download-drive` + `diff-drive`
  later if a full diff is wanted.
- 50 of 80 Water Dataset train/val rows (41 train + 9 val, see table above)
  have no resolvable public image (`flooding_pixabay` subset missing from
  our Kaggle water_v1/v2 mirror). Not fixable from within this subagent's
  file ownership (`data/raw/**` belongs to Subagent 1).
- The AIIA Lab page's exact non-commercial/no-redistribution license
  wording lives in a PDF license agreement, not on the public page itself; I
  did not fetch that PDF (it's the signed agreement already handled outside
  this task, per the brief's license context). `docs/FLOOD_MASTER.md`
  states the terms as given in this task's brief.

## Decisions for the orchestrator

1. Should the Drive full-content diff be retried later (quota reset), or is
   the current evidence (100% path/count parity + 54/54 identical sample)
   good enough to proceed?
2. Should Subagent 1 (or a follow-up) fetch the `flooding_pixabay` subset
   from V-FloodNet's own Google Drive link (in `sources.csv`) so the
   remaining 50 Water Dataset rows become usable, or should Phase 2 just
   proceed with the 498 resolved train/val images and drop the rest?
