# Phase 1 / Subagent 1: Public datasets

Scope: verify every public dataset already on disk, trace each to its
official source and license, check for corrupt files, resolve the
Water/V-FloodNet license question, confirm FRED completeness, record FloodNet
without downloading it, and gap-search (and where justified, download) for
wet-but-not-flooded / dry-vs-wet ground-view road data.

## Per-dataset table

| Dataset | License | Status | Images | Masks/labels | Corrupt |
|---|---|---|---|---|---|
| Roadway Flooding Image Dataset | CC BY 4.0 | verified (Mendeley Data page + Kaggle metadata) | 441 | 441 | 0 |
| Flood Area Segmentation | CC0 1.0 | verified (Kaggle metadata) | 290 | 290 | 0 |
| Water Segmentation Dataset (V-FloodNet/WaterNet) | unknown / "all rights reserved" | **unverified, likely no redistribution grant** | 15,117 | 4,568 real masks (+6 non-mask leftover files, see below) | 0 |
| FRED (front camera) | CC BY-NC-SA 4.0 | verified (HF dataset card) | 5,342 (dry 2,616 / flooded 2,726) | 2,674 (flooded only) | 0 |
| NYSDOT Road Surface Conditions (new download) | CC BY 4.0 | verified (Zenodo API) | 176 | 188 label rows | 0 |
| TinyCamML | MIT | verified (repo LICENSE) | 0 (no labeled dataset) | 0 | n/a |
| FloodNet (not downloaded) | CDLA-Permissive | verified (official GitHub README) | 2,343 (aerial, not counted here) | n/a | n/a |

Total on disk after this pass: **16.3 GB**, 21,366 images + 7,973 masks/labels
across the six downloaded datasets, **zero corrupt or zero-byte files** found
by `PYTHONPATH=src ../my_env/bin/python -m acquire.public_datasets verify`
(full detail in `data/raw/inventory.json` and
`logs/jobs/public_datasets_verify.log`).

## What I did

1. **Roadway Flooding Image Dataset**: confirmed the Kaggle mirror matches
   the primary source (Sazara, Cetin & Iftekharuddin 2019, Mendeley Data,
   DOI 10.17632/t395bwcvbw.1) on both count (441) and license (CC BY 4.0,
   stated on the Mendeley page itself). This corroborates the earlier
   session's byte-identical check against a copy obtained directly from
   Mecit Cetin.
2. **Flood Area Segmentation**: license re-confirmed as CC0-1.0 from Kaggle's
   own dataset metadata (`data/raw/meta/flood_area_segmentation.json`).
3. **Water Segmentation Dataset**: traced the Kaggle citation (WaterNet,
   Liang et al. 2020, *Computational Visual Media*) to its successor,
   V-FloodNet (Liang, Li, Tsai, Chen & Jafari, *Environmental Modelling &
   Software* 2023, github.com/xmlyqing00/V-FloodNet) -- same LSU GVCLSU lab;
   `water_v1` matches the WaterNet era, `water_v2` adds V-FloodNet-era
   folders. **The license question is resolved as "no usable grant found":**
   Kaggle says `"licenses": [{"name": "unknown"}]`, and the official
   V-FloodNet GitHub README states *"All rights are reserved"* by the
   corresponding author -- no LICENSE file exists anywhere in the chain. I
   listed per-folder image/mask counts for every one of the 9 (v1) + 22 (v2)
   named video folders and visually spot-checked a representative image from
   each ambiguous one. **None of the non-ADE20K folders are road-surface
   scenes**: `houston` looks down at a Houston bayou with an elevated
   freeway in the background (not the road itself); `buffalo0`/`canal0`/
   `creek0`/`stream0-3` are USGS-style streamgauge trail-cam images (staff
   gauges, sandbags); `lab0` is an indoor calibration rig, not a real scene;
   the ten UK/Ireland town folders added in v2 (`aberlour`, `auldgirth`,
   `bewdley`, `cockermouth`, `dublin`, `evesham-lock`, `galway-city`,
   `holmrook`, `keswick_greta`, `worcester`) are "Farson Digital water cams"
   branded commercial river-level webcam stills (rivers/harbors, no
   redistribution grant found); `mexico_beach_clip0`/`holiday_inn_clip0`/
   `gulf_crest`/`boston_harbor2_small_rois` are Hurricane Michael storm-surge
   beach/harbor footage. The embedded `ADE20K/` folders (1,888 images/masks
   in both v1 and v2) carry MIT CSAIL's own non-commercial,
   registration-gated terms independent of whatever the Kaggle page claims.
   **Recommendation:** keep this dataset internal-training-only (it never
   reaches git since `data/` is gitignored project-wide), do not redistribute
   it or any weights meaningfully derived from it, and treat it as low-value
   for the road classifier itself (no positives, just generic water
   texture) -- the orchestrator should decide whether the legal risk is
   worth keeping it at all for Phase 2/3.
4. **FRED**: listed the full HF repo (30,344 files) with `huggingface_hub`
   and diffed it against the local copy. Every KITTI-style front image
   (5,342/5,342), front-label (2,674/2,674 -- **zero** dry-sequence labels
   exist anywhere in the repo, so there was nothing extra to fetch), imu
   (4,476/4,476), utm (4,476/4,476), and `ground_plane_eqn.txt` (14/14) file
   is present; back-imgs/back-labels, ouster/ouster_ground_labels, and
   native-RTmaps are confirmed absent locally and correctly excluded. License
   is CC BY-NC-SA 4.0 per the HF dataset card's `cardData`. Documented the
   label encoding by combining the arXiv preprint text (*"three semantic
   classes: 'water hazard', 'road', and 'other'"*, road=red polygons,
   water=green polygons) with direct pixel inspection of several
   `front-labels/*.png` files: palette index **0 = other/background (black)**,
   **1 = road (dark red)**, **2 = water hazard (dark green)** -- confirmed
   consistent across five sampled files from different sequences.
5. **TinyCamML**: confirmed (again) no labeled roadway image dataset in the
   repo; the two `.tflite` files are pretrained MobileNetV2 flood
   classifiers. Traced the project to its now-published paper (Farquhar et
   al. 2026, *Water Resources Research* 62, e2025WR042023) but could not find
   a public training-image dataset reference: the EarthArXiv preprint excerpt
   only gives accuracy numbers, and the published version returned HTTP 403
   (Wiley paywall). **Unverified** where their training images live, if
   public at all.
6. **FloodNet**: recorded, **not downloaded**. Official source
   github.com/BinaLab/FloodNet-Supervised_v1.0 (Rahnemoonfar et al., 2020,
   arXiv:2012.02951), license **CDLA-Permissive** per the repo's own README
   (one secondary source claims CC BY-SA 4.0 instead; I trust the primary
   repo). 2,343 aerial images (3000x4000, DJI Mavic Pro post-Hurricane-Harvey),
   10 semantic classes including `Road-Flooded`/`Road-Non-Flooded`.
7. **Gap search and download decision**: searched for ground/vehicle-view
   road-surface-condition datasets with dry/wet labels (excluding satellite
   and aerial). Found three real candidates:
   - **NYSDOT Road Surface Conditions** (Zenodo 10.5281/zenodo.8370665,
     Sutter et al. 2023, CC BY 4.0): downloaded (see below).
   - **RoadSaW** (viscoda.com, Cordes et al., CVPRW 2022, CC BY-NC-SA 4.0,
     publicly downloadable, no registration seen): **not downloaded** --
     total size is unpublished and likely large ("large-scale" video +
     patch data), and its view type (rectified bird's-eye-view crops at
     7.5-30 m) is a bigger domain gap from 511GA's oblique wide-angle camera
     view than NYSDOT's raw DOT-camera frames. Flagged as a strong candidate
     if more wet/dry volume is needed later.
   - **RSCD** (github.com/ztsrxh/RSCD-Road_Surface_Classification_Dataset,
     370,151 images, dry/wet/water/snow/ice friction labels): **not
     downloaded** -- the GitHub page shows an "MIT" badge but says the repo
     "has been transferred to our new website
     (https://thu-rsxd.com/rscd/) and will no longer be maintained"; I did
     not independently confirm the data's own license at the new host, so
     it's unverified.

   I downloaded the **NYSDOT** dataset because it is the only one that met
   all four bars in the brief: official source verified (Zenodo, DOI
   resolves, md5 of the zip matches the record exactly:
   `fc7924e225393f732cc9b2545a034ebe`), license verified and permissive
   (CC BY 4.0 -- stronger than "non-commercial" is even required to satisfy),
   ground-level DOT roadside traffic-camera view with explicit human `wet`
   **and** `dry` labels, and total size (60.8 MB zip, 59 MB extracted) far
   under the 10 GB cap. One caveat surfaced while inspecting it: the parent
   paper (arXiv:2510.06440) trains on a private ~21,653-image corpus and
   states the full set is **not** publicly released; the Zenodo record is a
   smaller (176-image) subset the authors published specifically to
   document a 6-coder inter-coder-reliability exercise. I verified this by
   downloading and inspecting the zip directly rather than trusting page
   summaries alone, since one source implied it held only documentation. It
   genuinely contains 176 real camera images plus an Excel sheet of 6
   independent coders' per-image labels; I built `labels.csv` from that sheet
   (majority-vote label + every coder's raw vote + agreement fraction).
   Majority-vote label counts: **wet 44, dry 36, snow 35, snow_severe 32,
   poor_vis 28, obstructed 13**. This is small, but it is the only dataset in
   the whole pipeline that is genuine ground-level DOT traffic-camera imagery
   (same 511-network family as 511ga.org, via 511ny.org) with an explicit
   wet-not-necessarily-flooded label, directly filling the Stage B negative
   gap named in `PLAN.md` §7 risk 1.
8. Wrote `data/raw/SOURCES.md` (full detail, internal) and
   `docs/DATASETS.md` (public-safe summary, with the Flood Master row marked
   "filled by orchestrator" as instructed).
9. Wrote `src/acquire/public_datasets.py`: idempotent `download` (per-dataset
   or `all`, `--allow-large` gate for the two datasets over 5 GB) and
   `verify` subcommands. Ran `verify` only (no re-downloads); it writes
   `data/raw/inventory.json` and `logs/jobs/public_datasets_verify.log`.

## Files created / modified

- `data/raw/SOURCES.md` (overwritten, per file ownership)
- `data/raw/meta/nysdot_road_surface.json` (new)
- `data/raw/nysdot_road_surface/` (new dataset: `NYSDOT_quantitative_content_analysis/{Images/Trial1..4,NYSDOT_codebook_image_labeling.pdf,NYSDOT_ICR_Example.xlsx}`, `labels.csv`)
- `data/raw/inventory.json` (new, written by `verify`)
- `docs/DATASETS.md` (new)
- `src/acquire/public_datasets.py` (new)
- `logs/jobs/public_datasets_verify.log` (new)
- `docs/phase_reports/phase1_public_datasets.md` (this report)

I did not touch anything under `data/restricted/`, `data/ga511/`, `src/ga511/`,
`src/acquire/flood_master.py`, or `docs/FLOOD_MASTER.md`.

## Packages installed

- `openpyxl==3.1.5` (to read the NYSDOT Excel inter-coder-reliability sheet;
  pulled in `et_xmlfile==2.0.0` as a transitive dependency). Not yet in
  `requirements-train.txt` -- please add it.

No other new packages were needed; `kaggle==2.2.4` and
`huggingface_hub==1.32.0` were already installed and working (Kaggle auth via
`~/.kaggle/access_token`, HF auth via the `gthacksflood` login, confirmed
with `HfApi().whoami()`).

## What failed or remains unverified

- **Water Segmentation Dataset license**: could not find any redistribution
  grant. Kaggle says "unknown"; the official V-FloodNet GitHub says "all
  rights reserved". Treat as internal-training-only pending legal review;
  it is already excluded from git by the repo-wide `data/` gitignore.
- **TinyCamML training data provenance**: unverified. The published paper is
  paywalled (403 from Wiley); the EarthArXiv preprint excerpt I could fetch
  doesn't name a dataset or repository.
- **RSCD license**: unverified beyond a GitHub page badge; the dataset's
  primary host has moved to thu-rsxd.com/rscd/, which I did not fetch.
- One secondary source claims FloodNet is CC BY-SA 4.0 instead of
  CDLA-Permissive; I used the official repo's own README as the
  authoritative answer but flagged the conflict in `SOURCES.md`.

## Decisions for the orchestrator

1. **Water Segmentation Dataset**: decide whether to keep using it at all
   given the unresolved "all rights reserved" status. My recommendation is
   to keep it for internal augmentation only (never redistributed, already
   gitignored) and drop it if the team wants zero legal exposure, since it
   contributes no road-scene positives anyway.
2. **RoadSaW**: worth a follow-up look if Stage B still needs more
   wet-but-not-flooded volume after Phase 2 dedup/filtering -- CC BY-NC-SA
   4.0, publicly downloadable, but size and exact image count need to be
   checked on the actual download page before committing to it.
3. Add `openpyxl==3.1.5` to `requirements-train.txt`.
