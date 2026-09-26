# Flood photos: street/elevated-view gap fill

Why this exists: `reports/EVALUATION.md` found flood recall of 0.29 on the one
elevated fixed-camera flood video (vs. 0.91 on ground-level Roadway Flooding
photos), and validation has zero elevated-view floods, so the letterbox
(whole-frame) model built in `improve_v2` can't be judged. Flooded training
data was FRED dashcam + Roadway Flooding only. This pass searched for
verified, ungated, research-permitted street/elevated flood photo sources,
ideally ones that also carry real non-flooded images of similar scenes.

## What I searched, and what I ruled out

Leads followed (rule 6: never invent a dataset, name every one checked):

| Candidate | What it is | Why not used |
|---|---|---|
| MediaEval "Flood-Related Multimedia" / DIRSM (2017 Multimedia Satellite Task) | Flickr/YFCC100M images labeled flooded/non-flooded, 5,280+1,320 images | No public download found outside registering as a MediaEval task participant (a license-form gate); code repos (`keillernogueira/DIRSM`) ship only model code, not the images. **Gated, excluded.** |
| CrisisMMD (CrisisNLP, incl. 2017 Sri Lanka floods) | ~18k Twitter images/tweets across 7 disasters, informativeness + humanitarian-category + damage-severity labels | `crisisnlp.qcri.org/terms-of-use.html` requires agreeing to research-only, confidentiality, and delete-on-request terms before downloading -- a license-form gate per rule 4. Also: labels are informativeness/damage, not flooded/non-flooded, and images are Twitter-sourced with no clear per-image redistribution license. **Gated + label mismatch, excluded.** |
| `cvjena/twitter-flood-dataset` (Harz 2017 / Rhine 2018 floods) | Flood-relevance-annotated Twitter images from the same Jena group as European Flood 2013 | Ships tweet IDs + original-tweet image URLs only, not images (Twitter Developer Agreement restriction per its own README) -- no confirmed redistribution license for the actual pixels, and many tweet-hosted images will have gone dead. **Unverified per-image license, excluded.** |
| RoadSaW, RSCD | Vehicle-mounted road-surface-condition datasets | Already evaluated and left un-downloaded by Phase 1 / Subagent 1 (`docs/DATASETS.md`); not re-litigated here. |
| Zenodo 10.5281/zenodo.3472555 ("Scalable Flood Level Trend Monitoring with Surveillance Cameras") | CC0 code + DCNN weights for CCTV water-level segmentation | Contains only code and trained-weight files, no raw image dataset. **Nothing to download.** |
| AlleyFloodNet (Kaggle `seonyseony/alleyfloodnet`) | Ground-level alley/lowland flood photos | **Used** -- see below. |
| FloodIMG (Kaggle `hhrclemson/flooding-image-dataset`) | Multi-source flood photo compilation incl. SCDOT camera frames | **Used, but recommended against for bulk training** -- see below. |
| European Flood 2013 Dataset (`cvjena/eu-flood-dataset`) | Wikimedia Commons 2013 Central-Europe-flood photos + relevance labels | **Used** -- see below. |

## What was downloaded

All three land in new folders under `data/raw/`, downloaded and verified by
`src/acquire/flood_photos.py` (`download` / `verify` / `characterize` /
`contact-sheet` subcommands; idempotent, safe to re-run). 0 corrupt images
across all 5,350 files (`data/raw/*/inventory.json`). Total new disk: ~1.9 GB
(eu_flood_2013 1.1 GB + floodimg 747 MB + alleyfloodnet 67 MB), all well
under the 10 GB budget and the 5 GB ask-first threshold.

### 1. European Flood 2013 Dataset (primary recommendation)

- **Official source**: github.com/cvjena/eu-flood-dataset (Computer Vision
  Group, Univ. Jena); images via archive.org/details/european-flood-2013
  (no login); paper: Barz, Schröter, Münch, Yang, Unger, Dransch & Denzler,
  "Enhancing Flood Impact Analysis using Interactive Image Retrieval of
  Social Media Images," *Archives of Data Science, Series A* 5(1), 2018,
  arXiv:1908.03361.
- **License**: per-image, verified programmatically from `metadata.json`
  for all 3,435 Wikimedia-sourced images: CC BY-SA 3.0 (2,665), CC BY 3.0
  (212), CC BY 2.0 (125), CC BY-SA 4.0 (122), CC BY-SA 2.0 (112), CC0 (56),
  CC BY-SA 3.0 de (51), Public domain (32), CC BY 3.0 de (30), CC BY 4.0
  (8), CC BY-SA 2.5 hu (6), **GFDL (5)**, CC BY/CC BY-SA 3.0 at (3 each),
  Attribution (2), Copyrighted free use (2), CC BY 2.5 (1). All permit
  research use; most (CC BY-SA, GFDL) carry a share-alike/attribution
  obligation on redistribution -- same category of constraint as FRED, so
  I'd apply the same "don't publish weights publicly" caution the team
  already uses for FRED, to be safe. The 275 extra "pollution" (oil-spill)
  photos have no per-image license (manually harvested by the paper's
  authors from search engines) and are off-topic for road flooding;
  recommend excluding them from training.
- **Size / counts**: 1.1 GB, 3,710 images (3,435 Wikimedia + 275
  pollution), 0 masks. Label files are per-image relevance lists
  (`relevance/{flooding,depth,pollution,irrelevant}.txt`), hydrologist
  annotated.
- **Label rule I used for `candidates.csv`**: `not_flooded` if the image ID
  is in `irrelevant.txt` (327 images -- genuinely useful: these are
  same-event, same-place photos where the water is gone or never showed,
  e.g. sandbag cleanup, dry tram tracks, police cordons -- exactly the
  "similar non-flooded scenes" this search was for); `flooded` if in
  `flooding.txt` (3,107); else `unknown` (276, relevant only to the
  depth/pollution sub-task, ambiguous for a flooded/not-flooded classifier
  -- I'd drop these rather than guess).
- **View angle** (by eye, seeded 64-image contact sheet,
  `reports/flood_photos_eu_flood_2013_contact_sheet.png`, local only):
  roughly 85-90% ground-level street/riverbank photography, ~10-15%
  elevated (shot from a bridge, embankment, or hillside overlook), 0%
  aerial/drone. A second 32-image sheet of just the `not_flooded` pool
  (`reports/flood_photos_eu_flood_2013_notflooded_contact_sheet.png`)
  confirms real matched-scene negatives.
- **Road-scene relevance** (CLIP ViT-B-32 laion2b_s34b_b79k, via
  `prep.clip_filter`, read-only import): positive for 61% of `flooded` rows
  and 72% of `not_flooded` rows. The rest are river-bank/nature/plaza
  documentation shots with no road in frame -- this is a broad flood-
  documentation corpus, not a road-specific one, so Phase 2's CLIP filter
  should be applied here (it wasn't needed for FRED, but is needed here).
- **Grouping key**: Wikimedia uploader username (149 distinct groups; top:
  `Jedudedek` 641 photos, `Dr. Bernd Gross` 358, `Panoramio upload bot`
  227). Use as the `group_id` for group-aware splits -- one photographer's
  photos are very likely the same event/location.
- **Duplicates**: 9 exact-pHash clusters (18 rows) internally; 0 overlap
  with alleyfloodnet.

### 2. AlleyFloodNet (secondary recommendation -- cleanest label balance)

- **Official source**: Kaggle `seonyseony/alleyfloodnet`; paper: Lee, O. &
  Joo, H. (2025), "AlleyFloodNet: A Ground-Level Image Dataset for Rapid
  Flood Detection in Economically and Flood-Vulnerable Areas," *Electronics*
  14(10):2082, DOI 10.3390/electronics14102082.
- **License**: CC BY 4.0, verified via `kaggle datasets metadata`
  (`data/raw/alleyfloodnet/dataset-metadata.json`,
  `licenses: [{"name": "Attribution 4.0 International (CC BY 4.0)"}]`).
  **Caveat**: filenames (`gettyimages-*.jpg`, `hq720.jpg` [YouTube
  thumbnail], `images (N).jpg` [Google-search-saved]) and visible
  watermarks on a by-eye sample (a stock-photo-site watermark, a `JTBC
  News` watermark) show this was compiled from stock/news/social-media
  search results. The CC BY 4.0 tag is the authors' license over their
  curated dataset and labels, published in a peer-reviewed journal -- it is
  **not a demonstrated per-photo redistribution right** for every
  underlying image. I'm flagging this, not blocking on it; it's the
  orchestrator's call whether that's an acceptable risk for a hackathon
  research project.
- **Size / counts**: 64 MB, 1,110 images, natively split flooded (601) /
  non_flooded (509) by folder (`AlleyFloodNet/flood_{train_,test_}/
  {flooded,non_flooded}/`) -- the best-balanced flooded/not-flooded ratio
  of any source in the pipeline. (The paper describes a further train/val
  split that isn't materialized as separate folders in this Kaggle upload;
  only train/test folders exist on disk.)
- **View angle** (by eye, seeded 48-image contact sheet,
  `reports/flood_photos_alleyfloodnet_contact_sheet.png`, local only):
  **100% ground-level** -- narrow alleys, lowland streets, semi-basement
  entries, people wading, motorbikes splashing through water, both day and
  night. No elevated or aerial shots at all, so this source doesn't help
  the elevated-camera gap directly, but is a strong, clean ground-level
  flooded/not-flooded pair source.
- **Road-scene relevance**: CLIP-positive for 99.4% of rows (both labels)
  -- the cleanest hit rate of any source checked this pass, confirming
  these are genuinely street/alley scenes.
- **Grouping key**: **none available.** Filenames give no photographer,
  event, or location. Recommend Phase 2 use the pHash-dup-cluster as the
  group fallback, per PLAN.md §3's existing rule for "still-image sets."
- **Duplicates**: 35 exact-pHash clusters (70 rows, ~6%) internally; 0
  overlap with eu_flood_2013.

### 3. FloodIMG (downloaded, sampled, NOT recommended for bulk training)

- **Official source**: Kaggle `hhrclemson/flooding-image-dataset`; paper:
  Karanjit, R., Pally, R. & Samadi, S. (2023), "FloodIMG: Flood image
  DataBase system," *Data in Brief* 48:109164, DOI 10.1016/j.dib.2023.109164.
- **License**: CC0-1.0, verified via both the `kaggle datasets download`
  banner and `kaggle datasets metadata`
  (`data/raw/floodimg/dataset-metadata.json`). Kaggle's own metadata field
  names the compiled sources as **"Twitter, YouTube, South Carolina
  Department of Transportation (SCDOT), Google, Github"** -- the SCDOT
  mention is why I picked this dataset up in the first place (real
  elevated DOT traffic-camera flood frames are exactly the missing view
  type). Same per-photo-origin caveat as AlleyFloodNet, more acute: there
  is no per-image license or source tag anywhere in the download, and a
  by-eye sample shows visible stock/news watermarks (one placeholder image
  literally reads "Representative Image", another "alamy stock photo").
- **Why only a sample, not all 9,296**: ~12 GB upstream, over budget for a
  bulk pull of a dataset that turned out (see below) to be mostly waste.
  `src/acquire/flood_photos.py` samples deterministically
  (`deterministic_sample`, seed 42) via Kaggle's single-file download API,
  which I discovered **silently wraps some files in a server-side zip**
  (confirmed by inspecting the zip magic bytes and internal filename --
  this is a real, reproducible Kaggle API quirk, not a corrupt download);
  the download function now detects and unwraps this
  (`_kaggle_download_one`), and a mid-run cleanup pass fixed the ~103 files
  that landed as `*.jpg.zip` before the fix. Final state: 530 images
  downloaded, 0 corrupt, 0 stray zips, plus all 25 `Annotation/*.json`
  (a small labelme-style object-detection subset -- car/house/truck/sign/
  tree/person/bridge/boat polygons, each embedding its own image as base64
  -- unrelated to flooded/not-flooded labeling, kept as the official small
  label sample regardless).
- **No non-flooded images at all**: every image in this source is
  "flooded" purely by the dataset's stated scope. It cannot help the
  wet-but-not-flooded gap or provide matched negatives.
- **The finding that changed my recommendation** (this is the important
  part): I characterized the sample twice, once at n=243 and again at the
  final n=530, specifically to check whether an early signal held up:
  - **77.4% of the sample (410/530) is an exact pHash duplicate of
    European Flood 2013** (up from 61% at n=243 -- the rate *grew* with
    more data, so this isn't sampling noise). Kaggle's own source list
    names "Github" as one input, which is almost certainly this same
    eu-flood-dataset, and it looks like it was folded in more than once.
  - Of the 120 **non**-duplicate ("novel") rows, only **51 (42.5% of
    novel, 9.6% of the raw 530-image sample)** score road-scene-positive
    by CLIP.
  - I built a contact sheet of exactly that "novel + road-positive"
    residual (40 images, seeded,
    `reports/flood_photos_floodimg_novel_roadscore_contact_sheet.png`,
    local only) and looked at it by eye: it is **still mostly aerial/drone
    disaster photography** of flooded neighborhoods, trailer parks and
    farmland (CLIP's "road" prompts partially fire on aerial road-network
    shots too, so the score alone doesn't catch this). A real, genuine
    news-drone watermark ("QCAir Vision") is visible on one frame. Only a
    handful are true ground-level street-flood photos (e.g. two frames of
    a yellow "Water Over Roadway" / "Danger High Water" sign, one Humvee
    ground shot, one car in a flooded driveway).
  - Net effect: **roughly 2-4% of a raw random sample from this source is
    both genuinely new and genuinely a ground/elevated street-flood
    photo** -- the rest is either an exact duplicate of a source we
    already picked, or aerial/drone imagery PLAN.md §2 explicitly
    excludes.
- **Recommendation**: **do not bulk-train on this source.** If any of it
  is wanted, treat it as a small manual cherry-pick (pull the pHash-
  deduplicated, CLIP-road-positive residual and have a human confirm each
  one is ground/elevated, not aerial) rather than an automatic pipeline
  input. I stopped the sampling run at 530 (instead of the original
  2,000-image target) once this pattern was clear on two independent
  characterization passes -- spending more download time here would only
  have produced more of the same duplicate/aerial mix. A fresh
  `download --dataset floodimg --sample-size 2000` will resume from 530
  and pull more if the orchestrator wants a bigger sample anyway; it's
  fully idempotent.

## `src/acquire/flood_photos.py`

Subcommands (run from `flood-ml/` with `PYTHONPATH=src`):

```
acquire.flood_photos download   --dataset {eu_flood_2013,floodimg,alleyfloodnet,all} [--sample-size N] [--seed S]
acquire.flood_photos verify      --dataset {...,all}       # -> data/raw/<name>/inventory.json
acquire.flood_photos characterize --dataset {...,all} [--skip-clip]  # -> data/raw/<name>/candidates.csv
acquire.flood_photos contact-sheet --dataset <name> --n 48 --seed 0  # -> reports/flood_photos_<name>_contact_sheet.png (local only)
```

- All three downloads are idempotent (skip files already on disk).
- `verify` opens every image with PIL (`verify()` + a real decode) and
  writes per-dataset `inventory.json` + a log under `logs/jobs/`.
- `characterize` writes `data/raw/<name>/candidates.csv` with the columns
  the brief asked for: `path, label, label_source, road_score, view_guess,
  group_id, license, phash`. `road_score` is CLIP `best_positive -
  best_negative` (positive = more road-like); `view_guess` defaults to
  `"unknown"` for every row -- I did the ground/elevated/aerial estimate
  myself by eye on seeded contact sheets (reported above and saved to
  `reports/`) rather than trying to auto-tag ~5,350 images, per the
  brief's own "(if you tagged it)" allowance.
- Module-level imports are stdlib-only (kaggle/PIL/pandas/imagehash/
  open_clip are all imported lazily inside the functions that need them),
  so `tests/test_acquire_flood_photos.py` needs no entry in
  `tests/conftest.py`'s `NEEDS` map -- it imports the module and tests
  `deterministic_sample`, `floodimg_local_name`, and
  `parse_eu_flood_labels` with no network, data, or extra dependencies.
- Packages used, all already installed in `my_env` per the orchestrator's
  Phase 1 setup: `requests` (transitively, via `kaggle`), `kaggle`,
  `pandas`, `imagehash`, `Pillow`, `torch`/`open_clip` (via
  `prep.clip_filter`, imported read-only). **No new packages installed.**

## Numbers, at a glance

| Source | Images | Flooded | Not-flooded | Unknown | CLIP road-positive | Groups | Internal dup |
|---|---|---|---|---|---|---|---|
| eu_flood_2013 | 3,710 | 3,107 | 327 | 276 | 61% (flooded) / 72% (not_flooded) | 149 (uploader) | 9 clusters / 18 rows |
| alleyfloodnet | 1,110 | 601 | 509 | 0 | 99.4% | 1 (none real) | 35 clusters / 70 rows |
| floodimg | 530 (of 9,296) | 530 | 0 | 0 | 61% overall, 42.5% of non-eu-duplicate rows | 1 (none real) | 15 clusters / 30 rows; 77.4% exact-dup of eu_flood_2013 |

0 corrupt images in any source (`data/raw/*/inventory.json`). 0 pHash
overlap between eu_flood_2013 and alleyfloodnet.

## Recommendation for Phase-2-style prep

1. **eu_flood_2013**: add as source `eu_flood_2013`. Label rule:
   `flooding.txt` -> flooded, `irrelevant.txt` -> not_flooded, everything
   else dropped (ambiguous). Apply the CLIP road filter (it will remove
   real river/nature shots this time, unlike FRED). Group by uploader
   username. Treat CC BY-SA/GFDL rows the same as FRED for weight-sharing
   caution (private channel only, per PLAN.md §4). Consider excluding the
   275 unlicensed "pollution" images outright.
2. **alleyfloodnet**: add as source `alleyfloodnet`. Label rule: folder
   name (`flooded`/`non_flooded`). No CLIP filter needed (99.4% already
   road-relevant). Group by pHash-dup-cluster (no other key exists). Flag
   the per-photo-origin caveat for the record; it's a judgment call for
   the orchestrator whether a peer-reviewed CC BY 4.0 tag is sufficient
   given the underlying stock/news-photo appearance.
3. **floodimg**: **do not add as a bulk source.** If the orchestrator
   still wants the handful of genuine elevated DOT-camera / street-sign
   shots, that needs a manual pass over the CLIP-positive, non-duplicate
   residual (roughly 50 images in the current 530-image sample) -- not
   worth automating for this yield.

## What failed / remains unverified

- FloodIMG's per-image copyright status is unverified and, based on visible
  watermarks, likely includes commercially-licensed stock/news photography
  the uploader does not actually hold rights to redistribute -- flagged,
  not resolved. Same (milder) caveat applies to AlleyFloodNet.
- I could not find a legitimately downloadable street-level flood image
  dataset with confirmed per-image licensing that is dominated by elevated
  fixed-camera views specifically (the exact gap this search was for).
  European Flood 2013 has *some* elevated shots (~10-15%) but is mostly
  ground-level; FloodIMG's SCDOT component (the one lead that pointed at
  real elevated DOT-camera frames) turned out to be a small,
  visually-hard-to-isolate fraction of a mostly-duplicate, mostly-aerial
  9,296-image pool.
- MediaEval/DIRSM and CrisisMMD were the two best-fitting "social-media /
  crisis-image dataset with a flood category" leads named in the brief;
  both are genuinely gated (participant registration / terms-of-use
  click-through) and were not bypassed.
- I did not attempt to resolve individual-photographer contact for any
  GFDL/CC BY-SA images in eu_flood_2013; the per-image `metadata.json`
  already has everything Phase 2 needs (license string,
  `descriptionurl` back to the Wikimedia Commons page, `artist`) to credit
  correctly.

## Decisions for the orchestrator

1. Confirm the eu_flood_2013 + alleyfloodnet additions and the recommended
   label/group rules above.
2. Decide whether FloodIMG is worth a manual cherry-pick pass at all, given
   the ~2-4% genuine yield measured on the current 530-image sample -- my
   recommendation is no, but it's cheap to revisit since the raw data is
   already on disk (`data/raw/floodimg/candidates.csv` has the
   pre-computed `road_score` and `phash` needed to find that residual).
3. Decide how to treat the CC BY-SA/GFDL share-alike images in
   eu_flood_2013 for weight-publishing purposes (same open question as
   FRED, addressed the same way per PLAN.md §4: private channel, not a
   public weight release).
4. If the elevated fixed-camera gap still needs more coverage after this,
   the next thing I'd try is a targeted web/news search for individual,
   individually-licensed (or public-domain, e.g. government/NWS) photos or
   video stills of elevated urban flood cameras, rather than another bulk
   Kaggle "flood image database" compilation -- this pass's experience
   with FloodIMG suggests those compilations lean heavily on
   Google-Image-Search aerial content and duplicate each other.

## Files

- Code: `src/acquire/flood_photos.py` (new).
- Tests: `tests/test_acquire_flood_photos.py` (new, 6 tests, no network/data,
  no `tests/conftest.py` change needed).
- Data: `data/raw/eu_flood_2013/`, `data/raw/floodimg/`,
  `data/raw/alleyfloodnet/` (each: raw images + official label/metadata
  files + `inventory.json` + `candidates.csv`).
- Docs: this file; appended rows in `data/raw/SOURCES.md` and
  `docs/DATASETS.md` (existing rows untouched).
- Reports (local only, gitignored): `reports/flood_photos_eu_flood_2013_
  contact_sheet.png`, `reports/flood_photos_eu_flood_2013_notflooded_
  contact_sheet.png`, `reports/flood_photos_alleyfloodnet_contact_sheet.png`,
  `reports/flood_photos_floodimg_novel_roadscore_contact_sheet.png`.

## Packages

No new packages installed -- everything used
(`kaggle`, `pandas`, `imagehash`, `Pillow`, `torch`, `open_clip`) was already
present in `my_env` from the orchestrator's Phase 1/2 setup.
