# Datasets

Public-safe summary of every dataset used by the ML workstream. No dataset
images live in this repo; this page documents where each dataset comes from,
its license, and how it's used. See `data/raw/SOURCES.md` for the fuller
version with per-folder notes (also public-safe -- `data/` itself is
gitignored, so neither file ever ships dataset pixels).

## Public datasets

| Dataset | Official source | License | Size | Images | Masks/labels | View | Role |
|---|---|---|---|---|---|---|---|
| Roadway Flooding Image Dataset | Kaggle `saurabhshahane/roadway-flooding-image-dataset`; original: Sazara, Cetin & Iftekharuddin (2019), Mendeley Data, DOI 10.17632/t395bwcvbw.1 | CC BY 4.0 (verified) | 18 MB | 441 | 441 | ground/street | flooded/wet road positives |
| Flood Area Segmentation | Kaggle `faizalkarim/flood-area-segmentation` | CC0 1.0 (verified) | 109 MB | 290 | 290 | ground, mixed | flooded positives |
| Water Segmentation Dataset (V-FloodNet / WaterNet lineage) | Kaggle `gvclsu/water-segmentation-dataset`; code: github.com/xmlyqing00/V-FloodNet | **Unverified / likely all-rights-reserved** -- Kaggle says "unknown", the official V-FloodNet repo says "All rights are reserved". Internal use only pending legal review; never redistributed. | 5.0 GB | 15,117 | 4,568 | river/canal/creek/harbor/beach (not road) | generic water-texture augmentation only |
| FRED (Flooded Road Environments Dataset), front camera | Hugging Face `CMalone-Jupiter/FRED`; preprint arXiv:2605.22018 | CC BY-NC-SA 4.0 (verified) | 11 GB | 5,342 (dry 2,616 / flooded 2,726) | 2,674 (flooded only) | vehicle-mounted, forward-facing | best dry/flooded ground-vehicle-view data; weights trained on it are not published publicly |
| NYSDOT Road Surface Conditions (ICR subset) | Zenodo 10.5281/zenodo.8370665 (Sutter et al. 2023) | CC BY 4.0 (verified) | 59 MB | 176 | 188 human-label rows (6 coders/image) | ground, DOT roadside traffic camera | fills the wet-but-not-flooded gap for Stage B negatives |
| TinyCamML | github.com/TinyCamML/TinyCamML; paper: Farquhar et al. (2026), *Water Resources Research* | MIT (verified) | 107 MB | 0 labeled roadway images | 0 | n/a | reference only (ships 2 pretrained MobileNetV2 `.tflite` classifiers, no dataset) |
| Flood Master Database | AIIA Lab, Aristotle University of Thessaloniki: https://aiia.csd.auth.gr/flood-master-database/ (team copy via private Google Drive) | License agreement signed; non-commercial research only, no redistribution. Kept under `data/restricted/`, never committed. | 444 MB | 1,973 test video frames (Greek 567, Italian 1,406) + masks for 548 train/val images from Flood Area Segmentation, Roadway Flooding and the Water dataset | 2,521 binary water masks (1 = water) | ground, street-level video and photos | cleaned masks for public flooded images; Greek and Italian videos used as an external test set (one group per video) |
| Iowa DOT RWIS webcams (archived by the Iowa Environmental Mesonet, Iowa State University) | `mesonet.agron.iastate.edu` (IEM archive JSON, no key); source cameras are the Iowa DOT RWIS network | Public domain (verified, IEM disclaimer: "materials found on this website are in the public domain and may be used freely by anyone for any lawful purpose") | see `docs/OTHERCAMS.md` / `docs/phase_reports/othercams.md` for current counts | collected by `src/othercams/iowa_rwis.py` into `data/othercams/iowa_rwis/` | 0 (weak precipitation labels only: `likely_wet` / `likely_dry`, same rule as `src/ga511`) | roadside/bridge RWIS camera, elevated -- same style as 511GA | matched dry vs. wet road-surface pairs from the **same** cameras, to fill the Stage A wet-but-not-flooded gap without a camera-identity shortcut |

## Not downloaded

| Dataset | Official source | License | Size | View | Why |
|---|---|---|---|---|---|
| FloodNet (Hurricane Harvey aerial) | github.com/BinaLab/FloodNet-Supervised_v1.0; arXiv:2012.02951 | CDLA-Permissive (verified) | 2,343 images, 10 classes | aerial (drone) | out of scope for a ground traffic-camera classifier; recorded for reference only, not downloaded |
| RoadSaW | viscoda.com (Cordes et al., CVPRW 2022) | CC BY-NC-SA 4.0 (verified) | size unpublished | vehicle-mounted, bird's-eye-view crops | candidate for more wet/dry volume later; not downloaded this pass (unknown size, bigger view-angle domain gap than NYSDOT) |
| RSCD | github.com/ztsrxh/RSCD-Road_Surface_Classification_Dataset (moved to thu-rsxd.com/rscd/) | unverified (GitHub shows an MIT badge but the page is unmaintained; not confirmed against the data itself) | 370,151 images | vehicle-mounted, cropped to road | license needs re-verification at the new host before use |

## Notes on the Water Segmentation dataset

This is the one dataset in the pipeline with a genuinely uncertain license.
Two sub-collections carry their own third-party terms regardless of what the
Kaggle mirror says: the `ADE20K/` folders (MIT CSAIL, non-commercial
research/educational use only, redistribution requires registration) and a
set of UK/Ireland river-camera stills branded "Farson Digital water cams"
(a commercial webcam network, no redistribution grant found). It is kept for
internal training only, is never committed to git (`data/` is gitignored
project-wide), and none of its images or derived masks are used in any
public artifact. Visual spot-checks across every named video folder found no
road-surface scenes -- the dataset is rivers, canals, creeks, harbors, and
storm-surge beach footage, so its practical value here is limited to generic
water-appearance augmentation rather than road-flood positives.

## Licensing implications for the trained models

- Any model fine-tuned using **FRED** (CC BY-NC-SA 4.0) inherits a
  non-commercial, share-alike constraint. Weights are shared with the team
  over a private channel, not published in the public repo.
- The **Flood Master Database** is licensed for non-commercial research use
  only and must not be redistributed; see `docs/FLOOD_MASTER.md`.
- **Roadway Flooding** (CC BY 4.0), **Flood Area Segmentation** (CC0), and the
  **NYSDOT** gap-fill set (CC BY 4.0) all permit redistribution with
  attribution and impose no share-alike or non-commercial constraint.
- The **Water Segmentation** dataset's unresolved license is treated as the
  most restrictive case (no redistribution) until further legal review.

## Street/elevated flood photos (gap-fill pass)

Added because the independent evaluation (`reports/EVALUATION.md`) found
flood recall on the one elevated fixed-camera source (a Greek flood video) is
0.29 vs. 0.91 on ground-level photos, and validation has zero elevated-view
floods. Full per-source counts, license breakdowns and view-angle estimates
are in `data/raw/SOURCES.md` and `docs/phase_reports/flood_photos.md`.

| Dataset | Official source | License | Size | Images | Masks/labels | View | Role |
|---|---|---|---|---|---|---|---|
| European Flood 2013 Dataset | github.com/cvjena/eu-flood-dataset (Univ. Jena); Barz et al. 2018, arXiv:1908.03361 | Per-image Wikimedia license (verified for all 3,435 Wikimedia images): mostly CC BY-SA / CC BY, some CC0 / public domain, 5 GFDL. 275 "pollution" photos have no per-image license (unverified, manually harvested, off-topic). | 1.1 GB | 3,710 (3,435 Wikimedia + 275 pollution) | 0 masks; image-level relevance labels only | ground + some elevated (bridge/embankment); by-eye ~85-90% ground, ~10-15% elevated, 0% aerial | flooded (3,107) and **real non-flooded (327)** street/cleanup scenes of the same event -- the "similar non-flooded scenes" pool this gap-fill was looking for |
| AlleyFloodNet | Kaggle `seonyseony/alleyfloodnet`; Lee & Joo 2025, *Electronics* 14(10):2082 | CC BY 4.0 (dataset-level grant; per-photo origin unverified -- filenames indicate stock/news/YouTube/Google-search compilation) | 64 MB | 1,110 (601 flooded / 509 not_flooded) | 0 masks; folder-name label | 100% ground-level (alleys, lowland streets, semi-basement entries) on a 48-image by-eye sample | best-balanced flooded/non-flooded ratio of any source in the pipeline; highest CLIP road-scene hit rate (99.4%) |
| FloodIMG | Kaggle `hhrclemson/flooding-image-dataset`; Karanjit, Pally & Samadi 2023, *Data in Brief* 48:109164 | CC0-1.0 (dataset-level grant; no per-image license or source tag; visibly includes stock/news/social imagery) | ~12 GB upstream; downloaded a 530-image deterministic random sample instead (778 MB) | 530 of 9,296 available upstream | 0 (every image is "flooded" only by the dataset's own scope; 25 `Annotation/*.json` files carry unrelated object-detection polygons, not water masks) | **mixed and disappointing**: a large fraction of the sample is aerial/drone disaster photography, which this project excludes; genuine elevated DOT-camera views are a small, hard-to-isolate minority | **not recommended as-is** -- 77.4% (410/530) exact-pHash duplicate of European Flood 2013, and of the 120 non-duplicate rows only 51 (9.6% of the raw sample) score road-scene-positive by CLIP, and even those are still mostly aerial by eye; would need aggressive dedup + road-scene filtering + manual review before any training use |

**Caveat that applies to AlleyFloodNet and FloodIMG alike**: both carry a
single dataset-level CC BY 4.0 / CC0-1.0 tag from their Kaggle uploader, but
neither publishes a per-image license or original-source record, and their
filenames show real stock-photo/news/social-media compilation (Getty Images,
a TV news watermark, YouTube thumbnails, saved Google Image Search results).
The blanket license is the authors' grant over their curated dataset and
labels, not a demonstrated redistribution right for every underlying photo.
European Flood 2013 is the one new source here with real per-image (Wikimedia)
license provenance.

> Final cleanup (2026-09-26): the local copies of **Water Segmentation** (excluded, license unclear) and **floodimg** (skipped, duplicates/aerial) were deleted. Neither is used by any model. `src/acquire/` can re-download both.
