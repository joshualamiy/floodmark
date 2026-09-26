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
