# Flood Master Database

**Credit:** "Flood Master Database", created by the Artificial Intelligence and
Information Analysis (AIIA) Lab, Aristotle University of Thessaloniki (AUTH).
Official source page: <https://aiia.csd.auth.gr/flood-master-database/>

If you publish work that uses any portion of this dataset, the AIIA Lab asks
that you cite:

- P. Mentesidis, V. Mygdalis and I. Pitas, "Improve Real-time flood
  segmentation by encoding and distilling foreground information," IEEE
  International Conference on Image Processing (ICIP), Anchorage, Alaska,
  USA, 13-17 September 2025.
- A. Gerontopoulos, D. Papaioannou, C. Papaioannidis and I. Pitas,
  "Real-Time Flood Water Segmentation with Deep Neural Networks," IEEE 25th
  International Symposium on Cluster, Cloud and Internet Computing Workshops
  (CCGridW), Tromso, Norway, pp. 85-91, 2025.

## License

Non-commercial research use only, under a signed license agreement between
AUTH and this team. AUTH has confirmed in writing that the team may use the
data for this project. **The raw images, masks, and video frames must not be
redistributed or published outside the team** -- they stay under
`data/restricted/` on team machines, are never committed to this (public)
git repository, and never appear in any figure, notebook, or report. Any
model weights trained on this data should be shared with teammates through a
private channel, not by publishing them from this public repo.

## Structure

The team's local copy lives at `data/restricted/flood_master/` and mirrors
the official Google Drive folder (confirmed by a full file-listing
comparison: the Drive folder and the local copy contain the same 4320 files
at the same relative paths -- see the phase report for how the byte-level
diff was limited by Google's shared-file download quota).

```
flood_master/
  readme.txt, sources.csv        # dataset provenance (public dataset names + URLs)
  train/{train.csv, annotations/}
  val/{val.csv, annotations/}
  test/test.csv
  test/greek_test/{rgb, annotations}
  test/italian_test/{rgb, annotations}
  index.csv                      # built by src/acquire/flood_master.py, see below
```

Train and val ship **masks only** (416 + 132 = 548 rows). Their images come
from three public datasets we already downloaded to `data/raw/`:

| Source | Train | Val | Total |
|---|---|---|---|
| Flood Area Segmentation (Kaggle) | 217 | 72 | 289 |
| Roadway Flooding Image Dataset (Kaggle) | 134 | 45 | 179 |
| Water Dataset / V-FloodNet | 65 | 15 | 80 |

For Roadway Flooding rows, the "annotation path" in `train.csv`/`val.csv`
points at that dataset's own label file (already normalized to {0,1}), not a
separate FMD mask.

Test ships **frames and masks** from two real flood videos:

| Video | Frames | Frame ID range | Resolution |
|---|---|---|---|
| Greek (YouTube) | 567 | 8326-8892 (contiguous) | 1280x720 |
| Italian (NYTimes) | 1406 | 0-1405 (contiguous, whole video) | 1280x720 |

Each video's frames are one contiguous run extracted at its native FPS, so
consecutive frames are near-duplicates. Downstream splitting must keep each
video's frames entirely inside one split (`group_id` = `fmd_greek_video` /
`fmd_italian_video` in `index.csv`).

## Mask semantics

Every mask file (train/val annotations, test `frame<X>Ids.png`, and the
Roadway Flooding source labels) decodes to an 8-bit single-channel image
with exactly two values: **0 = background/road/other, 1 = water**. This was
confirmed across all 2521 train+val+test rows (no unexpected values, no
decode failures).

Water coverage (fraction of frame with mask value 1) for the test videos:

| Video | Whole-frame mean | Whole-frame median | Bottom-60% mean | Bottom-60% median |
|---|---|---|---|---|
| Greek | 0.65 | 0.67 | 0.64 | 0.69 |
| Italian | 0.36 | 0.28 | 0.44 | 0.35 |

## Using this dataset in the pipeline (`index.csv`)

`data/restricted/flood_master/index.csv` (restricted, not in git) has one
row per FMD annotation with columns `fmd_split, source, fmd_image_path,
local_image_path, fmd_mask_path, public_mask_path, match_method, phash,
group_id, restricted`. Phase 2 should join its public-dataset manifest rows
to this index on `local_image_path`; where a row matches, use
`fmd_mask_path` (FMD's cleaned mask) instead of the public dataset's own
mask, and keep exactly one manifest row per image. See the module docstring
in `src/acquire/flood_master.py` for the full contract.

Not every train/val row currently resolves to a local public image: the
Water Dataset source references a `flooding_pixabay` still-image subset that
is not present in our Kaggle mirror of the V-FloodNet water set (see the
phase report). Those rows are marked `match_method=unresolved` and have no
`local_image_path`; they carry an FMD mask but no image until that gap is
filled.

## Reproducing this

`src/acquire/flood_master.py` provides:

```
PYTHONPATH=src ../my_env/bin/python -m acquire.flood_master list-drive
PYTHONPATH=src ../my_env/bin/python -m acquire.flood_master download-drive
PYTHONPATH=src ../my_env/bin/python -m acquire.flood_master diff-drive
PYTHONPATH=src ../my_env/bin/python -m acquire.flood_master verify
PYTHONPATH=src ../my_env/bin/python -m acquire.flood_master build-index
```
