# Progress log

## 2026-09-25: Phase 0 (recon)

- **Hardware:** Apple M5 Max (18 CPU cores, 32-core GPU), 36 GB RAM, 1.5 TiB free disk.
- **Python:** `my_env/bin/python` is 3.12.14, which TensorFlow supports. The venv mixes 3.12 and 3.14: `pip` targets 3.14, so we always use `my_env/bin/python -m pip`.
- **Packages installed:** TF 2.21.0, keras 3.15.1, tensorboard, tf2onnx 1.17.0, onnx 1.23.0, onnxruntime 1.30.0, and the Phase 1 data packages (pandas, pillow, imagehash, folium, opencv-headless, scikit-learn, matplotlib, pytest, ruff). numpy is pinned to 2.4.6, because 2.5.x drops Python 3.11, which CI uses.
- **GPU:** `tensorflow-metal` 1.2.0 breaks `import tensorflow` on 2.21, so I uninstalled it. Training is CPU only.
- **CPU speed:** MobileNetV3Small about 280 img/s, EfficientNetB0 about 73 img/s (full fine-tune at 224 px).
- **Git:**
  - Pointed `origin` at https://github.com/joshualamiy/floodmark (public).
  - Remote `main` was rewritten (`62615c9`) and no longer shares history with local `ml`/`main` (`57fd474`).
  - Re-basing `ml` onto `origin/main` was first blocked by a permission check. The user approved, so `ml` now sits on `62615c9`.
  - Pushed `ml` and opened draft PR #1 (`ml` → `main`, never merged).
  - Fixed the venv `pip`/`pip3` shebangs to use 3.12 (user approved).
- **Data inventory:** an earlier session already downloaded Roadway Flooding, Flood Area Segmentation, Water/V-FloodNet, FRED (front camera), TinyCamML, and a local Flood Master copy. See `docs/PLAN.md` §1.
- **Files created:**
  - the `flood-ml/` layout and `flood-ml/.gitignore` (image-bearing reports stay local)
  - `pyproject.toml` (ruff and pytest config)
  - `requirements.txt` (inference runtime) and `requirements-train.txt`
  - `docs/PLAN.md` with all subagent briefs

## 2026-09-25: Phase 1 (data acquisition) started

The user approved the whole pipeline ("go ahead with everything"). Checkpoints are still reported, but work continues without waiting.

Three worker subagents are running in parallel:
1. **Public datasets:** verify existing downloads and licenses, gap search for wet-road data, `SOURCES.md` / `DATASETS.md`.
2. **Flood Master Database:** Drive vs. local diff, index, dedup against public data, mask semantics.
3. **511GA collector:** rate limiter, collector, flood-event capture, weak labels, map, background daemon.

### Phase 1 results so far

**Public datasets (done).** No corrupt files. Details in `docs/DATASETS.md` and `docs/phase_reports/phase1_public_datasets.md`.
- Licenses verified:

  | Dataset | License |
  |---|---|
  | Roadway Flooding | CC BY 4.0 (Mendeley DOI 10.17632/t395bwcvbw.1) |
  | Flood Area Segmentation | CC0 |
  | FRED | CC BY-NC-SA 4.0 |
  | TinyCamML | MIT |

- FRED is complete for front images and labels. Dry sequences have no labels in the repo. Label encoding: 0 = other, 1 = road, 2 = water hazard.
- **Water Segmentation / V-FloodNet: excluded from the manifest.** No license grant was found (the official repo says "All rights are reserved"), and a by-eye check found no road scenes. It stays on disk, and the 30 Flood Master rows that use its images are dropped too.
- **New: NYSDOT Road Surface Conditions** (Zenodo 10.5281/zenodo.8370665, CC BY 4.0). 176 real DOT camera images with 6-coder labels: 43 wet, 36 dry, the rest snow, poor visibility, or obstructed. It's the only real wet-but-not-flooded source.
  - **Each image carries a rendered text header with weather metadata ("currently precipitating True", precipitation, snow depth).** That's a label leak, so Phase 2 must crop to the embedded camera frame.
- **Not downloaded:** FloodNet (aerial, CDLA-Permissive), RoadSaW (CC BY-NC-SA, bird's-eye crops), RSCD (license unverified).

**Flood Master Database (done).** Details in `docs/FLOOD_MASTER.md`.
- Drive and local copies match: 4,320 files each with identical paths. A full byte diff was blocked by the Drive download quota; the 54 files sampled were md5-identical.
- Integrity: no missing files, no decode errors, and all masks use {0,1} with 1 = water.
- `data/restricted/flood_master/index.csv` resolves 498/548 train/val images to local public files. The 50 unresolved are a V-FloodNet `flooding_pixabay` subset we don't have and don't need.
- FMD cleaned masks vs. public masks: IoU 0.989 on Flood Area Segmentation.
- Test videos: Greek (567 frames) and Italian (1,406 frames), 1280×720, with no near-duplicates in any public download.
- Credit: https://aiia.csd.auth.gr/flood-master-database/

**511GA collector (in progress).**
- There are 2,215 enabled camera views in the Atlanta bounding box, and the map is written.
- The first worker was killed by a session restart after only 60 test frames. A new worker is resuming from its code.
