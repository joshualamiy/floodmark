# ML workstream plan

Goal: a two-stage classifier for 511GA traffic camera frames around Atlanta.

- **Stage A:** dry road vs. wet road surface (wet includes flooded).
- **Stage B:** for wet frames, flooded vs. not flooded.
- Grad-CAM heatmaps showing where the model sees water.
- Both models exported to ONNX for fast CPU inference, wrapped in `src/inference/` for the backend.

Priorities: (1) a working baseline end to end, (2) honest evaluation on real traffic camera frames, (3) accuracy improvements. Depth estimation is an optional stretch goal after everything else passes.

## 1. Environment (Phase 0 recon, 2026-09-25)

| Item | Finding |
|---|---|
| Machine | Apple M5 Max, 18 CPU cores, 32-core GPU (Metal 4), 36 GB RAM |
| Free disk | 1.5 TiB free of 1.8 TiB |
| Python | `my_env/bin/python` is **3.12.14**, which TensorFlow supports. See the venv quirk below. |
| TensorFlow | 2.21.0 (Keras 3.15.1) installed, **CPU only** |
| GPU | `tensorflow-metal` 1.2.0 (latest) fails to load against TF 2.21 (`Library not loaded: _pywrap_tensorflow_internal.so`), so I uninstalled it. TF sees only the CPU. |
| CPU speed | MobileNetV3Small full fine-tune ≈ 280 img/s, inference ≈ 700 img/s. EfficientNetB0 full fine-tune ≈ 73 img/s, inference ≈ 285 img/s (224 px, batch 32). |
| Export | tf2onnx 1.17.0, onnx 1.23.0, onnxruntime 1.30.0 installed |

**Training on hardware we have.** Training runs locally on the CPU. With about 15k training images, a full EfficientNetB0 fine-tune epoch takes about 3.5 minutes, and head-only epochs are much faster. The fast baseline uses **MobileNetV3Small at 224 px**, and the main model uses **EfficientNetB0 at 224 px**. We also generate `notebooks/train_colab.ipynb` as a GPU option. It is not required, and uploading restricted data to Colab has license implications (see §4).

**Venv quirk (important).** `my_env/pyvenv.cfg` says Python 3.14.7. However, `my_env/bin/python` points to Python 3.12, while `my_env/bin/pip` runs under 3.14. The venv also has two separate `site-packages` folders, one for 3.12 and one for 3.14. As a result, `pip install x` (or `source activate` followed by `pip`) installs into the 3.14 side, where `python` can't see it. TensorFlow has no 3.14 wheels. **Rule: always use `my_env/bin/python` and `my_env/bin/python -m pip`.**

**CI constraints.** `.github/workflows/ci.yml` on `main` does the following on Ubuntu with Python 3.11:
- installs `flood-ml/requirements.txt`
- runs `ruff check flood-ml`
- runs `pytest backend flood-ml`

So:
- `requirements.txt` holds only the light inference runtime, pinned to versions with 3.11 wheels. `numpy` is pinned to 2.4.6 because 2.5.x needs Python 3.12 or newer.
- Everything else goes in `requirements-train.txt`.
- All code must pass ruff, including notebooks. Ruff settings are in `flood-ml/pyproject.toml`, with `target-version = py310`.
- Tests must run without data or models, and tests that need TensorFlow must skip when it isn't installed.
- Because pytest runs from the repo root in CI, `pyproject.toml` pytest settings don't apply there. A `conftest.py` must add `src/` to the path (Phase 5).
- Python floor: `onnxruntime` 1.30 and `keras` 3.15 need **Python 3.11 or newer**, so in practice we support 3.11–3.12.

**Repo.** Code lives at https://github.com/joshualamiy/floodmark and is **public**. Local `origin` now points there. We work on the `ml` branch, open one draft PR from `ml` to `main`, and never merge it.

**Credentials.**

| Credential | Status |
|---|---|
| `GA511_API_KEY` | Present in root `.env` |
| Kaggle | `~/.kaggle/access_token` present |
| Hugging Face | Logged in as `gthacksflood` |
| GitHub | Logged in as `joshualamiy` (repo scope) |
| AUTH FTP | Not needed, since the Flood Master Database comes from the Google Drive link |

**Data already on disk** (from an earlier session, listed in `data/raw/SOURCES.md`):

| Folder | Size | Contents |
|---|---|---|
| `raw/roadway_flooding` | 18 MB | 441 images + 441 masks (Kaggle, CC BY 4.0) |
| `raw/flood_area_segmentation` | 109 MB | 290 images + 290 masks (Kaggle, CC0) |
| `raw/water_segmentation` | 5.0 GB | V-FloodNet water set: 15,117 frames, 4,574 masks. License **unverified**. |
| `raw/fred` | 11 GB | Front camera only: 2,616 dry + 2,726 flooded frames, 2,674 labels (flooded only). CC BY-NC-SA 4.0 |
| `raw/tinycamml` | 107 MB | Git clone. No labeled roadway images. MIT |
| `restricted/flood_master` | 444 MB | Masks for 416 train + 132 val public images; Greek (567) and Italian (1,406) test video frames with masks |

## 2. Rules every subagent follows

These rules are pasted into every brief.

1. Work in `/Users/jzlamiy/Desktop/gt-hacks/flood-ml`. Edit or create files **only inside `flood-ml/`**.
2. Git is off limits for subagents: no commit, push, checkout, reset, or stash. The orchestrator commits.
3. Use `/Users/jzlamiy/Desktop/gt-hacks/my_env/bin/python` for everything, and `.../my_env/bin/python -m pip install <pkg>==<ver>` for any extra package. Never use `pip`, `pip3`, `python3.14`, or `source activate`. List every package you install in your report; the orchestrator updates the requirements files. Do not create virtual environments.
4. Load secrets from `/Users/jzlamiy/Desktop/gt-hacks/.env` with python-dotenv. Never print, log, or save a secret, or any URL that contains one. Redact `key=` query params in every log line and exception message.
5. Download only from official sources. Never bypass logins, license forms, or access gates. Ask before any download over 5 GB. Never upload data anywhere.
6. Never invent dataset names, URLs, or licenses. If you cannot verify something, write "unverified" and say what you tried.
7. **Restricted data** (the Flood Master Database) stays under `data/restricted/` or `data/processed/`, flagged `restricted=true`. **No dataset images** go into `docs/`, `notebooks/`, `tests/`, or `reports/figures/`. Image-bearing reports go in `reports/` itself, which is gitignored for everything except `.md`/`.csv`.
8. Code rules:
   - Use Python 3.10-compatible syntax (CI runs 3.11).
   - Code must pass `my_env/bin/python -m ruff check flood-ml`.
   - Each module is runnable as `cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m <pkg>.<module>`, with an argparse CLI behind `if __name__ == "__main__":`.
   - Use deterministic seeds.
   - Unit tests go in `flood-ml/tests/test_*.py`. They must need no network, data, or trained models, and must `pytest.importorskip("tensorflow")` if they use TF.
9. Send long output to files: job logs to `logs/jobs/<name>.log`, TensorBoard runs to `logs/<run_id>/`. Keep console output short.
10. Do not edit `docs/PLAN.md` or `docs/PROGRESS.md`. Write your report to `docs/phase_reports/<phase>_<name>.md` (text only, no restricted content). End your reply with that same concise report:
    - what you did
    - files created
    - key numbers
    - what failed or remains unverified
    - decisions the orchestrator needs to make

## 3. Design decisions

### Labels and stages

`label ∈ {dry, wet, flooded}`. Stage A is trained on all rows, with dry=0 and wet/flooded=1. Stage B is trained on wet and flooded rows only, with wet=0 and flooded=1.

**Status logic** in inference, with pA = P(wet surface) and pB = P(flooded | wet):
- `dry` if pA < tA
- otherwise `flooded` if pB ≥ tB
- otherwise `wet`

**Combined probabilities** (these match `docs/API.md`): `stage_probabilities = {dry: 1−pA, wet: pA·(1−pB), flooded: pA·pB}`, and `confidence = stage_probabilities[status]`. We tune tB on validation to favor precision (target ≥ 0.90), because false flood alerts destroy trust.

### Model

- Keras application backbone with built-in preprocessing, taking 224×224×3 input with values 0–255.
- Head: GAP → Dropout → Dense(1, sigmoid).
- Training: freeze the base and train the head, then fine-tune the top blocks with BatchNorm kept in inference mode. Use class weights.

**Grad-CAM inside ONNX.** Because the head is linear after global average pooling, Grad-CAM for the last conv layer equals CAM up to a positive scale: `ReLU(Σ_k w_k·A_k)`. So each exported ONNX model has **two outputs**, `prob` and `cam` (the CAM computed in-graph with the Dense weights). As a result:
- The inference module needs only `onnxruntime`, `numpy`, and `pillow`: no TensorFlow and no gradients at serve time.
- Keras GradientTape Grad-CAM is still implemented, and used to verify the ONNX `cam` output (correlation ≥ 0.99).

### Manifest

`data/processed/manifest.csv` has the required columns in this order:
`path, label, source, group_id, camera_id, view_type, license, restricted`

followed by extra columns:
`split, label_source, orig_path, mask_path, water_frac_road, phash, width, height, dup_cluster, weak_label, notes`

- `label_source` is one of `mask`, `sequence_condition`, `manual`, `weak_precip`, `dataset_label`.
- `data/processed/VERSION` holds `data_version = v<N>-<sha256(manifest)[:8]>`, which is copied into every `reports/runs.csv` row.

### Splits

Splits are group-aware, and no group may appear in more than one split (checked by an assertion).
- **511GA:** grouped by `camera_id`. About 20% of cameras are held out entirely as the **final test set**; the rest are split into train and val by camera.
- **External data:** grouped by source video, scene, or location:
  - FRED: normalized location name, so each location's dry and flooded sequences stay together.
  - V-FloodNet: video folder.
  - Flood Master: one group per test video.
  - Still-image sets: perceptual-hash duplicate cluster.
- The Flood Master Greek and Italian videos each form one group and go to the external test set by default.

**Weak labels.** Precipitation labels on 511GA frames may be used in train/val only when flagged `label_source=weak_precip` (`likely_dry` → dry). `likely_wet` needs manual confirmation first. The 511GA test set uses manual labels only.

### Artifacts

| Path | Contents | In git? |
|---|---|---|
| `models/` | `stage_a.onnx`, `stage_b.onnx`, `config.json` (thresholds, input size, backbone, data_version, run_ids), plus Keras checkpoints under `models/<run_id>/` | No |
| `logs/<run_id>/` | TensorBoard runs. `run_id = YYYYmmdd-HHMMSS_<stage>_<backbone>` | No |
| `reports/runs.csv` | One row per training run | Yes |
| `reports/*.md` | Text reports | Yes |
| `reports/figures/` | Metric charts only | Yes |
| `reports/errors.html`, sample grids, Grad-CAM overlays | Image-bearing reports | No, local only |

## 4. Licensing and publishing

- **The repo is public.** Git never contains dataset images, restricted data, trained weights, or image-bearing reports.
- **Flood Master Database** is licensed for non-commercial research use and may not be redistributed. It is used fully for training and evaluation, stays on this machine, and is credited in the README as "Flood Master Database" with a link to the AUTH source.
- **FRED** is CC BY-NC-SA 4.0 (non-commercial, share-alike). Any weights trained on it inherit those constraints, so we don't publish weights publicly. How to share `models/` with the backend teammate (a private channel) is **your decision**.
- **Water/V-FloodNet** license is unverified. Subagent 1 must resolve it before we rely on it.
- **Colab:** uploading restricted data to Colab means putting it on team-controlled Google storage. Local training is fast enough that we don't need to.

## 5. Phases and subagents

| Phase | Agent type | Count | Runs |
|---|---|---|---|
| 0 Recon | orchestrator | — | done |
| 1 Data acquisition | worker | 3 in parallel | after CP0 go-ahead |
| 2 Data prep | worker | 1 | after CP1 go-ahead |
| 3 Modeling | worker | 1, reviewed by orchestrator | after CP2 go-ahead |
| 4 Evaluation | reviewer | 1 | right after Phase 3 |
| 5 Integration | worker | 1 | after CP3 go-ahead |

Before Phase 1, the orchestrator has already installed the shared packages, so parallel workers don't race on pip. Before Phase 2, it installs `torch`, `open_clip_torch`, and later `gradio`, all of which are checked to resolve together with TF 2.21.

**File ownership in Phase 1**, so parallel agents never write the same file:

| Subagent | Owns |
|---|---|
| 1 | `data/raw/**` (additions only; never modify or delete existing files except `SOURCES.md`), `src/acquire/public_datasets.py`, `docs/DATASETS.md` |
| 2 | `data/restricted/**`, `src/acquire/flood_master.py`, `docs/FLOOD_MASTER.md` |
| 3 | `data/ga511/**`, `src/ga511/**`, `tests/test_ga511_*.py`, `reports/ga511_*`, `reports/figures/ga511_*` |

### Phase 1 / Subagent 1: public datasets (worker)

> Rules §2 apply.
>
> **Context:** An earlier session already downloaded most public datasets into `data/raw/` (see the table in PLAN.md §1 and `data/raw/SOURCES.md`). Your job is to verify, fill gaps, and document. **Don't download again** anything already present.
>
> **Tasks:**
> 1. For every dataset below, find and verify the **official** source: the paper, official GitHub, Kaggle page, or HF dataset card. Record the exact license string and where it is stated. Compare local file and image counts with the official numbers. Open every image with PIL (`verify()`) and report any corrupt or zero-byte files.
>    - **Roadway Flooding Image Dataset**: highest priority. Road-level flood images with masks. Local copy at `raw/roadway_flooding/Dataset/{images,labels}`. Trace it to the original paper (Sazara, Cetin, Iftekharuddin 2019) and verify the Kaggle license.
>    - **Flood Area Segmentation** (Kaggle `faizalkarim/flood-area-segmentation`): verify the license.
>    - **Water Dataset**: the local copy is Kaggle `gvclsu/water-segmentation-dataset` under `raw/water_segmentation/water_v{1,2}`, and appears to be the V-FloodNet (LSU) water set. Verify that link through the V-FloodNet paper and official GitHub, and **find its license**. Note that sub-collections such as the `ADE20K` folder may carry their own terms. List the per-folder (per-video) counts and which folders have masks.
>    - **FRED** (HF `CMalone-Jupiter/FRED`, logged in as `gthacksflood`, terms accepted):
>      - List the repo files with `huggingface_hub`. Confirm we hold every KITTI-style **front** camera image and **semantic label** file for both the dry and flooded sequences. If dry sequences do have front labels in the repo, download them, after checking their size first.
>      - Document the label encoding (class ids or colors for water hazard, road, other) from the dataset card.
>      - Record the license from the dataset card.
>      - Confirm that LiDAR, rear camera, and RTMaps files were skipped.
>    - **TinyCamML** (`github.com/TinyCamML/TinyCamML`, cloned at `raw/tinycamml`): confirm whether it includes labeled flooded vs. not flooded roadway images. If the project points to an image dataset hosted elsewhere, record it, but don't download it.
> 2. **FloodNet** (drone imagery, Hurricane Harvey): record its official source, license, size, image count, and label format. **Do not download it.**
> 3. **Gap search:** we have almost no *wet but not flooded* road images. Search for verifiable datasets or archives with dry vs. wet road-surface labels, for example road-surface-condition classification datasets or archived road-weather/DOT camera imagery. For each candidate, record the official source, license, size, view type, and relevance. **Do not download** these; they are proposals for the user. Exclude satellite datasets (for example SpaceNet 8).
> 4. **Write `data/raw/SOURCES.md`** (overwrite, keeping the useful existing notes). Use one table row per dataset with these columns: dataset, local folder, official source URL, license (verified/unverified + where), size on disk, image count, mask/label count, label format, view type (ground/vehicle/aerial), usability for dry/wet/flooded, notes. Add a "not downloaded" section for FloodNet and the gap-search candidates. Write the same content, public-safe, to `docs/DATASETS.md`. Leave a Flood Master row marked "filled by orchestrator".
> 5. **Write `src/acquire/public_datasets.py`**: an idempotent script that reproduces each download from its official source. Use the Kaggle API, `huggingface_hub.snapshot_download` with `allow_patterns` for FRED front images and labels only, and `git clone` for TinyCamML. It must skip datasets already present. Add a `--verify` mode that writes `data/raw/inventory.json` with per-dataset file counts, bytes, and image/mask counts. Run `--verify`, but **do not re-download**.
>
> **Report:** the per-dataset table, the license verification status, anything missing or corrupt, and the gap-search candidates.

### Phase 1 / Subagent 2: Flood Master Database (worker)

> Rules §2 apply.
>
> **License context:** Created by AUTH (AIIA Lab, Aristotle University of Thessaloniki). Our license agreement is signed, and AUTH confirmed in writing that the team may use the data for this project. The terms are non-commercial research only, and raw data must not be redistributed or published outside the team. The repo is public, so nothing from this dataset may enter git, docs, or figures.
>
> **Context:**
> - A local copy is already at `data/restricted/flood_master/`:
>   - `readme.txt`, `sources.csv`
>   - `train/{train.csv, annotations/}`, `val/{val.csv, annotations/}`
>   - `test/{test.csv, greek_test/{rgb,annotations}, italian_test/{rgb,annotations}}`
> - The official team copy is on Google Drive: https://drive.google.com/drive/folders/1JDGK1DQUzRNqFsir0ptQfeSMQF9kMV9Q
> - Train and val folders contain **masks only**. Their images come from Flood Area Segmentation, Roadway Flooding, and the Water Dataset (V-FloodNet), which we already downloaded to `data/raw/`.
>
> **Tasks:**
> 1. List the Drive folder's full contents, using `gdown` (`download_folder(..., skip_download=True)` or similar) or any available Google Drive connector. Compare it with the local copy by file names, counts, and sizes. Download anything missing or different (for example cleaned masks) into `data/restricted/flood_master_drive/`, keeping the Drive structure. Stop and report if the total exceeds 5 GB. If access needs something you can't do legitimately, report it.
> 2. **Integrity check:**
>    - Every CSV row's annotation file exists, and every test RGB frame exists.
>    - All masks decode.
>    - For each part, record the unique mask values and what they mean (water = 1?). This includes the test `frame<X>Ids.png` files. Use the readme where it helps.
> 3. **Resolve train/val `Image path` values to our local public files.** Match first by path or filename, then confirm by md5 or pixel equality, then by perceptual hash if needed. Write `data/restricted/flood_master/index.csv` with these columns:
>    `fmd_split, source, fmd_image_path, local_image_path, fmd_mask_path, public_mask_path, match_method (md5|pixel|phash|filename_only|unresolved), phash, group_id, restricted`
>    Set `group_id` to `fmd_greek_video` or `fmd_italian_video` for the test frames. Report counts of resolved and unresolved images per source.
> 4. **Dedup:**
>    - FMD train/val entries are the same images as our public downloads. Mark them so Phase 2 keeps **one row per image**, preferring the FMD cleaned mask.
>    - Check that the FMD test frames don't overlap any public download (pHash Hamming ≤ 6).
>    - Compute IoU between the FMD cleaned masks and the public masks for the same images, and report the stats.
> 5. The Greek and Italian test videos are sequences of near-duplicate frames. Each video is **one group**. Record frame counts and how much of each frame the water covers.
> 6. **Write `src/acquire/flood_master.py`** (Drive listing and fetch, verify, index build) and **`docs/FLOOD_MASTER.md`**. The doc is public-safe: structure, counts, mask semantics, and a summary of license terms. It also holds the credit line "Flood Master Database" with a link to the official AUTH/AIIA source page. Find that page; if you can't, say so. Don't paste the contact email or any data into it.
>
> **Report:** Drive vs. local diff, integrity results, resolution counts, dedup results, mask semantics, and the official credit link.

### Phase 1 / Subagent 3: 511GA camera collector (worker)

> Rules §2 apply. The API key is `GA511_API_KEY` in `.env`. **Never print or log full API URLs.**
>
> **Verified API facts:**
> - **Cameras:** `GET https://511ga.org/api/v2/get/cameras?key=KEY&format=json` returns every camera in one call. Each camera has `Id`, `Roadway`, `Direction`, `Latitude`, `Longitude`, `Location` (ends with the county in parentheses), `Name`, and `Views`. Each view has `Id`, `Url` (`https://511ga.org/map/Cctv/{view_id}`), `Status`, and `Description`.
> - **Events:** `GET https://511ga.org/api/v2/get/event?key=KEY&format=json` returns all events with `Latitude`, `Longitude`, `Description`, `EventType`, `Subtype`, `IsFullClosure`, and Unix timestamps.
> - **Throttle:** 10 calls per 60 seconds per key.
> - **Docs:** https://511ga.org/developers/doc
>
> **Build in `src/ga511/`:**
> 1. **`ratelimit.py`**: a **cross-process** limiter capped at **8 API calls per rolling 60 s**. Use `fcntl.flock` on a small state file in `data/ga511/`, so the collector, the event poller, and ad-hoc scripts all share it. Add exponential backoff with jitter on 429 and 5xx responses, plus a `redact()` helper that is used in every log line and exception message.
> 2. **`api.py`**: `get_cameras()` calls the endpoint once and caches the result to `data/ga511/cameras.json`, refreshing at most once per day. `get_events()` goes through the limiter.
> 3. **`cameras.py`**: filters to the Atlanta metro (lat 33.5–34.1, lon −84.7 to −84.1) and **skips views whose Status is not "Enabled"**. Writes `data/ga511/cameras_atlanta.csv` with `camera_id, view_id, lat, lon, roadway, direction, location, county, name, view_status, view_url`. Report counts: total cameras, cameras and views in the bounding box, and enabled views.
> 4. **`snapshot.py`**: fetches `https://511ga.org/map/Cctv/{view_id}?t={unix_timestamp}`.
>    - Use a descriptive User-Agent, `floodmark-research/0.1 (+https://github.com/joshualamiy/floodmark)`, with no personal email.
>    - Confirm that the URL returns an image.
>    - Limit fetches to **at most one per camera every 5 minutes**, enforced by last-fetch times persisted to disk, and at most about 1–2 requests per second overall.
>    - **Check whether image fetches count against the API throttle.** Image URLs carry no key. As an empirical check, do about 15 image fetches within a minute, then one API call through the limiter, and confirm there's no 429. Don't risk the key beyond that.
> 5. **`quality.py`**: detects dead frames and drops them, recording the reason:
>    - HTTP errors and non-image responses
>    - tiny files
>    - near-uniform images (low variance)
>    - "no image" or offline placeholders: build a small pHash catalog of the placeholder images you actually observe
>    - near-identical repeats of the same view's previous frame (frozen feed: pHash distance ≤ 2 or tiny pixel MAE)
> 6. **`weather.py`**: weak labels. For each frame, look up precipitation at that location and time.
>    - Prefer measured observations from the nearest NWS station (`api.weather.gov` points → observationStations → observations). Use Open-Meteo (historical/archive, or the forecast API with `past_days`) as a fallback.
>    - Cache per station and hour. Handle null values.
>    - Set `likely_wet` if precipitation in the last hour ≥ 0.2 mm or in the last 3 hours ≥ 1 mm. Set `likely_dry` if there was none in the last 6 hours. Otherwise set `uncertain`. Document the thresholds.
>    - These are candidates for manual review, not ground truth.
> 7. **`collect.py`**: the collector.
>    - Samples frames across many cameras and times of day on a rotating schedule, by default about one frame per enabled view per 60 minutes (configurable).
>    - Saves `data/ga511/frames/<view_id>/<utc_ts>.jpg` and appends rows to `data/ga511/frames.csv` with these columns: `frame_id, path, camera_id, view_id, lat, lon, roadway, county, timestamp_utc, http_status, bytes, width, height, phash, dead_reason, weak_label, precip_1h_mm, precip_3h_mm, precip_source`.
> 8. **`events.py`**: flood-event capture.
>    - Poll events every 5 minutes through the shared limiter.
>    - Flag an event whose Description, EventType, or Subtype mentions flood, flooding, water, or high water (case-insensitive), and record the matched keyword. For example, "water main" is a likely false positive; keep it, but mark it.
>    - When a new flagged event appears, **immediately** capture frames from every enabled camera within 1 km (haversine), still honoring the per-camera 5-minute rule. Save them to `data/ga511/flood_event_frames/<event_id>_<utc_ts>/` together with `event.json`, and log to `data/ga511/flood_events.log`.
> 9. **`map.py`**: writes an interactive folium map of Atlanta camera locations to `reports/ga511_camera_map.html`, and a plain scatter PNG with no camera images to `reports/figures/ga511_cameras.png`.
> 10. **`daemon.py`** (or a `collect.py --daemon` flag): runs the collector and the event poller in one long-lived process. Start it **detached** (`nohup`, PID file `data/ga511/collector.pid`, log `logs/jobs/ga511_collector.log`) so it keeps sampling across times of day after you finish. Provide exact start, stop, and status commands.
>
> **Run:** do an initial sweep for **at least a few hundred good frames across ≥ 150 cameras**, then start the daemon. Add unit tests (`tests/test_ga511_*.py`, no network) for the limiter, redaction, bounding-box filter, haversine, and flood keyword matching.
>
> **Report:**
> - camera and view counts
> - whether image fetches count against the throttle
> - frames collected and dead frames dropped, by reason
> - resolution distribution
> - weak-label distribution
> - daemon commands
> - map paths

### Phase 2: data prep (worker, one subagent)

> Rules §2 apply.
>
> **Inputs:**
> - `data/raw/*` with licenses from `data/raw/SOURCES.md`
> - `data/restricted/flood_master/index.csv`
> - `data/ga511/frames.csv`, plus `data/ga511/labels.csv` if present
>
> **Tasks:**
> 1. **`src/prep/build_manifest.py`**: deterministic and end-to-end, re-runnable after new labels arrive.
>    - Write resized copies (short side 256 px, JPEG q95) to `data/processed/images/<source>/...`.
>    - Write `data/processed/manifest.csv` with the columns and order from PLAN.md §3, one row per unique image. Where FMD and a public source share an image, keep one row and prefer the FMD cleaned mask.
>    - Write `data/processed/VERSION`.
> 2. **Mask labeling**: label an image `flooded` only if water covers a meaningful part of the **road region**.
>    - Where road labels exist (FRED), road region = road ∪ water-hazard pixels.
>    - Otherwise use a documented road-region prior (for example a lower-frame trapezoid) or a pretrained segmentation model, whichever spot-checks better.
>    - Store `water_frac_road`. Choose the threshold by spot-checking.
>    - Images with some water but below the threshold are **excluded**, not labeled wet, unless a spot check shows that they're really wet roads.
>    - FRED dry sequences → `dry` (`label_source=sequence_condition`).
>    - **Spot-check** at least 100 images, stratified by source and assigned label, by actually looking at them. Report agreement and any rules you changed.
> 3. **Road-scene filter**: CLIP zero-shot (`open_clip`, e.g. ViT-B-32) with "a photo of a road"-style positive prompts and negative prompts such as aerial, river, indoor, and field.
>    - Apply it to external sources.
>    - Report the number removed per source, and spot-check both removed and kept samples.
>    - Save grids of removed samples to `reports/` (local only).
> 4. **Dedup**: perceptual hashing across **all** sources (`imagehash.phash`, Hamming ≤ 6, bucketed search).
>    - Cross-source duplicates collapse to one row.
>    - Near-duplicate clusters within a source share `dup_cluster`, and all members stay in one split.
>    - Thin long video runs (FRED, V-FloodNet) to reduce redundancy, and report how many frames were thinned.
> 5. **Splits**: group-aware, following PLAN.md §3.
>    - Hold out about 20% of 511GA cameras entirely as the final test set, stratified so it includes any manually confirmed wet or flooded frames.
>    - External data is split about 70/15/15 by group, stratified by label.
>    - Hold out one FRED location entirely for test.
>    - Both Flood Master test videos go to the external test set.
>    - Assert that no `group_id`, `camera_id`, or `dup_cluster` spans two splits.
>    - Write `data/processed/splits_report.json`.
> 6. **Domain augmentation** in **`src/prep/augment.py`**: `camera_style(img: np.ndarray, rng) -> np.ndarray`, usable offline and from `tf.data` (via `tf.numpy_function`).
>    - Operations: downscale 0.25–0.6× then upscale; JPEG quality 15–60; Gaussian and Poisson sensor noise; slight blur; color, white-balance, and saturation shifts; high-angle perspective crops; timestamp-style text overlays.
>    - Overlays are applied with some probability to **all** sources, including 511GA, so an overlay can't reveal the source.
>    - Save a before/after grid to `reports/augmentation_grid.png`.
> 7. **Reports**:
>    - `reports/class_counts.md` (tracked): class counts per split and per source, plus weak vs. manual label counts.
>    - `reports/figures/class_counts.png`: bar chart only.
>    - `reports/samples_{dry,wet,flooded}.png`: sample grids per class, local only.
> 8. **Labeling tool** in **`src/prep/label_tool.py`**: a local web page on `127.0.0.1` using stdlib `http.server` plus one HTML page, with no new heavy dependencies.
>    - Shows 511GA frames with their weak label and precipitation. Filters: unlabeled, `likely_wet`, by camera.
>    - Keyboard shortcuts: 1 = dry, 2 = wet, 3 = flooded, 0 = unusable, arrow keys to navigate.
>    - Appends to `data/ga511/labels.csv` with `frame_id, label, labeled_at`; the last write wins.
>    - Document the launch command.
> 9. Tests (no data): augmentation output shape and dtype, split disjointness on a synthetic manifest, and the dedup clustering logic.

### Phase 3: modeling (worker, one subagent; the orchestrator reviews)

> Rules §2 apply. **Never compute metrics on `split == test`.** Use only train and val. The orchestrator will check for this.
>
> 1. **`src/train/data.py`**: a `tf.data` pipeline from the manifest.
>    - Stage A mapping: all rows, dry=0, wet/flooded=1. Stage B mapping: wet and flooded rows only, wet=0, flooded=1.
>    - Class weights come from training counts.
>    - Augmentation: `camera_style`, random crop to 224, and horizontal flip (no vertical flip). Measure throughput so the pipeline isn't the bottleneck.
> 2. **`src/train/model.py`**: backbone ∈ {`mobilenetv3small`, `efficientnetb0`} with ImageNet weights and built-in preprocessing, input 224×224×3 with values 0–255. Head: GAP → Dropout → Dense(1, sigmoid), kept exactly this shape (see the Grad-CAM note in §3).
> 3. **`src/train/train.py`**: `--stage {a,b} --backbone ... --epochs-head --epochs-ft --ft-layers`.
>    - First train the head with the base frozen, then fine-tune the top layers with BatchNorm in inference mode and a low learning rate.
>    - Early stopping on val AUC-PR.
>    - TensorBoard to `logs/<run_id>/`.
>    - Save the best checkpoint to `models/<run_id>/`.
>    - Append one row per run to **`reports/runs.csv`** with these columns: `date, run_id, stage, model, data_version, img_size, epochs_head, epochs_ft, n_train, n_val, val_auc_roc, val_auc_pr, val_precision, val_recall, val_f1, threshold, val_false_alarm_rate, train_time_s, logdir, notes`.
> 4. **Order of work:**
>    - **First a fast baseline**: MobileNetV3Small, head only, both stages, all the way through ONNX export and parity.
>    - **Then improve**: EfficientNetB0, fine-tuning, augmentation strength, and re-tuning the thresholds.
>    - Log every run.
> 5. **Thresholds**:
>    - tB is the lowest value on val that reaches precision ≥ 0.90. If no threshold gets there, maximize F0.5 and say so.
>    - Also report the val false-alarm rate on wet (not flooded) frames.
>    - tA balances Stage A errors.
>    - Save both to `models/config.json`, together with backbone, input size, data_version, and run_ids.
> 6. **`src/train/gradcam.py`**: GradientTape Grad-CAM for Stage B. Write overlays for a val sample to `reports/gradcam_val/` (local only).
> 7. **`src/train/export_onnx.py`**: export both stages with two outputs, `prob` and `cam` (in-graph CAM from the Dense weights), using tf2onnx.
>    - Verify with onnxruntime on ≥ 100 val images: max |Δprob| ≤ 1e-4 against Keras, and ONNX `cam` vs. Keras Grad-CAM correlation ≥ 0.99.
>    - Measure CPU latency per image for each stage and for the full pipeline: batch 1, median and p95 over 100 runs, both with 1 thread and with default threads.
>    - Write `models/stage_{a,b}.onnx` and `reports/onnx_parity.md`.
> 8. **`notebooks/train_colab.ipynb`**: clones the repo's `ml` branch, installs `requirements-train.txt`, mounts the team's data, and runs `train.py`. It must include a warning that restricted data must stay on team-controlled storage. It must be ruff-clean.
> 9. For your own debugging only: a val error gallery at `reports/errors_val.html`, local only.
>
> **Report:** the runs table, the chosen models, thresholds, parity results, and latency.

### Phase 4: independent evaluation (reviewer, one subagent)

> You did not build these models. Assume a problem until you've ruled it out. Rules §2 apply, except that you write only in `src/eval/`, `reports/`, and `docs/phase_reports/`. Don't modify training code, data, or models.
>
> 1. Write your **own** evaluation code in `src/eval/`. Evaluate the **ONNX deliverables** (and confirm they agree with Keras) on `split == test` **only**. Report separately:
>    - (a) the held-out 511GA cameras, using manual labels only, with counts and 95% CIs. If positives are too few, say so plainly.
>    - (b) the external test sets, per source.
> 2. **Metrics**:
>    - Confusion matrices for the 3-class status and for each stage.
>    - Per-class precision and recall with 95% CIs (group bootstrap).
>    - PR curves, saved to `reports/figures/pr_*.png`.
>    - **False-alarm rate on wet but not flooded frames**, P(status=flooded | true=wet), plus the same on dry frames.
> 3. **Shortcuts**:
>    - Train a quick classifier (logistic regression on frozen embeddings) to predict each image's **source dataset**. If that's easy, check whether flood predictions follow source: within-source metrics, and label-balanced-within-source analysis.
>    - Also probe simple cues: resolution, JPEG quality, overlay presence, brightness.
>    - Inspect Grad-CAM on a seeded random sample of ≥ 50 test images, and measure CAM energy inside the road or water region where masks exist, to confirm that the model looks at the road.
> 4. **Leakage**: check across splits for shared pHashes (Hamming ≤ 6), shared `group_id`, `camera_id`, `dup_cluster`, and `orig_path`, and for FMD vs. public duplicates.
> 5. **`reports/errors.html`** (local only): **every** misclassified test image with its true label, prediction, confidence, and Grad-CAM overlay taken from the ONNX `cam` output. Group them by error type: dry→wet, dry→flooded, wet→dry, wet→flooded (false flood alarm), flooded→dry and flooded→wet (missed flood). Put the generator in `src/eval/error_gallery.py`.
> 6. **Failure slices**: night, rain on the lens, glare, moved cameras. Tag these heuristically (brightness, blur/Laplacian variance, highlight clipping, reference-view change) and confirm the tags by eye. Report metrics per slice.
> 7. **`reports/EVALUATION.md`**: honest numbers, sample sizes, CIs, shortcut and leakage findings, known failure cases, and limitations.

### Phase 5: integration package (worker)

> Rules §2 apply.
>
> 1. **`src/inference/`**:
>    - `load_models(model_dir=None)`: reads `models/config.json` and creates onnxruntime CPU sessions, cached.
>    - `predict(image, models=None, heatmap=True)`:
>      - Accepts a PIL Image, an RGB uint8 ndarray, bytes, or a path, and resizes internally.
>      - Returns `stage_a_probs {dry, wet}`, `stage_b_probs {not_flooded, flooded}`, `status`, `confidence`, `stage_probabilities {dry, wet, flooded}` (summing to 1, matching `docs/API.md`), `heatmap_png` (bytes; the overlay is at frame resolution, capped at 640 px on the long side), and the thresholds and model versions used.
>    - The status and confidence logic follows PLAN.md §3.
> 2. **`TemporalSmoother(n=3)`**: reports `flooded` for a camera only after N consecutive flooded frames. Until then it reports `wet`. Skipped frames neither count toward N nor reset the streak.
> 3. **`CameraMoveDetector`**: compares a frame with that camera's reference view using structure, not color (edges plus ORB/homography inliers or edge-SSIM), and flags large changes so the frame is skipped. Reference views are built from the first K good frames or set explicitly.
> 4. **CLI** at `src/inference/cli.py`: runs on a folder of frames and prints one result per frame. Options: `--camera-id`, `--smooth-n`, `--json`.
> 5. **Gradio app** at **`src/demo_app.py`**: upload an image and see the status, confidence, stage probabilities, and heatmap. Local only (`share=False`).
> 6. **Tests**:
>    - Build tiny ONNX models on the fly with `onnx.helper`, using the same inputs and outputs as the real ones.
>    - Test the status and threshold logic, that probabilities sum to 1, every input type, that the heatmap PNG decodes, smoothing, and the camera-moved check.
>    - Add an integration test on real models that is skipped when `models/` is absent.
>    - Add `conftest.py` so `src/` is importable when CI runs `pytest` from the repo root.
> 7. **`README.md`** in `flood-ml/`, covering:
>    - setup (including the `my_env/bin/python -m pip` rule)
>    - data collection, training, and evaluation
>    - the inference API, with a backend snippet
>    - every dataset source and license
>    - the model's known limits
>    - the credit to "Flood Master Database" with a link to the AUTH source
>
>    Also write `docs/INFERENCE_API.md` noting any changes the team should make to the root `docs/API.md` through its own PR.

### Optional stretch (after everything passes)

`src/depth/`, kept separate from the core pipeline: a pretrained YOLO car detector plus a waterline-on-wheel estimate that outputs `shallow | dangerous | impassable | unknown`.

## 6. Checkpoints and git

- I commit on `ml` at every checkpoint, with short, lowercase messages. I push to `origin` (floodmark) and keep a single draft PR from `ml` to `main`, which I never merge.
- Order:
  1. **CP0** (this plan)
  2. Phase 1 → **CP1**. I wait for your go-ahead before any download over 5 GB and before Phase 2.
  3. Phase 2 → **CP2**.
  4. Phase 3 (commit), then Phase 4 → **CP3**.
  5. Phase 5 (commit) → final report.

## 7. Risks and open questions

1. **Wet but not flooded data is scarce.** Almost every external image is flooded or dry. Stage B's negative class, and its false-alarm rate, depend on 511GA frames captured during rain and on anything the Phase 1 gap search turns up.
2. **Real 511GA flooded frames will be rare or absent** unless a flood event happens while the collector runs. The flooded-class numbers on held-out 511GA cameras may rest on very small counts, and we will report them that way.
3. **Licensing:**
   - The Water/V-FloodNet license is unverified.
   - FRED's NC-SA terms and the Flood Master terms rule out publishing weights, so model sharing needs a private channel.
   - Image-bearing reports stay local.
4. **The collector runs continuously in the background** from Phase 1 on, making about one frame per view per hour, which is polite and under 1 request per second. Stop it with the command in the Phase 1 report.
5. **Venv pip shim:** `my_env/bin/pip` targets 3.14. Fixing its shebang is a two-line change, which I haven't made.
6. **GPU:** TF 2.18 with `tensorflow-metal` 1.2.0 might enable the GPU, but CPU training is adequate, so it isn't planned.
