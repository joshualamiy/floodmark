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

**511GA collector (done, Checkpoint 1).** Details in `docs/phase_reports/phase1_ga511.md`.
- The first worker was killed by a session restart, and a later one was stopped by accident. Each new worker resumed from the existing code.
- There are 4,332 cameras statewide and 2,215 enabled views in the Atlanta bounding box. Map: `reports/ga511_camera_map.html` (local) and `reports/figures/ga511_cameras.png`.
- Image fetches do **not** count against the API key's rate limit (checked empirically). API calls go through a cross-process limiter capped at 8 per 60 s, and a grep confirms the key appears in no log or file.
- **Sweep:** one full pass over all 2,215 views gave 1,248 good frames from about 1,080 cameras. Dead frames: 1,428 placeholders (about 52% of views are offline right now), 41 frozen repeats, and a few others. Resolutions: 450×253 (most), 352×240, 320×240.
- **Weak labels:** all `likely_dry`, with a few `uncertain`. It's a dry night. Nearby NWS stations returned null precipitation, so Open-Meteo was used as the fallback.
- **Daemon:** running detached (PPID 1), about one frame per view every 85–90 minutes, plus an event poll every 5 minutes with flood-event capture. So far 150 events were checked and none were flagged.
  - Status: `cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m ga511.daemon status`
  - Stop: `cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m ga511.daemon stop`
- **Not yet exercised live:** the NWS precipitation path and end-to-end flood-event capture. Both are unit-tested only.

## 2026-09-25: Phase 2 (data prep) started

The user said not to wait for the collector. One worker is building the manifest, road filter, dedup, splits, augmentation, and labeling tool. The manifest build can be re-run as more 511GA frames arrive.

Decisions passed to the worker:
- **Excluded:** Water Segmentation / V-FloodNet, and the Flood Master rows that use its images.
- **NYSDOT:** crop to the camera frame to remove the label-leaking header. Keep majority dry/wet labels with ≥ 0.67 coder agreement.
- **511GA test labels:** the test split uses only `manual` (the user, via the labeling tool) or `ai_review` labels. `ai_review` means the worker looked at the frame itself; these go in a separate file, and the user's labels always override them.
- **Stable splits:** each 511GA camera's split comes from a salted hash of its ID, so new frames never move a camera between splits.

Installed `torch` 2.14, `torchvision` 0.29, and `open-clip-torch` 3.3.0 (MPS works) and pinned them in `requirements-train.txt`.

## 2026-09-25: Checkpoint 2 (data prep done)

The worker was stopped by accident once; a fresh worker resumed from its code. Details in `docs/phase_reports/phase2_prep.md` and `reports/class_counts.md`.

**What the worker built:**
- Manifest builder, mask rules, CLIP road filter, pHash dedup, group-aware splits, `camera_style` augmentation, the local labeling tool, and 40+ prep tests.
- Label tool: `cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m prep.label_tool` → http://127.0.0.1:8765/

**Whole sources excluded, confirmed by my own look at contact sheets:**
- Flood Area Segmentation: almost all aerial drone shots.
- The Flood Master Italian video: drone footage.
- The Greek video (a fixed elevated camera over a flooded street) is kept, thinned to 62 frames, and used as external test.

**Split changes after my review:**
- FRED locations are assigned whole: dairycreek, holmview, and mountcotton → train; pullenvale → val; cambogan → test.
- NYSDOT cameras are assigned so val and test both get wet and dry frames.
- CLIP is no longer applied to FRED, where it was removing real dashcam frames.

**Two label fixes I made myself** (the dataset is now at `data_version v1-c7dea35e`, with 107 tests passing):
1. **511GA wet labels.** 45 of the 46 frames the AI reviewer labeled "wet" had 0.0 mm of rain in the previous 3 hours by the weather data. They were all night frames. The "specular streaks" the reviewer cited are long-exposure headlight trails on dry pavement. New rule in `prep/sources.py`: an `ai_review` wet/flooded label that conflicts with a `likely_dry` weak label is dropped until the user confirms it in the labeling tool. A regression test covers it.
2. **FRED wet band.** Most of FRED's "wet" frames (water fraction 0.04–0.10 in the road region) show dry near-field pavement with a flooded crossing further ahead, i.e. distant floods, not wet surfaces. Training them as "not flooded" would teach Stage B to miss distant floods, so the bucket is dropped (`WET_BAND_VERIFIED_SOURCES` is now empty).

**Final counts (dry / wet / flooded):**

| Split | Dry | Wet | Flooded |
|---|---|---|---|
| Train | 1,694 | 15 | 783 |
| Val | 491 | 6 | 168 |
| Test | 812 | 10 | 168 |

**Wet but not flooded is now nearly all NYSDOT (30 of 31).** This is the known gap:
- Stage B has only 15 wet negatives to train on.
- The false-alarm rate on wet roads will rest on 10 test frames.
- Wet is effectively a single-source class, which is a shortcut risk for Phase 4 to test.

More wet data will come from 511GA frames captured during real rain, which the collector keeps sampling, and from any labels the user adds.

## 2026-09-25: User labels (during Phase 3)

The user labeled 429 511GA frames in the labeling tool: 350 dry, 78 unusable, 1 wet. 396 of them are on held-out test cameras, so the 511GA test set is now human-labeled.

- **The 45 disputed frames** (AI "wet" vs. weather "dry"): the user marked 42 dry and 3 unusable. This confirms they were night headlight glare, and it validates the conflict rule in `prep/sources.py`.
- **AI "dry" calls:** the user agreed on 171 of 205 and marked 33 as unusable.
- **The forecast is dry through 2026-10-01.** Real wet or flooded Atlanta frames are unlikely this week, so the 511GA test set measures false alarms on dry roads, and wet/flooded performance comes from the external test sets.
- **Rebuild deferred:** the manifest will be rebuilt after Phase 3 finishes and before Phase 4. Training runs on `v1-c7dea35e`, and the user labels mostly touch test cameras.

## Final step requested by the user (do only when everything is done)

1. Strip all comments and docstrings in `flood-ml/` code, and replace them with very short, hackathon-style comments. Ruff and the tests must still pass afterwards.
2. Remove the unnecessary intermediate files used for data prep: spot-check sheets, temporary grids, job logs, and caches. List the files and confirm with the user before deleting anything; don't touch the datasets needed to retrain.

## 2026-09-26: Phase 3 (modeling) done

Details in `docs/phase_reports/phase3_modeling.md`, `reports/runs.csv`, and `reports/onnx_parity.md`.

**Runs:** 6, all on `data_version v1-c7dea35e`.
- **Shipped Stage A:** MobileNetV3Small, fine-tuned (top 30 layers). Val AUC-ROC 0.991, AUC-PR 0.985.
- **Shipped Stage B:** the "mixed" variant (flooded vs. wet+dry, with wet ×8), fine-tuned. Val AUC-PR 0.995.
- EfficientNetB0 (head only) was worse than MobileNetV3Small (AUC-PR 0.975) at twice the cost, so it wasn't pursued.

**Stage B variant.** The as-specified "spec" variant (wet vs. flooded only) has never seen a dry image. With the baseline Stage A, its pipeline flood recall collapsed to 0.125. With the final Stage A, spec and mixed are about equal. I agree with shipping "mixed" for robustness, since swapping needs only a re-export.

**Thresholds:**
- tA = 0.816 (Youden's J).
- tB = 0.897: the lowest value that reaches pipeline precision ≥ 0.90 for "flooded" on all val rows.

**Pipeline val (n = 665):**

| Metric | Value |
|---|---|
| Flooded precision | 0.902 (165/183) |
| Flooded recall | 0.982 (165/168) |
| False flood alarms on dry | 3.7% (18/491) |
| False flood alarms on wet | 0/6 (too small to mean anything) |

**ONNX:** the two outputs `cam` (N,7,7) and `prob` (N,1) are **in that order, so fetch them by name**. I re-checked independently: max |Δprob| vs. Keras is 2.8e-6, and CPU latency is about 1.2 ms per stage (batch 1).

**Grad-CAM:** on flooded frames the heat sits on the water. On dry frames it's often on context (bridges, trees, barriers), which Phase 4 will check.

**Bugs the worker found and fixed** (both regression-tested):
- The EfficientNetB0 `name=` argument broke the weights URL.
- Threshold search accepted "zero predicted positives" as meeting any precision target.

## 2026-09-26: Phase 4 (independent evaluation) started

I rebuilt the manifest with the user's labels, giving `data_version v1-703f0040`. The training-time manifest is saved as `data/processed/manifest_train_v1-c7dea35e.csv` for leakage checks.

- **511GA test set:** 326 rows (324 manual, 2 ai_review), 325 dry and 1 wet.
- **Splits:** no camera moved between train/val and test.

The reviewer subagent (Opus) evaluates the ONNX deliverables on the test split only. It reports three things separately:
- **(a)** held-out 511GA cameras
- **(b)** external test sets
- **(c)** false-alarm rates on every live frame from the held-out cameras during a verified dry period

## 2026-09-26: Checkpoint 3 (independent evaluation done)

Full write-up in `reports/EVALUATION.md`. The reviewer (Opus) tested only on `split == test` of `v1-703f0040`. CIs bootstrap over camera/sequence clusters.

**Headline numbers:**

| Set | Result |
|---|---|
| (a) 511GA held-out cameras (325 dry, human-labeled) | 0/325 false flood alarms. This set can't measure flood recall. |
| (b) Roadway Flooding | flood recall 0.91 (81/89) |
| (b) Greek elevated-camera video | flood recall **0.29** (18/62) |
| (b) FRED cambogan | flood recall 8/17 |
| (b) NYSDOT wet | detected **0/9** |
| All test: flooded | precision 0.964 [0.90, 1.0], recall **0.637** [0.48, 0.93] |
| All test: false alarms | 4/930 on dry |
| (c) Live frames, held-out cameras, dry night | 0/466 flooded |

**Verdicts:**
- **Leakage:** clean.
- **Shortcuts:** present and partly used.
  - Source is predictable at 0.99 balanced accuracy.
  - Darkening an image or adding a 511GA-style text box pushes floods toward "dry".
  - 511GA is 100% dry and 100% night in the data, so the 0/325 may partly be a "looks like 511GA at night, so dry" shortcut.
- **Grad-CAM:** good on Roadway Flooding and FRED floods, near chance on the Greek video.

**Top failure modes:**
1. Missed floods from elevated fixed cameras (the Greek video).
2. Small or distant floods missed until water covers about 20% of the road.
3. Wet pavement is never detected.
4. False floods on non-road views (FRED field shots, 511GA camera 11372).
5. "wet" in practice means a flood score between the thresholds: 33 of 36 test "wet" calls were real floods.

**Decisions carried into Phase 5:**
- Keep the API contract values (`dry`/`wet`/`flooded`), but document `wet` as "water detected, below the flood alert threshold (possible flooding)". Don't claim wet-pavement detection. Temporal smoothing defaults to N=3 consecutive frames, with an optional per-camera blocklist (11372 for the demo).
- Pin inference preprocessing to exactly the training path.
- Thresholds unchanged until new data exists.
- Re-run the live check on daylight frames once the collector has them.

## 2026-09-26: Phase 5 (integration package) done

Details in `README.md`, `docs/INFERENCE_API.md`, and `docs/phase_reports/phase5_integration.md`.

**What was built:**
- `src/inference/`: `load_models`, `predict`, `predict_batch`, `Prediction`, `TemporalSmoother` (N=3 plus a blocklist), and `CameraMoveDetector`. It imports only numpy, PIL, and onnxruntime.
- The CLI (`python -m inference.cli`) and the Gradio demo (`src/demo_app.py`, http://127.0.0.1:7860).
- 56 new tests using tiny fake ONNX models built on the fly.

**Numbers:**
- **Preprocessing** matches training: LANCZOS resize to a 256 short side, a JPEG q95 round-trip, then a 224 center crop. On 100 val rows there were 0 status flips, with max |ΔpA| 0.038. The settings are recorded in the `models/config.json` `preprocess` block.
- **Camera-move threshold 0.35**, calibrated on real 511GA pairs: 4.4% false triggers on the same camera, 90% detection on a different camera.
- **CLI on camera 11372** (the repeat false alarm): with N=3 smoothing it never reaches "flooded", and the blocklist suppresses it entirely.

**CI fix (my miss).** PR #1 CI had been failing since the first code push, and I hadn't checked it. CI installs only `requirements.txt`, so 11 test files that need the training stack (requests, imagehash, cv2, pandas, keras) crashed at collection. Fix: `tests/conftest.py` skips those files when their deps are missing and prints which ones were skipped. With CI's package set simulated locally: 121 passed, 3 skipped. The full local run: 226 passed.

## 2026-09-26: Accuracy experiments (improve_v2): current model kept

Details in `docs/phase_reports/improve_v2.md`; runs in `reports/runs.csv` rows 17–26.

**What was built:**
- Input modes `crop` / `squash` / `letterbox` wired through training, export, and inference. Old configs default to `crop`.
- Night augmentation (`augment._night_style`, P=0.35) and a label-aware overlay rate.
- Multi-crop TTA (`train/tta.py`), a per-candidate evaluator, and a detached training driver.
- 38 new tests.

**Results on val** (tA/tB re-tuned per candidate):

| Candidate | Precision | Recall | Dry FAR | Live false floods (seen cams) |
|---|---|---|---|---|
| Shipped (crop, current) | 0.90 | 0.98 | 3.8% | 3/2073 |
| letterbox224 | 0.97 | 0.96 | 0.8% | 22/2242 |
| letterbox320 | 0.94 | 0.98 | 1.8% | 19/2242 |
| crop + night aug | 0.90 | 0.95 | 3.4% | 3/2236 (plus 33 "wet") |
| Multi-crop TTA, no retrain | 1.00 | 0.96 | 0% | 0/2081 |

TTA made the model *less* robust to darkening on val (23/168 floods flipped to dry vs. 1/168).

**Decision: keep the shipped model.** No candidate beat its val recall, and letterbox raised live night false alarms about 20×. Because the model is unchanged, the Phase 4 test results stand, and no test re-run was spent.

**The honest bottleneck is data.** Val has no elevated-camera floods, so the whole-frame fix can't be judged, and letterbox still sees too few off-center floods in training. Next: more wet data (the Iowa RWIS collection, running) and elevated or street-level flood photos. Then one letterbox retrain, then one test run.

## 2026-09-26: Heatmap display checkpoint

The user requested the heatmap improvements discussed in chat. The classifier
and its thresholds are unchanged; improve_v2 already tested full-frame inputs
and kept the deployed center-crop model.

- Low-evidence default overlays now show the plain frame with an explicit note.
- Display requires Stage A to pass its threshold and joint flood score >= 0.5.
  This is a display policy, not calibrated pixel confidence or a safety claim.
- Raw per-image-scaled attribution is opt-in and remains available for debugging.
- A simple transparent warm palette replaces the rainbow map. Display intensity
  also follows the joint flood score; crop/padding geometry is preserved.
- 33 focused inference tests pass, including low-score, raw-debug, uniform/invalid
  maps, and crop/padding regressions. Existing status/confidence behavior is intact.
- On 674 validation frames, 190 overlays are shown and 484 are suppressed for
  weak evidence. This is presentation behavior, not increased classification accuracy.
- Local comparison images: reports/heatmap_display_samples.png (ignored).

A worker is training a separate water segmentation baseline; another worker is
measuring raw CAM mask overlap and occlusion behavior. Water masks are not road
flooding extent, and their output will be separate from classifier explanations.

## 2026-09-26: Review of outside (ChatGPT/Codex) changes

The user had ChatGPT work in parallel and asked me to validate it. I reviewed every change.

**What it did:**
- **Commit `785fc38`: honest heatmap display.** The overlay shows only when `pA >= tA` and `pA*pB >= 0.5`; otherwise it returns the plain frame with a note. It uses a warm palette, and the raw attribution is opt-in (`raw_heatmap=True`). The classifier, thresholds, and status logic are unchanged, and the new `Prediction` fields are additive.
- **Separate water-segmentation model** (`src/train/water_*`, `src/inference/water.py`, `models/water/`, gitignored):
  - Trained on the FRED and Roadway Flooding train/val masks only. No Water Segmentation or Flood Master data went in, and there was no cross-split leakage.
  - Threshold 0.3, chosen on val, with one test pass: pooled IoU 0.67 (Roadway 0.78, FRED 0.36), and false positives on all 7 empty FRED frames.
  - It's opt-in in the demo and clearly labeled "not flooded-road extent".
  - `water.py` imports only numpy, PIL, and onnxruntime, so it's CI-safe.
- **Docs:** `README` and `INFERENCE_API.md` were updated to match. The `README` also fixed my wrong earlier claim that letterbox "ships".

**Checks:**
- 314 tests pass locally, and ruff is clean.
- The CI simulation gives 167 passed and 5 skipped.
- The smoke test runs `predict` and `predict_water` end to end.
- The shipped classifier ONNX files and `config.json` are untouched.

**My changes:**
- Restored the concrete numbers in the demo's limits text (0.64 recall, 0.29 on the elevated video, wet 0/9), which had been softened to "misses some floods".
- Deleted a stray tool session file, `flood-ml/:memory:.ses`.

**Team note:** `main` now has the backend (PR #4). Its `model_run()` is still a stub. The real `predict()` matches its `model_version` keys exactly. Its adapter will need to map `stage_a_probs` → `stage_a_probabilities`, `stage_b_probs` → `stage_b_probabilities`, and `heatmap_png` → `heatmap_bytes`.

## 2026-09-26: Real wet-road frames from Iowa DOT RWIS cameras

Details in `docs/OTHERCAMS.md` and `docs/phase_reports/othercams.md`.

**Source:** the Iowa Environmental Mesonet archive of Iowa DOT RWIS webcams. It's public domain (I verified the statement on the IEM disclaimer page), needs no key, and has archives back to about 2010. Other options found:
- 511WI, 511NY, and UDOT all need keys and signed agreements, so they weren't registered. That's the user's call.
- Caltrans CCTV is open but live-only.

**Frames:** 340 good daytime frames from 44 cameras, 176 `likely_wet` and 164 `likely_dry`. **41 cameras have both wet and dry frames**, so camera identity can't act as a label. Resolutions are 800×450, 640×480, and 480×270.

**Weak labels are noisy near the threshold.** In the worker's spot check of 8 wet frames, 2 clearly wet, 2 plausible, 2 ambiguous, and 2 looked dry. The labels need the user's review in the labeling tool before training:
```
PYTHONPATH=src ../my_env/bin/python -m prep.label_tool --port 8000 --ga511-root data/othercams/iowa_rwis
```

**Fixes during the run:** the worker added a global pacer and backoff after Open-Meteo 429s. A failed lookup was being cached as "no data" and could silently poison labels.

**Not committed:** a stray `reports/othercams_sample_rows.json` from the outside session. It's added to the final-cleanup list.

**User review of the Iowa frames** (all 340 labeled): 226 dry, 114 wet, 0 unusable.
- Weak label vs. user: `likely_wet` → 108 wet / 68 dry, so it's right 61% of the time. `likely_dry` → 158 dry / 6 wet, right 96% of the time.
- 39 cameras have human-confirmed wet and dry frames.
- Real wet-not-flooded images go from 31 to 145.
- Labels are in `data/othercams/iowa_rwis/labels.csv`, and only these manual labels will be used for training.

## 2026-09-26: Street and elevated flood photos

Details in `docs/phase_reports/flood_photos.md`.

- **eu_flood_2013** (cvjena, Wikimedia Commons, 2013 Central Europe floods): 3,107 flooded and 327 not_flooded images, each with a verified per-image license (mostly CC BY-SA 3.0; also CC BY and CC0). The 275 off-topic "pollution" images have unverified licenses and are excluded. Views are roughly 85–90% ground and 10–15% elevated, with no aerial shots. There are 149 uploader groups. Many flooded images are just rivers over their banks, so the road filter is needed. The not_flooded images include both dry and rain-wet streets.
- **alleyfloodnet** (Lee & Joo 2025): 601 flooded and 509 not_flooded ground-level alley images. It's CC BY 4.0 at the dataset level (I confirmed with the Kaggle API), but some images carry Getty or news watermarks, so rights for individual photos are unverified. Included, with that flag. Weights stay private.
- **floodimg**: skipped. 77% of it duplicates eu_flood_2013, and the rest is mostly aerial.

Since "not_flooded" in these sets can mean either dry or wet, those rows get a new label, `not_flooded`. It's used only as a Stage B negative and excluded from Stage A.

## 2026-09-26: v3 data integration (manifest `v1-7251bbd2`)

Details in `docs/phase_reports/v3_retrain.md`.

**New sources:**
- **iowa_rwis:** 340 rows (226 dry, 114 wet), manual labels only.
- **eu_flood_2013:** 2,130 rows kept (1,894 flooded, 236 not_flooded) after dropping unlicensed rows. The CLIP road filter (road_score > 0) removed 1,304.
- **alleyfloodnet:** 1,110 rows, behind the `INCLUDE_ALLEYFLOODNET` switch.
- No cross-source duplicates.

**New label `not_flooded`:** a Stage B negative only, excluded from Stage A.

**Totals:** 9,337 rows.

| Split | Dry | Wet | Flooded | Not flooded |
|---|---|---|---|---|
| Train | 3,050 | 92 | 2,530 | 521 |
| Val | 826 | 22 | 542 | 112 |
| Test | 957 | 31 | 542 | 112 |

**The legacy rows are untouched,** which I verified myself against my byte-exact backup `manifest_v1-703f0040.csv`:
- Every row of every old source is identical: path, label, and split.
- All 1,825 old 511GA rows keep their split and label.
- No group, camera, or duplicate cluster spans two splits.

The worker found and fixed a real bug on the way: adding new groups was reshuffling the splits of old groups. The new guard `legacy_forced_splits` pins them.

**Checks:** 348 tests pass, and the CI simulation gives 181 passed and 6 skipped.

**Training** (`train.run_campaign_v3`, detached) is running. Night augmentation screened worse on val, so it's off. Candidates are crop224, letterbox224, and letterbox320. `train.select_v3` will then apply the pre-registered rule.

## 2026-09-26: v3 selection. Shipped letterbox320 (old model kept in `models/v1/`)

`train.select_v3` scored the candidates on the new val (n=1,502; wet is n=22, flagged) with the pre-registered rule. Eligible: crop224 and letterbox320. **Winner: letterbox320.** Details in `reports/eval/v3_candidates.json` (local).

| Candidate | Precision | Recall | False floods on dry | on wet | on not_flooded | Live flooded (seen cams) | Dark / label-box flips |
|---|---|---|---|---|---|---|---|
| Old model, as deployed | 0.885 | 0.876 | 39/826 | 7/22 | 16/112 | 27 | 37 / 40 |
| crop224 | 0.900 | 0.963 | 20/826 | 3/22 | 35/112 | 39 | 13 / 14 |
| letterbox224 | 0.900 | 0.948 | 24/826 | 2/22 | 31/112 | 79 | 17 / 4 |
| **letterbox320** | **0.901** | **0.972** | **12/826** | 2/22 | **44/112** | 42 | 23 / 14 |

**Per source (false floods):**
- Iowa dry: 16/32 → **0/32**.
- 511GA labeled dry: 6/593 → 3/593.
- FRED dry: 17 → 9.
- **Worse:** AlleyFloodNet not_flooded 7 → 26 of 83, and EU not_flooded 9 → 18 of 29. These are rain-wet alleys and cleanup scenes.

**Live false floods on seen 511GA cameras** (6,121 frames, 1,370 daytime): 29 → 42 frames (23 → 28 cameras). With 3-frame smoothing that's **1 → 2 alerts**. Both models false-alarm more in daylight.

**Shipped:**
- ONNX parity: max |Δprob| 7.7e-6 and 6.3e-6. Stage B CAM r=0.99999; Stage A CAM r=0.987 (Stage A's CAM is not used for display).
- Latency: about 2.0 / 1.6 ms per stage.
- New thresholds: tA 0.891, tB 0.298. `data_version v1-7251bbd2`.

**Fix I made during shipping.** Inference letterbox didn't match training:
- It skipped the 256 short-side + JPEG step.
- It used PIL's antialiased bilinear, while training used TF's plain bilinear.

It now mirrors training exactly, with a numpy TF-bilinear, the 256/JPEG step, and uint8 truncation. A regression test checks it against TF. Measured max |ΔpB| between inference and training fell from 0.186 to 0.045, with the remainder coming from JPEG decoder differences. Only NYSDOT raw files differ more, because training used header-cropped copies.

**Next:** `src/eval` hardcodes crop-224 preprocessing, so it must be made mode-aware before the one test re-run.
