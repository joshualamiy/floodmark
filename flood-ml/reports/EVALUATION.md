# Phase 4: independent evaluation

The reviewer did not build these models. Everything below comes from my own code in `src/eval/`, run on the shipped ONNX files (`models/stage_a.onnx`, `models/stage_b.onnx`) with the shipped `models/config.json` thresholds (tA = 0.816, tB = 0.897). All metrics use `split == "test"` of `data/processed/manifest.csv` (`v1-703f0040`), except (c), which uses live 511GA frames.

**Notation used in every table**

- `k/n`: numerator over denominator.
- `grp`: number of independent clusters behind a number.
- `[lo, hi]`: 95% cluster-bootstrap CI (2,000 resamples of clusters, not rows). Clusters are: 511GA camera, FRED sequence, NYSDOT camera, Roadway Flooding dup-cluster, and the whole Greek video as one cluster.
- **Flagged** marks any number resting on fewer than 30 examples or fewer than 5 clusters. With one cluster the bootstrap CI collapses to a point, so treat that number as a single observation.

## TL;DR

1. **Real Atlanta cameras don't produce false flood alarms, but they were only ever seen dry and at night.**
   - Held-out 511GA test cameras: **0/325** dry frames called flooded, **1/325** called wet.
   - Every live frame from those cameras in a 0 mm rain window: **0/466** flooded.
   - All 1,376 live cameras: **2/2,473** flooded, both from one odd-view val camera.
   - **Caveat:** every 511GA frame in train, val and test is a dry night frame. The one real wet 511GA frame gets pA = 0.0008. Synthetic darkening and a 511GA-style text box both push true floods toward "dry" (see Shortcuts). This set shows the model stays quiet on Atlanta cameras at night. It says nothing about whether the model would catch a flood on them.
2. **Flood recall is much worse than val suggested:** **0.637 (107/168)** on test vs 0.982 on val.
   - It depends on the source:

     | Source | Recall |
     |---|---|
     | Roadway Flooding (the source that dominates val) | 0.91 |
     | FRED (dashcam) | 0.47 |
     | Greek flood video | **0.29** |

   - The Greek video is the most traffic-camera-like flood data we have: a fixed, elevated camera over a street with half-submerged cars. Its Grad-CAM energy on water is at chance.
3. **"Wet" doesn't work.**
   - Stage A calls **all 10** true wet frames dry.
   - **0 of the 36** test frames given status "wet" are actually wet (33 are floods, 3 dry).
   - Stage A and Stage B are nearly the same model: Spearman(pA, pB) = 0.96, and the median per-image CAM correlation is 0.98.
   - In practice, "wet" means "probably a flood, but low confidence."
4. **Leakage is clean.** There are no shared groups, cameras or orig_paths between test and anything the models saw, and no pHash ≤ 6 matches. The few camera-level near-overlaps are listed below.
5. **Shortcuts exist, and at least one is exercised.**
   - Source is trivially predictable from images: balanced accuracy 0.99 with CLIP, 0.89 with the model's own features.
   - Inside the one FRED sequence that has both labels, the model ranks dry above flooded (AUC **0.40**).
   - It calls a farmhouse in a grass field "flooded" at 0.99.

## Deliverable checks

- **ONNX vs. Keras parity.** I checked 176 test images, stratified by source.
  - Max |Δprob|: Stage A 2.5e-6, Stage B 2.9e-6.
  - I also recomputed the CAM myself as ReLU(Σ w_k·A_k) from the Keras backbone and the Dense kernel. Its max relative difference from the ONNX `cam` output is 2.3e-6.
  - Both run_ids match `config.json`.
  - Outputs come back as (`cam`, `prob`), so my code fetches them by name.
- **Preprocessing skew.** I compared my PIL bilinear pipeline (short side 256, center crop 224) with a replica of the training-time TF pipeline (TF bilinear, uint8 truncation).
  - Max |ΔpA| was 0.069.
  - Statuses flipped on 6 of 1,108 test rows and 2 of 2,473 live frames.
  - Headline numbers use PIL, since that is what a pillow backend will do. The TF numbers are in `eval_metrics_tfpreproc.json`; recall is 109/168 there instead of 107/168.
  - `config.json` should state the interpolation.

## (a) Held-out 511GA cameras: 326 rows, 219 cameras

The labels are 324 manual (the user) and 2 ai_review: 325 dry and 1 wet. **This set can only measure false alarms on dry roads at night.** It can't measure flood or wet recall.

| Metric | Value |
|---|---|
| P(status = flooded \| dry) | **0.000** (0/325, 219 grp) [0.000, 0.000] |
| P(status = wet \| dry) | 0.003 (1/325, 219 grp) [0.000, 0.009] |
| P(status = flooded \| wet) | 0/1 **flagged** |
| Dry precision / recall | 0.997 / 0.997 (324/325) |
| The one wet frame | called dry (pA 0.0008) **flagged** |

3-class confusion (rows are true labels):

| True | Dry | Wet | Flooded |
|---|---|---|---|
| dry (325) | 324 | 1 | 0 |
| wet (1) | 1 | 0 | 0 |

The false "wet" frame is a night view of a gated booth and parking area (pA 0.92, pB 0.86).

## (b) External test sets, per source

| Source | n | grp | Labels | Result |
|---|---|---|---|---|
| roadway_flooding | 89 | 89 | 89 flooded | flood recall **0.910** (81/89) [0.843, 0.966]; 5 → wet, 3 → dry |
| flood_master_test (Greek video) | 62 | **1** | 62 flooded | flood recall **0.290** (18/62) **flagged: 1 group**; 25 → wet, 19 → dry |
| fred (cambogan) | 615 | 4 seq | 598 dry, 17 flooded | flood recall 0.471 (8/17) **flagged**; flood precision 0.667 (8/12) **flagged**; P(flooded \| dry) 0.007 (4/598) [0.000, 0.053] **flagged: 4 seq** |
| nysdot_road_surface | 16 | **1** | 7 dry, 9 wet | wet recall **0.000** (0/9) **flagged**; P(flooded \| wet) 0/9; P(flooded \| dry) 0/7 |

All four FRED flood-day frames come from one sequence, and all 17 FRED flooded frames come from that same sequence. The three dry-day sequences (591 frames) are all correct.

**Pooled external** (782 rows, 95 grp):

| Metric | Value |
|---|---|
| Flooded precision | 0.964 (107/111, 83 grp) [0.897, 1.000] |
| Flooded recall | 0.637 (107/168, 91 grp) [0.477, 0.934] |
| P(flooded \| dry) | 0.007 (4/605) [0.000, 0.058] |
| P(flooded \| wet) | 0/9 **flagged** |
| Missed floods | 28 called dry, 33 called wet |

The recall CI is wide because the Greek video counts as one cluster.

**All test pooled** (1,108 rows, 314 grp): 3-class confusion (rows are true labels).

| True | Dry | Wet | Flooded |
|---|---|---|---|
| dry (930) | 923 | 3 | 4 |
| wet (10) | 10 | 0 | 0 |
| flooded (168) | 28 | 33 | 107 |

Per-class precision and recall:

| Class | Precision | Recall |
|---|---|---|
| Dry | 0.960 (923/961) [0.878, 0.996] | 0.992 (923/930) [0.966, 1.000] |
| Wet | **0.000 (0/36)** [0, 0] | **0.000 (0/10) flagged, 2 grp** |
| Flooded | 0.964 (107/111) [0.897, 1.000] | 0.637 (107/168) [0.477, 0.934] |

False-alarm rates:

| Rate | Value |
|---|---|
| P(status = flooded \| true = wet) | **0/10** **flagged** (n < 30, 2 grp). This is 0 because Stage A calls every wet frame dry, so it says nothing about Stage B. |
| P(status = flooded \| true = dry) | **0.004** (4/930, 224 grp) [0.000, 0.022] |
| P(status ≠ dry \| true = dry) | 0.008 (7/930) [0.000, 0.034] |

Per-stage results (confusion as [[TN, FP], [FN, TP]]):

| Stage | Confusion | Notes |
|---|---|---|
| A (not-dry vs. dry, at tA) | [[923, 7], [38, 140]] | AUC-ROC 0.968, AUC-PR 0.940 |
| B, all rows (flooded vs. not, at tB) | [[935, 5], [61, 107]] | AUC-ROC 0.994 |
| B, on true wet+flooded | [[10, 0], [61, 107]] | |
| B, after the Stage A gate | [[3, 4], [33, 107]] | |
| Pipeline score pA·pB | — | AUC-ROC 0.995, AUC-PR 0.968 |

PR curves: `reports/figures/pr_stage_a.png` and `reports/figures/pr_flooded.png`. The ranking is good (AP 0.97 pooled). The shipped operating point sits at recall 0.64 because tB = 0.897 was calibrated on a val set where 154 of the 168 floods are Roadway Flooding frames that score about 1.0.

**Threshold what-if** (sensitivity only, not a recommendation to tune on test; see `eval_threshold_sensitivity.csv`). Lowering tB to 0.5 at the shipped tA changes these numbers:

| Metric | Shipped (tB = 0.897) | tB = 0.5 |
|---|---|---|
| Flood recall | 0.637 | 0.827 |
| Greek video recall | 0.29 | 0.68 |
| Dry test frames called flooded | 4/930 | 7/930 |
| Live frames called flooded | 2/2,473 | 10/2,473 |

## (c) Live 511GA frames in a verified dry window

Frame set:
- All good frames (empty `dead_reason`) in `data/ga511/frames.csv` up to snapshot ts 1790394061.
- Window: 2026-09-25, 21:01 to 23:41 EDT, entirely after sunset.
- 2,237 frames have `precip_1h_mm` = 0.0 (the max). 236 have no precipitation value.
- These are presumed dry, whether or not they are labeled.

| Camera set | Frames | Cams | Status = flooded | Status = wet |
|---|---|---|---|---|
| **Held-out test cameras** | 466 | 257 | **0.000** (0/466) [0.000, 0.000] | 0.004 (2/466) [0.000, 0.011] |
| Test cameras, unlabeled frames only | 70 | 69 | 0/70 | 0/70 |
| Test cameras, excluding user-"unusable" frames | 394 | 231 | 0/394 | 1/394 |
| Unmapped cameras (in no manifest, so unseen) | 114 | 95 | 0/114 | 1/114 |
| Val cameras (**seen in threshold selection**) | 379 | 204 | 2/379 | 1/379 |
| Train cameras (**seen in training**) | 1,514 | 820 | 0/1,514 | 5/1,514 |
| All cameras | 2,473 | 1,376 | 0.0008 (2/2,473) [0.000, 0.002] | 0.004 (9/2,473) [0.002, 0.006] |

**I looked at every frame called flooded or wet.**

- **Both "flooded" calls come from camera 11372 (a val camera)**, at 21:06 and 23:27 EDT, with pB 0.94 and 0.96.
  - The view is defocused green grass and a guardrail, with no road visible.
  - The same camera's frame was already one of Phase 3's 18 val false alarms. The model will keep flagging this camera on most captures.
- **The 9 "wet" calls are all dry frames**, and most are odd views: a pole close-up, a sidewalk corner with a utility box, a tree canopy, a bridge wall, headlight trails and bright pavement.

This is a **night-only** false-alarm check. Daytime Atlanta frames don't exist in the data yet. Re-running it takes under a minute (see Reproduce).

## Shortcuts

**Source is easy to predict.** I trained a logistic-regression source classifier with 5-fold StratifiedGroupKFold. The CV groups are FRED sequence, NYSDOT camera, 13-frame blocks of the Greek video, 511GA camera and Roadway dup-cluster. Results over all 4,297 manifest rows:

| Features | Balanced accuracy |
|---|---|
| CLIP ViT-B/32 embeddings | **0.989** |
| Stage A's own GAP features | **0.889** (511GA 1.00, Roadway 0.99, FMD 1.00, FRED 0.78, NYSDOT 0.68) |
| 12 simple cues only | **0.92** |

**The dataset is confounded by source.**

| Source | Labels |
|---|---|
| 511GA | 100% dry, 100% night |
| Roadway Flooding | 100% flooded |
| Greek video | 100% flooded |
| NYSDOT | dry and wet |
| FRED | dry and flooded |

The simple cues are resolution, aspect ratio, whether the original is PNG, JPEG quality, brightness, saturation, sharpness, clipping and overlay text. In train, some of them predict the label on their own:

| Cue | AUC for "flooded" (train) |
|---|---|
| Short side (511GA frames are 240–253 px; everything else is 256) | 0.85 |
| Original file is PNG (FRED) | 0.74 |
| Sharpness | 0.72 |
| Brightness | 0.71 |

**Does the model track source?** Two sources contain more than one label, so I checked each.

*FRED, within the one sequence that has both labels* (Cambogan_20250811_113017, 17 flooded and 7 dry):
- pA ranks dry above flooded, AUC **0.40**.
- 6 of 7 dry frames are called not-dry, and 4 of them flooded at pA 0.93–0.99.
- Those "dry" frames are side views of a grass field with a farmhouse and fences, taken the same sunny day. The CAM sits on the house and the fence line.
- Across all of FRED the AUC is 0.987, but that comes from separating sequences and days, not water. All three dry-day sequences (591 frames, all recorded 2025-08-12) are 100% correct.
- **Flagged:** n = 24, 1 sequence.

*NYSDOT, one camera* (7 dry, 9 wet):
- pA AUC for wet vs. dry is 0.68, but no wet frame gets anywhere near tA. The maximum wet pA is 0.67.
- The wet frames are obvious: rain sheen, headlight reflections, drops on the lens, spray.
- **Flagged:** n = 16, 1 camera.

*Across sources* (median pA on true floods):

| Source | Median pA |
|---|---|
| Roadway Flooding (seen in training) | 0.999 |
| FRED | 0.887 |
| Greek video (source never seen in training) | 0.858 |

**Synthetic stress test** (`eval_perturb.json`) on the 168 true floods:

| Edit | Floods called dry (baseline 28) | Greek video called flooded (baseline 18/62) |
|---|---|---|
| Darkening (gamma 2.2, ×0.6) | **75** | **1** |
| 511GA-style black text box at the top | **49** | 6 |
| Grayscale | 37 | — |
| 511GA-like resize + JPEG q60 | 39 | — |

On true dry frames, all four edits changed flooded calls by at most 2 (the baseline is 4/605). This isn't real night data. It does show that two traits that define 511GA frames, darkness and overlay text, push the model toward "dry." That fits a learned "looks like 511GA, so dry" shortcut, and I can't rule it out with the data we have.

## Grad-CAM

**By eye.** I looked at 59 test images (seeded, stratified by source and label; `eval_gradcam_review.csv`) and classified the region where the Stage A heat peaks:

| Region | Count |
|---|---|
| Water | 20 |
| Vegetation / horizon | 9 |
| Road | 7 |
| Border / corner | 7 |
| Vehicle | 6 |
| Structure (fences, poles, sign gantries) | 5 |
| Sky / lights | 3 |
| Overlay text | 2 |

- **On true floods, heat is on the water in 20 of 34:** Roadway 12/12, FRED 6/6, but the Greek video only 2/10. On the Greek video, 8 of 10 sit on car roofs, fences and poles.
- On dry and wet frames, pA is usually below 0.1, so the normalized CAM is mostly noise. Even so, the peak landed on overlay text twice (a 511GA "I-575 S before Booth Rd" label and a NYSDOT header) and on image corners 7 times.
- The Stage B CAM peaked in the same region as Stage A on all 59 images.

**Measured against masks** (`eval_gradcam_energy.json`). "Energy" is the share of CAM energy inside the water mask. The shuffled baseline applies other same-source images' CAMs to this mask, which controls for center bias. "Peak" is the fraction of images whose CAM peak lands in water.

| Source (stage A) | Water area | Energy on water | Shuffled baseline | Peak |
|---|---|---|---|---|
| FRED flooded (17) | 0.06 | 0.30 (4.6× area) | 0.24 | 0.94 |
| Roadway Flooding (89) | 0.44 | 0.66 (1.5×) | 0.56 | 0.84 |
| Greek video (62) | 0.64 | 0.68 (**1.06×**) | 0.67 | **0.55** |

Verdict:
- On Roadway Flooding and FRED the model does look at water, although part of that is a center or horizon bias: the shuffled baseline is already high.
- On the unseen Greek video it doesn't localize water at all. Its peak lands in water less often (0.55) than a random pixel would (0.64).
- Missed floods have much less energy on water than detected ones (Roadway 0.36 vs. 0.67, FRED 0.21 vs. 0.49).

## Leakage (`eval_leakage.json`)

**Between test and everything the models saw** (current test vs. the training manifest's train and val):
- **0** shared `group_id`, `camera_id` or `orig_path`, and **0** pHash pairs within Hamming ≤ 6.
- The same holds between test and train/val inside both manifests.

Note that the rebuild renumbered the `ga511_*.jpg` files, so the same `path` now points to a different frame. I compared the manifests by `orig_path` and pHash instead. No frame changed split.

**Train vs. val** (affects threshold selection, not test):
- 4 shared dup_clusters and 5 pHash pairs, one at Hamming 0, all 511GA.
- These are different camera IDs serving the same feed.

**Greek video vs. public data:**
- 0 pHash matches against any public image in the manifest.
- The FMD index resolves 237 of its train/val images to manifest rows, 100 of them in test (Roadway Flooding). Those FMD rows weren't used for training, so this isn't leakage.

**511GA scene overlap** (live frames):
- Test camera 14075 serves exactly the same feed as val camera 14066 (Hamming 0). 14075 isn't in the labeled test set, but it is in (c).
- 11 test cameras sit within 30 m of a train or val camera: an I-85/I-285 DeKalb pair and a Cobb Pkwy interchange cluster, showing overlapping but different views.
- 11 labeled test rows are on those cameras, all predicted dry.
- This could make (a) slightly optimistic. It can't explain the results.

## Failure slices (`eval_slices.json`; tags checked by eye)

| Slice (how it's tagged) | Result |
|---|---|
| **Night** (every 511GA frame by timestamp; others by top-band brightness) | Dry night frames: 0/329 flooded. Flooded night frames: 8/8 detected, all Roadway, **flagged**. No wet or flooded 511GA night frame exists, so night flood recall on Atlanta cameras is unmeasured. The NYSDOT night wet frames are all called dry. |
| **Glare** (clipped highlights in night frames; brightness alone mislabeled all of FRED as glare, so I restricted it) | 0/192 dry test frames and 0/282 live test-camera frames called flooded. Glare-driven "wet" calls do happen: 1 on test, and some of the 9 live ones. |
| **Blurry / lens haze** (Laplacian variance < 60; checked by eye: defocus, fog, halation) | Live: 2/257 flooded, both the camera-11372 frames, vs. 0/2,211 for sharp frames. Test: 0/16. |
| **Grayscale / IR** | 0/48 test and 0/399 live called flooded. Ungated 511GA monochrome feeds are fine on dry frames. |
| **Overlay text** (511GA only; checked 8/8 positive and 8/8 negative correct) | 0/23 test and 0/247 live flooded. See the stress test for the effect of overlays on floods. |
| **Odd view** (CLIP P("traffic camera view of a road") < 0.5; mixes odd angles with glare-dominated frames) | **Main false-alarm driver.** Dry test frames: 4/19 flooded vs. 0/911 for normal views, **flagged**. Live: 2/138 flooded and 5/138 not-dry, vs. 0/2,330 and 6/2,330. |
| **Moved camera** (pHash distance from the camera's first frame ≥ 22) | **Unreliable:** only 1 of 8 checked pairs had really panned; night pHash follows the headlights. 0/228 flagged frames were called flooded. |
| Rain on the lens | Not measurable. There is no rain in the 511GA data. The one NYSDOT frame with drops on the lens was called dry (true: wet). |

## Top failure modes, with examples

1. **Floods on an elevated fixed camera get missed.** Greek video: 44/62 missed. Cars submerged to the door handles in brown floodwater get pA 0.43–0.80. The CAM sits on car roofs, fences and poles. The water fills the frame, so there's no dry road to contrast it with.
2. **Distant or small floods get missed.** FRED: 9/17 missed. These are frames where the flooded dip is 100+ m ahead and water covers 10–19% of the road. Detection starts at about 20%.
3. **Wet pavement isn't detected at all.** NYSDOT: 9/9 wet frames called dry, including night rain with strong reflections, a gray daytime sheen and drops on the lens. The user-labeled wet 511GA frame gets pA 0.0008.
4. **Non-road views trigger false floods.**
   - FRED: a farmhouse in a field, 4 frames, pA 0.99.
   - 511GA camera 11372: defocused grass, 2 of its 3 captures.
   - The false "wet" calls on live frames: poles, a sidewalk, trees.
5. **The "wet" status is misleading.** It only means "flood score between the thresholds." 33 of the 36 test "wet" calls are real floods, and the rest are dry.

## What I ruled out, and what I couldn't

**Ruled out:**
- ONNX/Keras mismatch (2.5e-6).
- Output-order bugs (outputs fetched by name).
- Wrong run_ids.
- Leakage by group, camera, orig_path or pHash, including against the pre-rebuild training manifest.
- Path renumbering fooling the leakage check.
- Preprocessing skew as an explanation (6/1,108 flips).

**Couldn't rule out:**
- The 511GA dry-specificity result could come from a domain shortcut ("511GA at night means dry") rather than road reading. Nothing in the data can separate the two, because no wet or flooded 511GA frame exists.

**The val numbers were too good because:**
- 92% of val floods are Roadway Flooding, a source the model knows well.
- Wet in val is n = 6.
- Thresholds were tuned on that same val set.

## Limitations

- Wet is n = 10 in test: 9 frames from a single NYSDOT camera plus 1 frame from 511GA.
- 511GA is dry and night only, from a 2h40m window on one night.
- Flood recall on real 511GA cameras is **unmeasured**.
- The Greek video is one cluster, and FRED test floods are one sequence.
- The Grad-CAM categories are my own judgment on 59 images.
- The slice heuristics are approximate; I checked them by eye on 8 positive and 8 negative per tag.
- The perturbation test is synthetic.

## Reproduce (from `flood-ml/`, `PYTHONPATH=src ../my_env/bin/python -m ...`)

```
eval.predict test            # ONNX on test -> reports/eval/test_preds.csv, test_cams.npz
eval.predict live --snapshot-ts 1790394061
eval.predict tf_test / tf_live / parity
eval.evaluate [--tf-preproc] # metrics json + reports/figures/pr_*.png
eval.gradcam_eval energy|sample
eval.shortcut [--skip-embed] [--perturb]
eval.slices [--eye]
eval.leakage
eval.error_gallery           # reports/errors.html (local only)
```

Local-only outputs (gitignored) are `reports/errors.html` and `reports/eval/`. The latter holds per-row predictions, CAMs, contact sheets, the Grad-CAM review sheets and the slice sheets.
