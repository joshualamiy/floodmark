# v3 re-evaluation (2026-09-26)

I didn't build either model. This section compares the new shipped model **v3** (`models/`: MobileNetV3Small, letterbox 320, tA = 0.891, tB = 0.298, data `v1-7251bbd2`) with the old model **v1** (`models/v1/`: crop 224, tA = 0.816, tB = 0.897). This is the one and only test run for v3; it was selected on val alone. Both models are scored on `split == "test"` of the current manifest (1,642 rows), plus live 511GA frames. My original v1 evaluation follows below, unchanged. Where that section says `models/…`, read `models/v1/`. Its JSON files (`eval_metrics.json`, `eval_perturb.json`, …) are the first run's and I didn't overwrite them. This section's outputs are `eval_v3_*.json` and `figures/v3_vs_v1_pr_flooded.png`.

The notation matches the original section: `k/n`, `[lo, hi]` is a 95% cluster-bootstrap CI, and **F** flags any number resting on fewer than 30 rows or fewer than 5 clusters. New clusters: Iowa camera, EU uploader, and AlleyFloodNet pHash dup-cluster. The AlleyFloodNet cluster is nearly one per image, so its CIs are too narrow; see Leakage. "Δ" is v3 minus v1 on the same rows, with a paired cluster bootstrap.

## TL;DR

1. **v3 catches more floods and raises more false floods on street photos that aren't flooded.**

   | Set | v1 recall | v3 recall | Δ v3 − v1 |
   |---|---|---|---|
   | All test | 0.760 (412/542) | 0.891 (483/542) | +0.131 [0.081, 0.194] |
   | Legacy test (same rows as the first run) | 0.637 | 0.750 | +0.113 [−0.022, 0.197] |

   On not_flooded photos, v3 calls **46/112** flooded where v1 called 25/112 (Δ +0.188 [0.095, 0.26]). Flood precision is about the same: 0.912 for v1, 0.893 for v3.
2. **Greek elevated-camera video: 18/62 → 34/62 (0.29 → 0.55), one cluster.** Most of the gain comes from the much lower tB, not from more confident scores.
   - v3's median pB on these frames is 0.385, versus v1's 0.799.
   - At matched false-alarm levels v3 still ranks these floods better than v1.
   - The result is fragile: a 511GA-style title bar drops it to **6/62**, darkening drops it to 4/62, and both together to 0/62.
   - Stage B Grad-CAM on this video still lands on water no better than chance.
3. **Real Atlanta cameras rarely false-alarm, but daytime is several times worse than night** (about 7× on held-out cameras).

   | Set | v1 | v3 |
   |---|---|---|
   | Held-out labeled 511GA (dry, night) | 1/325 flooded | 0/325 flooded |
   | Live test cameras, night | 2/1,094 | 2/1,094 |
   | Live test cameras, day | 6/508 | **7/508** |
   | N=3 smoothed alerts, test cameras | 0 | 0 |
   | N=3 smoothed alerts, all 1,554 cameras | 1 | 2 |

   Each camera was captured only about every 2 hours, so the smoothing numbers aren't what a live poller would see (see (c)).

   As in the first run, 0/325 can't separate road reading from a "looks like 511GA, so dry" cue. The overlay test (Shortcuts) now shows that cue exists in v3.
4. **"Wet" still doesn't work, and v3 has almost no "wet" output.**
   - 1 of 31 true wet frames gets status wet (v1: 5/31).
   - Real wet traffic-camera frames (Iowa + NYSDOT, 30 frames, 7 cameras): v3 calls **23 dry, 6 flooded, 1 wet**.
   - Expect v3 to say either "dry" or "flooded" on a rainy day.
5. **One ordinary dry Iowa highway camera (IDOT-027) is called flooded on all 8 of its test frames**, by both models. Val had shown Iowa dry at 0/32. Iowa test has 6 cameras, so this is one camera's behavior, not a rate.
6. **Leakage doesn't explain the gains.**
   - There are no shared group_id, camera_id or orig_path values between test and v3's train/val. One pHash pair turned up: the same 511GA feed under two camera IDs, affecting 2 dry test rows.
   - CLIP found a few near-duplicate scenes in AlleyFloodNet and EU. With every test row within CLIP cosine 0.85 of a seen image removed, v3 new-source recall is still 0.942 (was 0.955).
7. **Recommendation: demo v3 rather than v1**, with the caveats in the Recommendation section. The most important ones: say nothing about wet roads, and keep N=3 smoothing plus a per-camera blocklist.

## Checks before trusting the numbers

- **Preprocessing is now read from each model's `config.json`.** `src/eval/common.py` imports the deployed `inference.preprocess.preprocess` and reads `mode`, `size` and `jpeg_roundtrip` from each model directory, so eval inputs are the deployed inputs by construction.
  - On 56 test images (stratified by source), my pipeline vs the deployed `predict_batch()`: max |ΔpA| = max |ΔpB| = **0.0**, 0 status flips, for both models.
  - ONNX vs Keras: v3 3.7e-6 / 2.2e-6, v1 2.4e-6 / 3.2e-6 (`eval_v3_parity.json`).
- **v1 reproduces my first run.** On `models/v1`, my old PIL loader gives the first run's 1,108 predictions exactly: max |Δp| 3e-8, 0 flips. So `models/v1` is the model I evaluated before.
  - With the deployed crop path instead, 1 of 1,108 legacy rows changes: a 511GA dry night intersection with a pole in the middle goes from pA 0.809 to 0.850 and is now called **flooded**. Mean |ΔpA| is 0.007.
  - So the first run's "0/325" was one frame optimistic relative to what v1 actually does when deployed.
- **Which file goes in.** Test rows use the manifest's processed copy: the same file training, val and my first run used. I also checked two alternatives:

  | Alternative input | v1 status flips | v3 status flips |
  |---|---|---|
  | Raw originals (NYSDOT kept header-cropped, because its raw header leaks the label) | 3/1,642 | 4/1,642 |
  | Training-time TF pipeline | 31/1,642 | 7/1,642 |

  - The v3 Greek-video result is 34, 34 and 33 across the three paths.
  - 22 of v1's 31 TF flips are small AlleyFloodNet images. Deployed v1 crop mode **zero-pads black borders** onto anything with a short side under 224, while training upscaled it. That's a real v1 train/serve skew; it doesn't affect 511GA frames (short side 240–253).
- **The legacy test rows are identical to the backup manifest `manifest_v1-703f0040.csv`:** 1,108/1,108 rows with the same path, label and group. No row of any source changed split.
- **The live weather was dry.**
  - Open-Meteo gives 0 mm for every frame that has a value (7,865 of 8,149).
  - As an independent check, ASOS METARs at ATL, PDK and FTY show precip 0.00 in all 48 hourly observations from 20:52 EDT 9/25 to 11:53 EDT 9/26, with no weather codes.
- **The selection-time overlay test didn't test what it claims.** The builder's selection "label-box" flip count for letterbox candidates (`train.candidate_eval`) draws the box on the 320 canvas. For a 16:9 frame, the top 70 px of that canvas are gray padding, so the box mostly lands on padding. The "14 label-box flips" for v3 in PROGRESS.md therefore don't measure overlay robustness. My image-space test (Shortcuts) does, and v3 is clearly sensitive.

## v1 vs v3 on test

| Set | n (grp) | Metric | v1 | v3 | Δ [paired CI] |
|---|---|---|---|---|---|
| (a) 511GA held-out cams | 326 (219) | flooded \| dry | 0.003 (1/325) [0.00, 0.01] | **0.000 (0/325)** | −0.003 [−0.009, 0] |
| | | wet \| dry | 1/325 | 0/325 | |
| | | the one wet frame | dry (pA 0.0008) | dry | **F** |
| (b) Roadway Flooding | 89 (89) | flood recall | 0.910 (81/89) [0.84, 0.97] | **0.978 (87/89)** [0.94, 1.00] | +0.067 [0.022, 0.124] |
| (b) Greek video | 62 (**1**) | flood recall | 0.290 (18/62) | **0.548 (34/62)** | +0.258 **F** (1 cluster, no CI) |
| (b) FRED | 615 (4 seq) | flood recall | 0.471 (8/17) | **0.294 (5/17)** | −0.176 **F** (1 seq) |
| | | flood precision | 0.667 (8/12) | 1.000 (5/5) | **F** |
| | | flooded \| dry | 0.007 (4/598) [0, 0.05] | 0.000 (0/598) | **F** (4 seq) |
| (b) NYSDOT | 16 (**1**) | Stage A wet recall | 0/9 | 2/9 | **F** |
| | | flooded \| wet | 0/9 | 1/9 | **F** |
| | | flooded \| dry | 0/7 | 0/7 | **F** |
| (c) Iowa RWIS | 48 (6 cams) | flooded \| dry | 0.185 (5/27) [0, 0.50] | 0.222 (6/27) [0, 0.54] | +0.037 [0, 0.107] **F** |
| | | flooded \| wet | 0.238 (5/21) [0, 0.59] | 0.238 (5/21) [0, 0.59] | 0 **F** |
| | | Stage A wet recall | 0.476 (10/21) [0.10, 0.82] | **0.238 (5/21)** [0, 0.59] | **F** |
| | | status = wet on wet | 5/21 | **0/21** | **F** |
| | | wet \| dry | 5/27 | 1/27 | **F** |
| (c) EU flood 2013 | 319 (19) | flood recall | 0.905 (256/283) [0.84, 0.96] | **0.972 (275/283)** [0.96, 0.99] | +0.067 [0.010, 0.131] |
| | | flood precision | 0.941 (256/272) | 0.908 (275/303) | |
| | | flooded \| not_flooded | 0.444 (16/36) [0.12, 0.50] | **0.778 (28/36)** [0.50, 0.89] | +0.333 [0.25, 0.61] (8 grp) |
| (c) AlleyFloodNet | 167 (161) | flood recall | 0.538 (49/91) [0.43, 0.64] | **0.901 (82/91)** [0.83, 0.96] | +0.363 [0.265, 0.467] |
| | | flood precision | 0.845 (49/58) | 0.820 (82/100) | |
| | | flooded \| not_flooded | 0.118 (9/76) [0.05, 0.20] | **0.237 (18/76)** [0.14, 0.34] | +0.118 [0.027, 0.208] |
| (d) **Legacy test pooled** (= first run's "all test") | 1,108 (314) | flood recall | 0.637 (107/168) [0.48, 0.93] | **0.750 (126/168)** [0.64, 0.99] | +0.113 [−0.022, 0.197] |
| | | flood precision | 0.955 (107/112) [0.89, 1.0] | **0.992 (126/127)** [0.97, 1.0] | |
| | | flooded \| dry | 0.005 (5/930) [0, 0.02] | **0.000 (0/930)** | −0.005 [−0.023, 0] |
| | | flooded \| wet | 0/10 | 1/10 | **F** |
| | | Stage A wet recall | 0/10 | 2/10 | **F** |
| (d) Legacy external only (= first run's "pooled external") | 782 (95) | flood recall | 0.637 | 0.750 | |
| (d) **All test pooled** | 1,642 (500) | flood recall | 0.760 (412/542) [0.62, 0.86] | **0.891 (483/542)** [0.79, 0.97] | **+0.131 [0.081, 0.194]** |
| | | flood precision | 0.912 (412/452) [0.86, 0.96] | 0.893 (483/541) [0.85, 0.94] | |
| | | flooded \| dry | 0.010 (10/957) [0.001, 0.03] | 0.006 (6/957) [0, 0.02] | −0.004 [−0.019, 0.003] |
| | | flooded \| wet | 0.161 (5/31) [0, 0.46] | 0.194 (6/31) [0.03, 0.47] | +0.032 [0, 0.073] (8 grp) |
| | | flooded \| not_flooded | 0.223 (25/112) [0.09, 0.33] | **0.411 (46/112)** [0.22, 0.56] | **+0.188 [0.095, 0.26]** |
| | | Stage A wet recall | 0.323 (10/31) | 0.226 (7/31) | |
| | | status = wet on wet | 0.161 (5/31) | **0.032 (1/31)** | |

The first run reported 0.964 for the pooled legacy v1 precision (107/111). The 0.955 above (107/112) is the extra 511GA flip under the deployed loader.

**Confusion matrices, all test** (rows are the true label, columns the status):

| True | v1: dry | v1: wet | v1: flooded | v3: dry | v3: wet | v3: flooded |
|---|---|---|---|---|---|---|
| dry (957) | 939 | 8 | 10 | 950 | 1 | 6 |
| wet (31) | 21 | 5 | 5 | 24 | 1 | 6 |
| flooded (542) | 72 | 58 | 412 | 43 | 16 | 483 |
| not_flooded (112) | 64 | 23 | 25 | 10 | 56 | 46 |

**Confusion matrices, legacy test** (v1 | v3):
- dry 930: 922/3/5 | 930/0/0
- wet 10: 10/0/0 | 8/1/1
- flooded 168: 28/33/107 | 35/7/126

v3 moves v1's "wet" band into "flooded": 33 → 7 floods called wet. It also calls **more** legacy floods dry: 28 → 35, of which 21 are Greek and 12 FRED.

**Ranking** (AUC of pA·pB for flooded vs everything else):

| Set | v1 | v3 |
|---|---|---|
| All test | 0.968 | 0.985 |
| AlleyFloodNet | 0.759 | 0.932 |
| EU flood | 0.847 | 0.861 |

For wet vs dry (Stage A AUC), v3 improves on NYSDOT (0.64 → 0.76, **F**) and is near chance on Iowa for both models (0.54 → 0.57, **F**).

Figure: `reports/figures/v3_vs_v1_pr_flooded.png` shows legacy and new-source PR curves plus Greek recall vs live false floods.

## What improved, what got worse

**Better (CI excludes 0, or the count is unambiguous):**
- Flood recall pooled over all test: +0.131 [0.081, 0.194].
- AlleyFloodNet recall +0.363 [0.27, 0.47], EU +0.067 [0.01, 0.13], Roadway +0.067 [0.02, 0.12].
- Legacy dry false floods: 5/930 → 0/930. All of FRED's dry frames are now correct.
- The FRED farmhouse shortcut is fixed. Inside the one FRED sequence with both labels, pA's AUC goes from 0.40 to 0.80, and all 7 farmhouse frames are now dry (v1 called 4 flooded).
- Legacy flood precision: 0.955 → 0.992.
- The Greek video, 18 → 34 of 62, but see below.

**Worse:**
- False floods on not_flooded street photos: 25/112 → 46/112, Δ +0.188 [0.095, 0.26].
- FRED flood recall: 8/17 → 5/17 (1 sequence, **F**). These are small, distant floods behind a "ROAD SUBJECT TO FLOODING" sign; v3's pA on the misses is mostly 0.1–0.6.
- Status "wet" has almost vanished: 5/31 → 1/31 on true wet. On not_flooded, 56 of 112 get "wet", which says those scenes look wet, not that wet is detected.
- Stage A wet recall on Iowa: 10/21 → 5/21 (**F**). Every one of those 5 then goes on to "flooded".
- Daytime live false floods are about level, 6 → 7 of 508.

**Unchanged:**
- The Iowa camera IDOT-027: 8/8 flooded under both models.
- Night live false floods: 2/1,094 under both models.

## The Greek elevated-camera video (62 frames, one cluster)

This is still the most 511GA-like flood footage in the data: a fixed, elevated camera over a street with half-submerged cars. **v1 gets 18/62 (0.290); v3 gets 34/62 (0.548).** One video is a single observation, so no CI is possible.

**Where v3's calls go:** 34 flooded, 7 wet, 21 dry. v1 split them 18 / 25 / 19. v3 calls slightly *more* of these frames dry than v1 did.

**The gain is mostly the threshold:**

| | v1 | v3 |
|---|---|---|
| Median pA | 0.861 | 0.924 |
| Median pB | 0.799 | **0.385** |

tB fell from 0.897 to 0.298. At v3's own tA, the what-if (sensitivity only) gives Greek recall of 0.37 at tB = 0.5 and 0.15 at tB = 0.7.

**At matched false-alarm levels v3 is still better.** Sweeping one threshold on pA·pB:

| Matched on | v1 Greek recall | v3 Greek recall |
|---|---|---|
| 9 live test-camera false floods | 14/62 | 27/62 |
| 58 false floods among the 1,100 non-flooded test rows | 26/62 | 37/62 |

So v3 ranks these floods better. It just isn't confident about them.

**It's fragile to 511GA-like edits** (Shortcuts section): a black title bar at the top of the frame gives 9/62, the full 511GA overlay 6/62, darkening 4/62, and overlay plus darkening 0/62. The same black bar moved to the bottom of the frame costs almost nothing (31/62). This is the strongest evidence of a learned "looks like 511GA, so dry" cue that I've found. Every 511GA training frame is dry, carries that title bar, and v1's crop only partly showed it, while letterbox shows it in full.

**Grad-CAM** (v3 Stage B, the CAM the demo displays):
- Energy on water is 0.648, versus 0.647 for other frames' CAMs applied to the same mask.
- The CAM peak lands in water 0.47 of the time, while water covers 0.66 of the image area.
- By eye (12 frames), the heat sits on the utility pole, car roofs and edges, the fence, and the buildings at the top. Detected and missed frames are nearly identical scenes, with pB swinging between 0.14 and 0.79.

## (c) Live 511GA frames, day vs night

**Frame set:**
- Every good frame (empty `dead_reason`) up to snapshot ts 1790438064.
- Window: 21:01 EDT 9/25 to 11:54 EDT 9/26, 8,149 frames from 1,554 cameras.
- Day is 07:30–19:30 EDT, but the day frames only cover **07:30–11:54**. There's no afternoon, dusk or evening daylight.
- The weather was verified dry (see Checks), so every "flooded" here is a false alarm.

| Camera set | Time | Frames | Cams | v1 flooded | v3 flooded | v1 wet | v3 wet |
|---|---|---|---|---|---|---|---|
| **Held-out test cams** | night | 1,094 | 296 | 2 (0.0018) [0, 0.006] | 2 (0.0018) [0, 0.005] | 5 | 0 |
| | **day** | 508 | 270 | 6 (0.012) [0.002, 0.024] | **7 (0.014)** [0.004, 0.027] | 33 | 1 |
| | all | 1,602 | 296 | 8 | 9 | 38 | 1 |
| Test cams, excluding user-"unusable" frames | night | 1,020 | 296 | 2 | **0** | 4 | 0 |
| Unmapped cams (unseen) | day / night | 118 / 76 | 83 | 3 / 1 | 1 / 0 | 8 / 1 | 1 / 0 |
| Val cams (seen in selection) | day / night | 410 / 882 | 234 | 8 / 10 | 6 / 5 | 26 / 4 | 3 / 0 |
| Train cams (their night frames were in training) | day / night | 1,604 / 3,457 | 941 | 26 / 6 | **35** / 2 | 95 / 12 | 6 / 1 |
| All cams | day / night | 2,640 / 5,509 | 1,554 | 43 / 19 | 49 / 9 | 162 / 22 | 11 / 1 |

- **The daytime false-flood rate is several times the night rate for both models.**
  - Held-out test cameras: v3 1.4% day vs 0.18% night (7.5×); v1 1.2% vs 0.18% (6.5×).
  - All cameras: v3 1.9% vs 0.16% (11×); v1 1.6% vs 0.34% (5×).
- 511GA training data is 100% night, so daylight is a domain gap for both models. v3 makes it slightly worse on train-camera daytime frames (35 vs 26).
- v1's "wet" status fires on 6.1% of daytime frames. v3's almost never fires.

**Temporal smoothing** (the deployed `TemporalSmoother`, N = 3, replayed per camera in time order):

| Camera set | v1 alerts | v3 alerts |
|---|---|---|
| Test cameras | **0** | **0** |
| All cameras | 1 (val camera 11372, the known defocused-grass view, 6/8 frames flooded) | 2 |

v3's two alerts:
- Val camera **17397**, GDOT-0233 I-285 CW at I-85: a close-up bridge pier filling half the frame. 6/7 frames are flooded, day and night.
- Train camera **14187**, I-75 S past Woodstock Rd: a hazy morning view, 3 consecutive frames.

On the test cameras, v3 has 8 cameras with at least one flooded frame and none with 3 or more. v1 has 5, and camera 15139 has 3 flooded frames, though never 3 in a row.

**Caveat:** the collector only captured each camera about every 2 hours (median gap 119 min, IQR 88–219) and got about 1.9 daytime frames per test camera. "3 consecutive frames" here means roughly 6 hours of persistence. That's far stricter than a live poller running every few minutes, where a camera whose view persistently looks flooded (13750 below, or 17397) would alert within minutes. **These smoothed-alert counts are a lower bound.**

**I looked at every live frame v3 calls flooded:** 9 on test cameras, 1 on an unmapped camera and 48 on train/val cameras (`reports/eval/v3/sheets/live_flooded_*.jpg`, `live_test_flooded_big.jpg`). None shows water on the road. The test-camera frames:

| Camera | Time (EDT) | pB | What's in the frame |
|---|---|---|---|
| 11226 | 07:41 | 0.85 | SR 10 Freedom Pkwy @ Boulevard, just after sunrise. A smeared, hazy lens over the left third, a red signal, a building and trees. No road surface visible. |
| 13539 | 21:03, night | 0.74 | IR grayscale close-up of an "EXPRESS LANES CLOSED" sign and dark foliage, with no road. The user labeled this frame unusable. |
| 13750 | 09:47 and 11:19 | 0.73, 0.92 | United Ave E at Walker St: a gated guard booth and driveway on sunlit asphalt with hard tree shadows. **Both of this camera's daytime frames are flagged**, by both models. This is the same booth scene the first run's false "wet" came from. |
| 14188 | 22:07, night | 0.75 | IR grayscale, I-75 S before Woodstock Rd: dark tree canopy over a road with headlight streaks. The user labeled this frame unusable. |
| 15124 | 07:51 | 0.31 | Cobb Pkwy at I-285: sunrise glare over a highway and ramps. pB is barely above tB. |
| 15205 | 11:37 | 0.34 | Johnson Ferry Rd intersection: close-up signal heads and a pole, cracked, sealed asphalt with shadows. |
| 15436 | 07:54 | 0.57 | Pleasant Hill @ Satellite: a wide intersection just after sunrise, long shadows. |
| 17774 | 11:50 | 0.59 | I-675 N: a highway beside a grass embankment, with a camera housing in the foreground. |
| 15420 (unmapped) | 09:53 | 0.59 | Old Norcross Rd: a road in dappled shade with parked trucks. |

**Patterns:**
- Daytime false floods are sunlit pavement with hard shadows, low sun or glare at 07:40–07:55, haze or smears on the lens, and foreground clutter.
- Night false floods are IR grayscale views with no road.
- On train and val cameras, the same patterns show up plus bright, washed-out highways and the bridge pier (17397).

## The not_flooded trade-off

Val had already shown this (44/112 not_flooded called flooded). Test confirms it: **46/112 (0.411 [0.22, 0.56]) for v3 vs 25/112 (0.223 [0.09, 0.33]) for v1**, Δ +0.188 [0.095, 0.26].

| Source | v1 flooded | v3 flooded | What those rows are |
|---|---|---|---|
| EU flood 2013 not_flooded | 16/36 | **28/36** (8 uploaders) | Flood aftermath (mud, debris, sandbags, fire crews, a water-level marker plaque) and also plainly dry streets: a crosswalk, a church street, a townhouse row. All get pA 0.97–1.00. |
| AlleyFloodNet not_flooded | 9/76 | **18/76** | Rain-wet alleys: puddles, reflective paving, night streets after rain. A few arguably have standing water. |

**Why it happens:**
- v3's Stage A calls **97%** of EU not_flooded and **88%** of AlleyFloodNet not_flooded "not dry" (median pA ≥ 0.995). Stage A never trained on those rows, since they're excluded from it, so it treats a street photo from those sources as wet regardless of the pavement.
- Stage B then decides, and at tB = 0.298 it says "flooded" often.
- Within EU, ranking barely improved (AUC 0.847 → 0.861) while both recall and false floods rose. That means v3 shifted the whole EU source toward "flooded", which is a source prior, not better discrimination. AlleyFloodNet is different: its AUC rose from 0.76 to 0.93, so part of that gain is real.

**What this means for Atlanta:** rain-wet pavement is what a 511GA camera sees on an ordinary rainy day. The only real wet traffic-camera frames we have (Iowa and NYSDOT, 30 frames on 7 cameras) went to **flooded in 6 cases** (Iowa 5/21, NYSDOT 1/9), wet in 1, and dry in 23. **The first rain in Atlanta will probably produce false flood alerts.** This can't be measured on 511GA yet, because no wet 511GA frames exist.

## Shortcuts (`eval_v3_perturb.json`)

These edits are applied to the full frame *before* each model's own deployed preprocessing, so a 511GA-style overlay lands on image content. I made the overlay match real 511GA frames: a black title band across 7% of the height and 70% of the width at top left, with white text-like marks, plus a white logo box at top right. Darkening is gamma 2.2 × 0.6.

**True floods (542).** Cells are frames still called flooded, then frames called dry in parentheses.

| Edit | v1 | v3 | Greek video: v1 | Greek video: v3 |
|---|---|---|---|---|
| none | 412 (72 dry) | 483 (43 dry) | 18 | 34 |
| darken | 390 (115) | 433 (99) | 3 | **4** |
| 511GA overlay | 411 (77) | 457 (81) | 14 | **6** |
| overlay + darken | 387 (116) | 429 (105) | 1 | **0** |
| black band, top only | 410 (78) | 446 (88) | 14 | 9 |
| same band at the bottom (occlusion control) | 386 (94) | 464 (61) | 10 | 31 |
| gray band, top | 407 (79) | 464 (63) | 11 | 16 |
| logo box only | 413 (72) | 482 (47) | 18 | 31 |

**Reading the table:**
- **v3 is more sensitive to the 511GA title bar than v1.** The overlay turns 38 more floods dry for v3; for v1 it's 5. v1's center crop cuts most of the top band away, while letterbox keeps all of it.
- A black bar at the top hurts much more than the same bar at the bottom (Greek video 9 vs 31). So it's the position and look of the 511GA title bar, not just the pixels it covers.
- A gray bar at the top costs less (16), so part of the effect is losing the top of the frame, and part is the black title-bar look.
- Other sources barely move under the overlay (Roadway 87 → 88, EU 275 → 275). The damage concentrates on the one set that looks like a traffic camera, and those are the frames that matter.
- **Darkening** takes 56 more floods to dry for v3 and 43 for v1, and the Greek video drops to 4/62. Night floods on Atlanta cameras will probably be missed by both models.
- **The edits don't create false floods.** On true dry non-511GA frames (632), darkening and the overlay change v3's flooded count from 6 to 5.

**Aspect ratio.** Letterbox bars encode the source's aspect ratio, and in train a 4:3 frame is 79% likely to be flooded while a 16:9 frame is 22% likely. I tested whether v3 uses the bars:

| Test | Crop to the other aspect | Control: crop the same area, same aspect |
|---|---|---|
| Non-16:9 floods (305) cropped to 16:9 | 297 → 290 flooded | 297 → 299 |
| 511GA dry (325) cropped to 4:3 | 0 → 3 flooded | 0 → 3 |

Aspect is at most a weak cue for v3. I found no evidence it's being exploited.

## Grad-CAM (`eval_v3_gradcam_energy.json`; letterbox geometry, masks mapped into the canvas)

"Energy" is the share of CAM energy inside the water mask; "shuffled" applies other same-source images' CAMs to the same mask. The "Water, of content" column is the chance level for "Peak".

| Source (v3 Stage B) | Water, of content | Energy on water | Shuffled | Peak on water | v1 energy / shuffled / peak |
|---|---|---|---|---|---|
| Roadway Flooding (89) | 0.42 | 0.69 | 0.54 | 0.82 | 0.66 / 0.56 / 0.84 |
| FRED flooded (17) | 0.05 | 0.20 | 0.13 | 0.82 | 0.31 / 0.25 / 0.94 |
| **Greek video (62)** | 0.66 | **0.65** | **0.65** | **0.47** | 0.68 / 0.67 / 0.60 |

- The gray padding gets only 1–6% of the CAM energy (median), so v3 isn't looking at the bars.
- On Roadway and FRED, v3's CAM sits on water well above the shuffled baseline. Missed floods have much less energy on water than detected ones: Roadway 0.31 vs 0.69, FRED 0.17 vs 0.33.
- On the Greek video, v3 is **no better than v1: at chance.**

**By eye** (`reports/eval/v3/sheets/`):
- **Greek video:** heat on the pole, car roofs, fences and buildings.
- **EU not_flooded false floods:** heat on buildings, a water-level plaque, sandbags and fences, rarely on pavement.
- **AlleyFloodNet not_flooded false floods:** heat on the wet, reflective paving, which is at least the right region.
- **Iowa IDOT-027:** heat on the tree line behind the highway and on a truck.
- **Live false floods:** heat on the smeared lens, the signal heads, the pier, and sunlit asphalt.

## Leakage (`eval_v3_leakage.json`)

The current manifest is v3's training manifest. `data_version` is v1-7251bbd2, the sha256 prefix matches, and the campaign log shows n_train of 5,672 for Stage A and 6,193 for Stage B.

**New-source test rows (534) vs v3 train/val (7,695):** 0 shared group_id, camera_id, orig_path or dup_cluster, and 0 pHash pairs within Hamming ≤ 6.

**Legacy test rows (1,108) vs v3 train/val:**
- 0 shared group, camera or orig_path.
- **1 pHash pair, which is new since the first run.** Test camera 17408 and a new train camera 13479 serve the same feed (GDOT-0331, I-20 W past SR 280). This affects 2 dry test rows, both predicted dry. It can't inflate anything that matters.
- On live frames, 28 test-camera frames have a pHash-≤ 6 twin on a train or val camera. That's the same "one feed, two IDs" problem as in the first run.

**CLIP near-duplicates** (ViT-B/32, cosine between every test row and every train/val row). This catches crops and re-framings that pHash misses.
- The 19 test rows at ≥ 0.95 are all FRED ↔ FRED: dashcam frames on similar rural roads, the same for both models.
- For the new sources, the top pairs are:

  | Source | Top cosine | What it is |
  |---|---|---|
  | AlleyFloodNet | 0.945 | Two frames of the same flooded night street, one test and one train |
  | AlleyFloodNet | 0.934 | A near-identical dry alley photo |
  | EU | 0.918–0.928 | The same Prague riverside or building, shot by different uploaders |

- Grouping by uploader (EU) and by pHash cluster (AlleyFloodNet) doesn't stop the same scene from appearing in both train and test.
- **Impact:** dropping every EU/AlleyFloodNet test row with cosine ≥ 0.90 (28 of 486 rows) leaves v3 recall at 0.957 and v1 at 0.815. At ≥ 0.85 (155 of 486 rows dropped), v3 is at 0.942 and v1 at 0.759, and the not_flooded false-flood rates barely move. **The v3 gains aren't explained by leakage.**

## Updated top failure modes

1. **Rain-wet pavement is called flooded, or not seen at all.**
   - There's effectively no "wet" output (1/31).
   - On real wet traffic-camera frames: 23 dry, 6 flooded, 1 wet.
   - On rain-wet street photos: 18/76 flooded.
   - This is the main risk on the first rainy day in Atlanta, and it's unmeasured on 511GA.
2. **Floods on elevated fixed cameras are fragile.**
   - The Greek video is at 0.55, but only because of the low tB.
   - A 511GA title bar or darkness removes almost all of it (6/62, 4/62).
   - The CAM on water is at chance.
3. **Whole-camera false alarms.** Some views are called flooded on nearly every frame:
   - Iowa IDOT-027 (a dry highway side view): 8/8.
   - 511GA 17397 (a bridge pier): 6/7.
   - 511GA 13750 (a gated booth in sun and shadow): both daytime frames.

   These need a per-camera blocklist or calibration, not a better threshold.
4. **Daytime false floods run 6–11 times the night rate.** Causes are low sun, hard shadows, lens smears and haze. 511GA training data has no daytime frames.
5. **Street photos from flood events are called flooded whether or not they show flooding.** EU not_flooded: 28/36, including plainly dry streets. That's a source prior, not road reading.
6. **Small or distant floods are missed.** FRED recall fell from 8/17 to 5/17, with floods 100+ m ahead given pA mostly 0.1–0.6.

## Recommendation

**Demo v3, not v1.**
- Pooled over all test, v3 catches more floods (+0.131 [0.081, 0.194]).
- v1's farmhouse and dry false floods are gone on the legacy sets (5/930 → 0/930).
- At matched false-alarm levels v3 ranks floods better, including on the Greek video.
- Legacy flood precision rises (0.955 → 0.992).
- Live false alarms on held-out Atlanta cameras are no worse: 9 vs 8 of 1,602, with 0 smoothed alerts for either.

**Caveats to state in the demo and README:**
1. **Don't claim wet-road detection.** Say that v3 almost never outputs "wet", and that on rain-wet pavement it can say "flooded". Expect false flood alarms in the first rain. Nothing has been measured on wet 511GA frames.
2. **Keep N=3 smoothing on.** At a real polling rate, also add 17397 and 13750 to the blocklist (along with 11372 for v1 history) and watch for new persistent cameras. The ~2-hourly collector can't show how often smoothing would fire at a 1–5 minute poll.
3. **Present the elevated-camera flood demo honestly.** The 0.55 on the one Greek video is one clip, and it depends on tB = 0.298. The same frames with a 511GA title bar drop to 6/62.
4. **Daytime:** the live false-alarm rate is about 1.4% of daytime frames on held-out cameras, versus 0.2% at night.
5. **The Grad-CAM overlay** is a hint about where the score came from, not a water map. On the elevated video it's at chance.

**If there's time for one more data step:** collect real *wet* and daytime 511GA frames (the first rain), plus any flooded 511GA frames. Then add daytime 511GA frames and 511GA-overlay-style augmentation on flood images to training. That targets failure modes 1, 2 and 4 directly. Don't re-tune tB on this test set.

## Unverified or not measurable

- **Flood recall on real 511GA cameras** is still unmeasured: no flooded 511GA frame exists.
- **Wet 511GA frames** are limited to the same single wet test frame.
- **Afternoon, dusk and evening daylight:** there are no frames from 12:00–21:00.
- **Live smoothing at a real polling interval.**
- **The Greek video is one cluster.** Its 0.29 → 0.55 has no CI and shouldn't be read as a rate for elevated cameras.
- **Small, few-cluster sets:** Iowa test has 6 cameras, NYSDOT 1, and FRED floods 1 sequence (**F** throughout).
- **Clustering on the new sources:** the AlleyFloodNet bootstrap clusters are nearly per-image, and EU groups by uploader but same-scene photos cross uploaders. Both make those CIs too narrow.
- **The image-space overlay is my replica of the 511GA title bar,** not a real 511GA overlay on real floods.
- **Day/night** is a clock rule (07:30–19:30 EDT). I didn't use sun angle, and I didn't check dawn frames by eye.

## Reproduce (from `flood-ml/`, `PYTHONPATH=src ../my_env/bin/python -m ...`)

```
eval.predict test --tag v1|v3        # -> reports/eval/<tag>/test_preds.csv, test_cams.npz
eval.predict raw  --tag v1|v3        # raw-original sensitivity
eval.predict tf   --tag v1|v3        # training tf-pipeline sensitivity
eval.predict live --tag v1|v3 --snapshot-ts 1790438064
eval.predict parity                  # deployed predict() parity, onnx/keras, v1 reproduction -> eval_v3_parity.json
eval.evaluate                        # -> eval_v3_metrics.json, figures/v3_vs_v1_pr_flooded.png
eval.stress                          # -> eval_v3_perturb.json
eval.gradcam_eval energy --tag v3    # -> eval_v3_gradcam_energy.json (and --tag v1)
eval.leakage --rerun                 # -> eval_v3_leakage.json (CLIP embeddings cached locally)
eval.error_gallery --tag v3          # -> reports/errors.html (local only; v1's first-run gallery kept at reports/eval/errors_v1_firstrun.html)
```

The first run's per-row files are still in `reports/eval/` (top level). The re-run's are in `reports/eval/v1/` and `reports/eval/v3/`, along with the review sheets in `reports/eval/v3/sheets/`. All of it is gitignored.

---

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
