# Phase 4: independent evaluation

Full write-up: `reports/EVALUATION.md`. The numbers come from `reports/eval_*.json`, `reports/eval_*.csv` and `reports/figures/pr_*.png`. I evaluated the shipped ONNX models with the `config.json` thresholds on `split == "test"` of `v1-703f0040` only, plus live 511GA frames for (c).

## What I did

- **Wrote my own evaluation code** in `src/eval/`:

  | Module | Purpose |
  |---|---|
  | `common` | Preprocessing, ONNX pipeline, status logic, cluster bootstrap |
  | `predict` | Test and live inference, Keras parity, TF-preprocessing skew |
  | `evaluate` | Metrics and PR curves |
  | `gradcam_eval` | CAM energy inside masks, seeded review sheets |
  | `shortcut` | Source probes, cue probes, within-source checks, synthetic perturbations |
  | `slices` | Failure-slice tagging and metrics, eye-check sheets |
  | `leakage` | Leakage checks |
  | `error_gallery` | `reports/errors.html` |
  | `cues`, `viz` | Shared image cues and overlay helpers |

- **Tests:** added 20 in `tests/test_eval_{common,metrics,cues}.py`. They need no data, models or network, and skip when pandas, sklearn or cv2 are missing, since CI installs only `requirements.txt`. Suite: **170 passed**. `ruff check .` is clean.
- **Looked at images myself:**
  - every test error
  - every live frame called flooded or wet
  - 59 seeded Grad-CAM samples
  - the FRED flood-day sequence
  - the leakage pairs
  - an 8-positive / 8-negative sample for every slice tag
- **Packages installed:** none.

## Key numbers

Brackets are 95% cluster-bootstrap CIs. **Flagged** means fewer than 30 examples or fewer than 5 clusters.

| Set | Result |
|---|---|
| **(a) 511GA held-out, 326 rows, 219 cameras** (dry/night only; no flood recall possible) | P(flooded \| dry) **0/325** [0, 0]; P(wet \| dry) 1/325; the single wet frame is called dry (pA 0.0008) **flagged** |
| **(b) Roadway Flooding**, 89 | Flood recall **0.910** (81/89) [0.843, 0.966] |
| **(b) Greek video (FMD)**, 62, 1 cluster | Flood recall **0.290** (18/62) **flagged: 1 cluster** |
| **(b) FRED cambogan**, 615 | Flood recall 0.471 (8/17) **flagged**; precision 8/12; P(flooded \| dry) 4/598, all 4 from one sequence **flagged: 4 sequences** |
| **(b) NYSDOT**, 16, 1 camera | Wet recall **0/9**; P(flooded \| wet) 0/9 **flagged** |
| **All test**, 1,108 | Flooded P **0.964** (107/111) [0.897, 1.000], R **0.637** (107/168) [0.477, 0.934]; wet P 0/36, R 0/10; P(flooded \| dry) **0.004** (4/930) [0, 0.022]; P(flooded \| wet) **0/10 flagged** |
| **(c) Live, test cameras**, 466 frames, 257 cameras, 0 mm rain, 21:01–23:41 EDT | Flooded **0/466**, wet 2/466 |
| **(c) Live, all cameras**, 2,473 frames | Flooded 2 (both from val camera 11372, a defocused grass view), wet 9 (all dry odd views) |
| Parity | ONNX vs. Keras max \|Δprob\| 2.5e-6 / 2.9e-6. PIL vs. TF preprocessing: 6/1,108 status flips |

## Verdicts

- **Leakage: clean.**
  - Current test vs. the training manifest's train and val: 0 shared group, camera or orig_path, and 0 pHash ≤ 6.
  - The rebuild renumbered `ga511_*.jpg` paths, so I compared by `orig_path` and pHash.
  - Minor items:
    - one test camera (14075) serves the same feed as val camera 14066; it's not in the labeled set
    - 11 labeled test rows are on cameras within 30 m of train or val cameras, all predicted dry
    - 5 train/val pHash pairs
- **Shortcuts: present, and at least partly used.**
  - Source is predictable at balanced accuracy 0.99 with CLIP, 0.89 with the Stage A features, and 0.92 with 12 simple cues.
  - Within the one FRED sequence that has both labels, AUC is **0.40** (n = 24, **flagged**). A farmhouse in a field is called flooded at pA 0.99.
  - Synthetic darkening turns 47 more true floods into "dry," and a 511GA-style text box turns 21 more. Both are traits of 511GA, which is 100% dry.
  - The 511GA dry result may therefore be a "looks like 511GA at night, so dry" shortcut. I can't rule that out without wet or flooded 511GA frames.
- **Grad-CAM: good on familiar floods, not on new ones.**
  - Energy on water relative to water area: Roadway 1.5×, FRED 4.6× (peak in water 0.84 and 0.94).
  - The unseen Greek video is at chance (1.06×, peak in water 0.55, below the 0.64 area share).
  - By eye, heat is on water for 20 of 34 true floods, but only 2 of 10 for the Greek video.
  - The Stage A and Stage B CAMs and scores are nearly identical (median per-image CAM correlation 0.98, Spearman(pA, pB) 0.96).
- **The "wet" status doesn't work.**
  - Stage A is effectively a flood detector.
  - "Wet" means "a flood score between the thresholds": 33 of 36 test "wet" calls are floods, and none are wet.
  - The spec-vs-mixed Stage B question can't be settled on test, because Stage A stops all 10 wet frames before Stage B.

## Top failure modes

1. Missed floods seen from an elevated fixed camera (Greek video).
2. Missed distant or small floods (FRED, where water covers 10–19% of the road).
3. No wet-pavement detection (NYSDOT rain, reflections, drops on the lens).
4. False floods on non-road or odd views: the FRED field and farmhouse, and 511GA camera 11372.
5. The "wet" label is misleading.

## Recommendation

**This is OK to show as a demo, but only with these caveats stated:**

- On live Atlanta cameras on a dry night it is quiet: 0 flood calls on unseen cameras and 2 in 2,473 frames overall.
- Flood detection is validated only on external photo and video datasets. Recall runs from 0.29 to 0.91 depending on source, and it has never been tested on a real flooded 511GA frame.
- It does not detect wet roads.
- It hasn't been tested in daylight.
- Darkness and overlay text make it less likely to call a flood.

Don't present the val numbers (0.98 recall, 3.7% false alarms) as the model's performance.

## Decisions for the orchestrator

1. **Relabel "wet"** in the demo and API as "possible flooding / low confidence," or drop it. Don't claim wet-road detection.
2. **Suppress persistent false alarms.** Either require 2 consecutive "flooded" frames per camera, or add a road-view gate (a CLIP "is this a road view" check at 0.5 flags the known false-alarm frames). At minimum, blocklist camera 11372 for the demo.
3. **Thresholds.** tB = 0.897 was calibrated on a val set that is 92% Roadway Flooding. At tB = 0.5, test flood recall would be 0.83 at a cost of 3 more dry false alarms and 8 more live ones (sensitivity only). Any change has to be validated on new data, not on this test set.
4. **Pin the interpolation** in `config.json` (PIL bilinear vs. TF bilinear). The effect is small but real.
5. **Data is the bottleneck.** The most valuable additions are real wet or flooded 511GA frames, including at night, and daytime 511GA frames. For daytime, re-run `eval.predict live` plus `eval.evaluate` once the collector has daylight frames (after about 07:30 EDT). It takes under a minute.
6. **Keep "mixed."** The test can't tell the variants apart.

## Not verified

- Flood or wet recall on real 511GA cameras.
- Daytime false-alarm rate.
- Rain on the lens.
- Moved cameras: the pHash tag was unreliable, with 1 of 8 checked pairs actually moved.
- Night recall beyond 8 Roadway Flooding frames.
- The Grad-CAM categories and slice tags are heuristic or my own judgment, as described in `reports/EVALUATION.md`.
