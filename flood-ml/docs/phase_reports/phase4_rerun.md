# Phase 4 re-run: v3 vs v1 on test

Full write-up: the top section of `reports/EVALUATION.md`, "v3 re-evaluation (2026-09-26)". The original v1 section below it is unchanged. Numbers come from `reports/eval_v3_{metrics,parity,perturb,gradcam_energy,leakage}.json` and `reports/figures/v3_vs_v1_pr_flooded.png`. This is the single test run for v3, which was selected on val only.

## What I did

- **Made `src/eval` model-agnostic.** `common.preprocess` now *imports* the deployed `inference.preprocess.preprocess`, and `Pipeline(model_dir)` reads `mode`, `size` and `jpeg_roundtrip` from that directory's `config.json`.
  - Eval inputs vs the deployed `predict_batch()` on 56 test images: max |Δp| **0.0** for both models.
  - `models/v1` with my first-run loader reproduces the first run exactly (3e-8). With the deployed crop path, 1 of 1,108 legacy rows flips: a 511GA dry night frame goes to flooded.
- **New and changed modules:**

  | Module | Change |
  |---|---|
  | `stress` (new) | Image-space perturbations, applied before each model's own preprocessing |
  | `evaluate` | Rewritten: not_flooded metrics, paired v3 − v1 cluster-bootstrap deltas, live day/night split, a replay of the deployed `TemporalSmoother` |
  | `gradcam_eval` | Letterbox mask mapping and padding energy |
  | `leakage --rerun` | Per-source checks plus CLIP near-duplicates |
  | `error_gallery --tag v3` | v3 errors and live test-camera flooded frames |

- **The first-run functions still run and are pinned to v1:** `shortcut.perturb` and `shortcut.embed_stage_a`. `shortcut._perturb`, which `train.candidate_eval` imports, is unchanged.
- **Tests:** 14 new in `tests/test_eval_rerun.py`. Full suite **366 passed**, and `ruff check .` is clean.
- **Looked at by eye:**
  - every live frame v3 calls flooded (58)
  - all 48 Iowa test frames
  - all 46 not_flooded false floods
  - the misses and the wet frames
  - 12 Greek-video frames with CAMs
  - the top CLIP near-duplicate pairs
- **Checked the weather** against ASOS at ATL, PDK and FTY: 0.00 in over 48 hourly observations.

## Key numbers

Cells are v1 → v3. Brackets are 95% cluster-bootstrap CIs; Δ is a paired v3 − v1 bootstrap. **F** means fewer than 30 rows or fewer than 5 clusters.

| Set | v1 → v3 |
|---|---|
| (a) 511GA held-out, labeled dry (325, 219 cams) | flooded 1/325 → **0/325** |
| (b) Roadway Flooding (89) | recall 0.910 → **0.978**, Δ +0.067 [0.022, 0.124] |
| (b) **Greek elevated video** (62, 1 cluster) | recall **0.290 (18/62) → 0.548 (34/62)** **F** |
| (b) FRED | recall 8/17 → **5/17** **F**; dry false floods 4/598 → 0/598 |
| (b) NYSDOT wet (9) | Stage A 0/9 → 2/9; flooded 0 → 1 **F** |
| (c) Iowa RWIS (48, 6 cams) | false floods on dry 5/27 → 6/27; on wet 5/21 → 5/21; Stage A wet recall 10/21 → 5/21 **F** |
| (c) EU flood 2013 | recall 0.905 → **0.972** [0.96, 0.99]; not_flooded → flooded 16/36 → **28/36** |
| (c) AlleyFloodNet | recall 0.538 → **0.901**, Δ +0.363 [0.27, 0.47]; not_flooded → flooded 9/76 → **18/76** |
| (d) Legacy pooled (1,108; same rows as the first run) | recall 0.637 → **0.750**, Δ +0.113 [−0.022, 0.197]; precision 0.955 → 0.992; dry false floods 5/930 → 0/930 |
| (d) All test (1,642) | recall 0.760 → **0.891**, Δ **+0.131 [0.081, 0.194]**; precision 0.912 → 0.893; not_flooded → flooded 0.223 → **0.411**, Δ +0.188 [0.095, 0.26]; status wet on true wet 5/31 → **1/31** |
| (c) Live, held-out test cams, **night** (1,094) | flooded 2 → 2 |
| (c) Live, held-out test cams, **day** (508, 07:30–11:54 only) | flooded 6 → **7**; wet 33 → 1 |
| N=3 smoothed alerts, test cams / all 1,554 cams | 0 / 1 → **0 / 2** |

## Findings that matter

1. **The Greek-video gain comes mostly from tB = 0.298.**
   - v3's median pB on those frames is 0.385; v1's is 0.799.
   - At matched false alarms, v3 still ranks them better (27/62 vs 14/62 at 9 live false floods).
   - The gain is fragile. A 511GA-style black title bar at the top of the frame gives **6/62** (9/62 for the bar alone), while the same bar at the bottom gives 31/62. Darkening gives 4/62, and overlay plus darkening gives 0/62.
   - Grad-CAM on water is at chance: energy 0.65 vs 0.65 shuffled, and the peak lands in water 0.47 of the time against a water share of 0.66.
2. **The selection-time "label-box" robustness number for letterbox candidates was measured on padding,** because the box lands in the gray bars of 16:9 frames. It says nothing about overlay robustness.
3. **There's no usable "wet" output.**
   - On real wet traffic-camera frames: 23 dry, 6 flooded, 1 wet.
   - On rain-wet alleys: 18/76 flooded.
   - Expect false flood alerts in the first Atlanta rain. That's unmeasurable today because no wet 511GA frames exist.
4. **Some cameras are called flooded on nearly every frame:**
   - Iowa IDOT-027, a dry highway: 8/8, under both models.
   - 511GA 17397, a bridge pier: 6/7.
   - 511GA 13750, a gated booth in sun and shadow: both daytime frames. This is also the camera behind the first run's false "wet".
5. **EU not_flooded photos get pA 0.97–1.00, even plainly dry streets.** Within-EU ranking barely improved (AUC 0.847 → 0.861). That's a source prior.
6. **Leakage:**
   - 0 shared group, camera or orig_path; 1 pHash pair (the same 511GA feed under two IDs, affecting 2 dry test rows).
   - CLIP found a few same-scene pairs in AlleyFloodNet and EU.
   - Dropping every test row within CLIP cosine 0.85 of a seen image leaves v3 new-source recall at 0.942. Leakage doesn't explain the gains.
7. **Preprocessing checks.**
   - Raw-vs-processed inputs flip 3 (v1) and 4 (v3) statuses of 1,642. Against the training TF pipeline, v1 flips 31 and v3 flips 7.
   - Deployed v1 crop mode zero-pads images with a short side under 224; training upscaled them. This is a v1 train/serve skew.

## Recommendation

**Demo v3.** Caveats:
- Don't claim wet detection, and warn that rain-wet roads can read as flooded.
- Keep N=3 smoothing and blocklist 17397 and 13750.
- Present the elevated-camera flood recall as one clip that depends on the low tB and breaks under a 511GA overlay.
- Daytime false floods are about 1.4% of frames on held-out cameras.

Next data step: real wet, daytime and flooded 511GA frames, and 511GA-overlay augmentation on flood images. Don't tune on this test set.

## Unverified

- Flood recall on real 511GA cameras.
- Wet 511GA frames.
- Afternoon and evening daylight.
- Smoothing at a real polling interval (the collector sampled each camera about every 2 h).
- CIs for the Greek video (1 cluster), FRED floods (1 sequence), NYSDOT (1 camera) and Iowa (6 cameras) are **F**. AlleyFloodNet and EU clusters are too fine, so their CIs are too narrow.
