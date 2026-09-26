# Phase 4: independent frozen water-overlay review

Reviewed 2026-09-26 on branch `ml`. **Evaluation completed. Retain the experimental water-only qualification.** Roadway Flooding performance is substantially better than the held-out FRED location, and every annotated empty-water test frame has some false-positive water. This review does not establish flooded-road extent or road safety.

## Freeze and leakage audit

- Frozen model: `water_20260926_mnv3s320_v1`, threshold **0.3**, `road=false`. ONNX SHA-256: `a68e2bfe23cc335e79d8f359d53dfd3f45284bf3c051fd79c9cdaf522ec4c27b`.
- Manifest SHA-256: `703f0040ee222528e86c7b0d8a11b6f7093a43bc1b9f71c3ac9100816de418d6`, matching the freeze record. Model/config, manifest, reviewed water code, training records, and classifier artifacts had identical hashes before and after this run. These checks precede main's authorized import/logging housekeeping. No weights, thresholds, or splits changed.
- Inspected `water_data`, `water_model`, `water_train`, `water_eval`, `water_metrics`, inference, `training_report.json`, and `selected_rows.json`. Saved selections exactly reproduce the seed-42 caps: **308 train** (125 FRED, 183 Roadway) and **186 validation** (32 FRED, 154 Roadway). No test rows are selected. Epoch 7 and threshold 0.3 were selected from validation metrics. The encoder is frozen, and padding is excluded from training loss/validation metrics.
- No cross-split group, original/processed path, mask path, duplicate-cluster, or exact perceptual-hash conflict involves either water source. SHA-256 checks of all **607 selected train/validation plus supported test frames** found no cross-split identical image files. Minimum test-versus-selected perceptual-hash distance is **12**, with **0 pairs at distance ≤6**. Unrelated GA511 train/validation collisions exist (one hash, four duplicate clusters), but no GA511 rows enter water training or evaluation. No water test leakage was found in the inspected code and provenance.

## Labels, geometry, and scope

All 607 inspected masks have supported values: FRED **2 = water**, 1 = road, 0 = other, consistent with its green/red palette and the local dataset audit. Roadway uses **1 = water**, 0 = background. FRED masks are 1728×1080 for 1920×1200 frames and are expanded by nearest-neighbor sampling. Roadway pairs are 512×384. Every pair has matching aspect ratio. The reviewed samples show consistent spatial alignment.

Evaluation uses original frames, full-frame 320×320 letterboxing, removal of padding, bilinear probability restoration, and thresholding at the original resolution. No classifier crops or sequence-level labels substitute for pixel masks.

Included all **113 supported masked test rows**. Excluded **62 Flood Master test rows** because that source is unsupported by the frozen water loader, even though the local dataset documentation describes binary labels. Also excluded **933 unmasked test rows** (591 FRED, 326 GA511, 16 NYSDOT), without inventing empty masks. FRED's 24 scored frames are one Cambogan sequence/location; Roadway contributes 89 duplicate groups. This is limited source coverage, not a broad camera-generalization result.

## One held-out pass

Called `inference.water.predict_water` once per supported test image. Independently counted binary confusion pixels and checked pooled scores against `water_metrics.summarize`. No threshold sweep, retraining, classifier test, or external download occurred. IoU = TP/(TP+FP+FN), Dice = 2TP/(2TP+FP+FN).

**Aggregation convention:** water training/evaluation assigns both-empty IoU/Dice **1**, while main's separate raw-CAM diagnostic assigns both-empty IoU **null**. Do not combine those means without reconciling their denominators. This water test contains **0 both-empty cases**, so that convention has no effect here. Empty-truth/nonempty-prediction images score 0 and remain in water mean-image scores. They are also reported separately below.

| Source | n | Pooled pixel IoU | Pooled pixel Dice | Mean image IoU | Mean image Dice |
|---|---:|---:|---:|---:|---:|
| FRED | 24 | 0.362779 | 0.532411 | 0.203357 | 0.284976 |
| Roadway Flooding | 89 | 0.777498 | 0.874823 | 0.753756 | 0.847368 |
| Overall | 113 | **0.665774** | **0.799357** | 0.636857 | 0.727922 |

Equal-source mean of pooled scores: **IoU 0.570139, Dice 0.703617**. Original-resolution pooling weights large FRED frames more heavily. For FRED's **17 nonempty masks**, mean image IoU/Dice are **0.287092/0.402320**.

**Empty-water false positives, reported separately:** FRED **7/7 frames (100%)** have at least one predicted water pixel, totaling **223,619 / 16,128,000 pixels (1.386527%)**. Individual false-positive area ranges from **0.308724% to 3.188802%** of a frame. Roadway has **0 empty masks**, so its empty-water false-positive rate is **not estimable**. These seven correlated FRED frames do not establish a general dry-camera false-alarm rate.

## Fixed visual sample and limitations

Selection was saved before inference: FRED minimum/median/maximum positive manifest water fraction plus two empty masks selected by a fixed path hash, four Roadway images selected by the same hash, and the first local Holmrook river JPG alphabetically. Inspected all **10 overlays**:

- FRED `21070048.png` nearly misses distant water (**IoU 0.003317**). `22570104.png` fragments it (**0.299953**). `29534471.png` detects the closer pool but extends onto adjacent pavement (**0.521695**). Empty frames `4462560.png` and `7355411.png` have bottom-edge/foreground false positives.
- Roadway `image_157.jpg`, `image_218.jpg`, and `image_239.jpg` capture broad water areas but bleed over vehicles, fences, or people. `image_92.jpg` misses part of the left water area, and its coarse annotation also covers a car/vegetation region. Sample IoUs are **0.894282, 0.791025, 0.928052, 0.696643**, respectively.
- Holmrook `2019-08-03-08-00-10.jpg` highlights the river (**41.607747%** of the frame), with some bank spill. This is a qualitative illustration only, excluded from every test metric. River highlighting cannot establish water covering a road.

The frozen config, inference note, demo, and API documentation explicitly describe water-only segmentation, keep it separate from classifier attribution/status, and disclaim flooded-road extent. No road-flood claim was found in that interface.

CPU `predict_water` latency was **10.501 ms mean, 7.787 ms median, 23.263 ms p95** across 113 images, including preprocessing, ONNX execution, restoration, and PNG encoding; excluding model loading/file decoding. The test loop including sample rendering took **2.334 s** on macOS arm64. This is one local run, not a deployment benchmark.

Evidence stays in ignored `reports/water_review/`: `audit.json`, `test_metrics.json` (full precision and per-image counts), `sample_selection.json`, `river_qualitative.json`, `overlays_fred.png`, `overlays_roadway_flooding.png`, `overlay_river_qualitative.png`, `run_review.py`, and `review_run.log`. Frozen `config.json` retains its original `test_evaluated=false`; this report records the subsequent independent test pass.
