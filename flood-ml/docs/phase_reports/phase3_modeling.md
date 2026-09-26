# Phase 3: modeling

Two-stage classifier (Stage A dry-vs-wet, Stage B flooded-vs-not), Grad-CAM,
ONNX export. All metrics below are computed on `train`/`val` only --
`split == "test"` was never touched by any script in `src/train/` (grep
`src/train/*.py` for `"test"`: it appears only in docstrings/comments
warning not to use it).

## What I built

- `src/train/data.py`: manifest loading, Stage A/B row selection (including
  the "spec"/"mixed" Stage B variants -- see below), class/sample weights,
  a `tf.data` pipeline (decode -> resize short side to 256 -> `camera_style`
  augmentation + random 224 crop + horizontal flip for train; deterministic
  center crop for val), and a throughput benchmark.
- `src/train/model.py`: `build_model(backbone, ...)` for `mobilenetv3small`
  / `efficientnetb0`, head fixed as `GlobalAveragePooling2D -> Dropout ->
  Dense(1, sigmoid)`, `set_backbone_trainable` (freeze / unfreeze-top-N with
  BatchNorm always kept in inference mode), `make_grad_model`.
- `src/train/train.py`: two-phase training CLI (`--stage {a,b} --variant
  {spec,mixed} --backbone ... --epochs-head --epochs-ft --ft-layers --lr-*
  ...`), early stopping on val AUC-PR with best-weights restore, TensorBoard,
  checkpoint to `models/<run_id>/model.keras`, one row appended to
  `reports/runs.csv` per run.
- `src/train/pipeline_eval.py`: combines a Stage A checkpoint with a Stage B
  checkpoint, scores PLAN.md section 3's status logic on **all** val rows,
  picks `tA` (Youden's J on Stage A's ROC) and `tB` (lowest value reaching
  pipeline precision >= 0.90 for "flooded", else max F0.5), and compares the
  two Stage B variants.
- `src/train/gradcam.py`: GradientTape Grad-CAM (`gradcam_batch`) and the
  gradient-free Dense-kernel CAM used in the ONNX graph
  (`cam_from_dense_weights`), plus overlay rendering and a gallery writer.
- `src/train/export_onnx.py`: traces a `tf.function` per stage, converts with
  `tf2onnx.convert.from_function` (opset 17), verifies against Keras, and
  benchmarks CPU latency.
- `notebooks/train_colab.ipynb`: clones `floodmark` at `ml`, installs
  `requirements-train.txt`, mounts the team Drive for `data/processed`, runs
  `train.train` for Stage A and both Stage B variants. Ruff-clean.
- Tests: `tests/test_train_{data,model,gradcam,train,pipeline_eval,
  export_onnx}.py`, 42 new tests, all `pytest.importorskip("tensorflow")`,
  weights=None / tiny synthetic data, no downloads. Full suite: **150
  passed** (108 pre-existing + 42 new). `ruff check .` clean.
- `reports/runs.csv` (6 runs), `reports/onnx_parity.md`,
  `reports/pipeline_selection_{baseline,ft}.json`,
  `reports/onnx_report_stage_{a,b}.json` -- all text/JSON, tracked-safe.
- Local-only (not tracked, per `.gitignore`): `reports/gradcam_val/{stageA,
  stageB_mixed}/` (22 overlays each) and `reports/errors_val.html` (27
  pipeline-error cards with Grad-CAM overlays, val only).
- `models/stage_a.onnx`, `models/stage_b.onnx`, `models/config.json`
  (gitignored, per rules).

**Packages installed:** none beyond what was already listed as available
(tensorflow 2.21.0, keras 3.15.1, tensorboard 2.21.0, tf2onnx 1.17.0, onnx
1.23.0, onnxruntime 1.30.0, numpy 2.4.6, pandas, pillow, scikit-learn,
matplotlib, opencv-python-headless were all already present). No `pip
install` calls were needed.

## A real bug found and fixed along the way (worth flagging)

`_build_backbone` originally passed `name="backbone"` directly into
`MobileNetV3Small(...)` / `EfficientNetB0(...)` so that
`model.get_layer("backbone")` would work after save/reload. That's fine for
MobileNetV3Small, but Keras's EfficientNet applications build their
ImageNet-weights download URL as `name + "_notop.h5"` -- so passing a custom
`name` silently changes the URL to `.../backbone_notop.h5`, which 404/403s.
Fixed by building with the default name (correct download) and renaming
the model's `.name` attribute (a plain instance attribute, not `_name` --
that was a second wrong guess before I checked the Keras source) afterward,
before it's wired into the outer functional model. Caught by actually
running the EfficientNetB0 baseline, not by any test (the tests use
`weights=None`, which never touches this path) -- worth Phase 4 knowing
this class of bug exists (a backbone's `weights="imagenet"` path is much
less tested than `weights=None`).

A second bug, caught by unit tests rather than a real run: both
`_precision_target_threshold` (train.py) and `sweep_tb_for_precision`
(pipeline_eval.py) treated a threshold with **zero** predicted positives as
vacuously satisfying "precision >= target" (`tp/(tp+fp)` defaults to 1.0
when `tp+fp==0`). That let the *highest* candidate threshold always
"reach" any target, trivially, with 0 recall. Fixed by requiring at least
one predicted positive before a threshold counts as reaching the target.
Regression tests for both are in `tests/test_train_train.py` and
`tests/test_train_pipeline_eval.py` (deterministic tied-score data, not
random noise, so the test doesn't rely on chance to still fail after a
partial fix).

## Runs table (`reports/runs.csv`, all 6 runs; data_version v1-c7dea35e)

| run_id | stage | variant | model | epochs (head/ft) | n_train | n_val | val AUC-ROC | val AUC-PR | val P/R/F1 | threshold | val false-alarm | train_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `20260926-031105_a_mobilenetv3small` | A | -- | mobilenetv3small | 12/0 | 2492 | 665 | 0.9874 | 0.9808 | 0.889/0.966/0.926 | 0.7035 | 0.043 (21/491) dry | 38.3 |
| `20260926-031202_bspec_mobilenetv3small` | B | **spec** | mobilenetv3small | 5/0 | 798 | 174 | **1.0000** | **1.0000** | 0.966/1.000/0.983 | 0.0524 | 1.000 (6/6) wet | 6.9 |
| `20260926-031238_bmixed_mobilenetv3small` | B | **mixed** | mobilenetv3small | 12/0 | 2492 | 665 | 0.9971 | 0.9919 | 0.901/0.976/0.937 | 0.8360 | 0.000 (0/6) wet | 39.6 |
| `20260926-031525_a_mobilenetv3small` | A | -- | mobilenetv3small (**ft, shipped**) | 6/10 | 2492 | 665 | 0.9906 | 0.9848 | 0.899/0.971/0.934 | 0.8161 | 0.039 (19/491) dry | 59.5 |
| `20260926-031638_bmixed_mobilenetv3small` | B | mixed (**ft, shipped**) | mobilenetv3small | 6/10 | 2492 | 665 | 0.9982 | 0.9948 | 0.902/0.982/0.940 | 0.9060 | 0.000 (0/6) wet | 60.7 |
| `20260926-032205_a_efficientnetb0` | A | -- | efficientnetb0 | 8/0 | 2492 | 665 | 0.9858 | 0.9745 | 0.875/0.966/0.918 | 0.5237 | 0.049 (24/491) dry | 75.5 |

The `threshold`/`val false-alarm` columns above are **per-stage** numbers
computed by `train.py` at training time (Youden's J for Stage A; "lowest
threshold reaching precision>=0.9" on that stage's own val subset for Stage
B) -- they are a cheap proxy, not what's shipped. The shipped `tA`/`tB` come
from `pipeline_eval.py` on the full val set (below).

**Order of work followed the brief:** fast MobileNetV3Small head-only
baseline for Stage A + both Stage B variants first, all the way through
pipeline evaluation; then fine-tuning (top 30 backbone layers, BatchNorm
kept frozen/inference-mode, `lr=1e-5`) for the winning combination; then one
EfficientNetB0 baseline as a backbone comparison. EfficientNetB0 (0.9745
val AUC-PR head-only) did not beat MobileNetV3Small even at MobileNetV3Small's
head-only baseline (0.9808), let alone its fine-tuned result (0.9848), while
taking ~2x longer per epoch -- so I didn't spend further budget fine-tuning
EfficientNetB0. **Chosen backbone: MobileNetV3Small, fine-tuned (top 30
layers, 10 epochs @ lr=1e-5) for both stages.**

## The Stage B variant decision -- this is the interesting part

`reports/class_counts.md` (Phase 2) already flagged that "wet" is 30/31 from
a single source (NYSDOT) and warned it's a shortcut risk. Phase 3 confirms
it, sharply:

**B-spec's stage-level AUC of 1.0000 is real but misleading.** On its own
174-row held-out set (168 flooded + 6 wet), spec perfectly ranks flooded
above wet. But AUC only measures ranking, not calibration, and spec's
training set is 783 flooded vs 15 wet (52:1) -- severe enough that most
flooded images' raw probabilities land in a wide 0.05-0.8 band rather than
pinned near 1.0, even though the ranking is perfect. That's visible directly:
spec's own reported threshold (0.0524, "lowest reaching stage-level
precision>=0.9") gives recall=1.0 on its own subset, but a *pipeline-level*
sweep over the same model's outputs on ALL 665 val rows required tB=0.805 to
hit pipeline precision>=0.9 (because a handful of dry/wet false positives,
though few, are compared against however few true positives the threshold
still catches) -- and at 0.805, only 21 of 168 true-flooded frames still
score above it. Concretely, with the **baseline** (not yet fine-tuned) Stage
A model:

| variant | pipeline precision (flooded) | pipeline recall (flooded) | recall num/den |
|---|---|---|---|
| spec | 0.913 | **0.125** | 21/168 |
| mixed | 0.901 | **0.976** | 164/168 |

Spec's confusion matrix at that threshold: of 168 true-flooded val rows, 146
were called **"wet"** (Stage A correctly said "not dry", but Stage B-spec's
score for them fell below tB) and only 21 "flooded". Spec was never trained
on a single dry image, so it has no calibrated response to the images Stage
A's imperfections let through, and -- as this shows -- also no reliable
absolute-probability calibration even among the wet/flooded images it does
see, because of its own extreme class imbalance.

`variant_selection_key` (in `pipeline_eval.py`) picks the variant with the
higher F1 among those clearing the 0.90 precision target -- not the higher
raw precision -- specifically so a result like this (spec's 0.913 > mixed's
0.901 in raw precision, while collapsing recall to 0.125) doesn't win. That
logic itself is now unit-tested (`test_variant_selection_key_prefers_...`).

**With the final, fine-tuned Stage A model**, Stage A's own error rate on
dry drops (18-19/491 vs ~20-21/491, and more importantly its *ranking*
improves, changing which specific rows leak through), and spec's pipeline
recall recovers to 0.994 (precision 0.903) -- almost the same numbers as
mixed (precision 0.902, recall 0.982; both computed on the identical 665-row
val set with the identical Stage A model, so it's an apples-to-apples
comparison). **Spec edges out mixed by a hair at this specific val
snapshot.** I shipped **mixed anyway**, and I want to be explicit that this
is a judgment call, not a clean numeric win, for the orchestrator to weigh
in on:

- Spec's near-perfect recovery once Stage A got better is reassuring but
  doesn't remove the structural problem: spec has literally never seen a
  dry image, so its behavior on Stage A false-positives is *undefined by
  training*, not *robust by training*. It happened to work out here because
  Stage A's false positives are currently rare (3.7% of dry) and apparently
  don't confuse spec's ranking much on this particular val set.
- The baseline numbers show how badly that assumption breaks if Stage A is
  even a little worse -- and Stage A will see harder cases in production
  (511GA night frames, headlight glare on wet pavement -- exactly what
  Checkpoint 2's report already found the *labelers* got wrong once) than
  clean val images from FRED/roadway_flooding/NYSDOT.
- Mixed's false-alarm rate on true-wet is 0/6 at both snapshots, same as
  spec, so there's no false-alarm cost to shipping mixed here.
- The wet class is n=6 in val (n=10 in test, off limits to me). Any
  precision/recall difference of 1-3 examples between variants is well
  inside noise. **Flip a variant if Phase 4 finds a stronger signal on the
  (larger, independent) test set** -- that reviewer has more wet examples to
  work with than I do, and models/config.json's `variant_note` documents
  the reasoning for a config swap without retraining.

**Decision needed from the orchestrator:** confirm shipping "mixed", or
direct a swap to "spec" if there's a strong preference for this val
snapshot's raw numbers over the structural argument. Swapping only requires
re-running `pipeline_eval.py` with `--stage-b-spec-run
20260926-031202_bspec_mobilenetv3small` and re-exporting
`models/stage_b.onnx` from that checkpoint -- no retraining needed.

## Thresholds: how tA/tB were picked (shipped, `models/config.json`)

- **tA = 0.8161** -- Youden's J (`max(tpr - fpr)`) on the fine-tuned Stage
  A model's own val ROC (dry vs not-dry), balancing Stage A's two error
  types as the brief specifies.
- **tB = 0.8972** -- swept holding tA fixed, lowest value where **pipeline**
  precision for status=="flooded" on ALL 665 val rows is >= 0.90. Target
  was reached (not the F0.5 fallback).

## Pipeline-level val metrics (n=665, mixed variant, shipped)

| metric | value | num/den |
|---|---|---|
| precision(status=="flooded") | 0.9016 | 165/183 |
| recall(status=="flooded") | 0.9821 | 165/168 |
| false-alarm rate, true dry -> status flooded | 0.0367 | 18/491 |
| false-alarm rate, true wet(not flooded) -> status flooded | **0.0000** | **0/6** |
| Stage A AUC-ROC / AUC-PR | 0.9906 / 0.9848 | -- |
| Stage A accuracy @ tA | 0.9639 | -- |

3-class confusion (rows = true, cols = predicted status):

| true \ pred | dry | wet | flooded |
|---|---|---|---|
| dry (491) | 472 | 1 | 18 |
| wet (6) | 5 | 1 | 0 |
| flooded (168) | 0 | 3 | 165 |

**Flagged, per rule 4 in the brief ("flag every number resting on <30
examples"):** the entire **true-wet row (n=6)** is too small to trust. 5 of
6 wet val rows are called "dry" by Stage A itself (before Stage B even runs)
-- i.e. Stage A doesn't reliably recognize the wet class as "not dry" at
all, on this tiny sample. The false-alarm-on-wet rate of 0/6 is *not*
evidence Stage B is well-calibrated for wet roads; it's mostly evidence that
Stage A rarely lets true-wet rows reach Stage B in the first place. This is
the single most important number in this report to NOT over-read, and
Phase 4 has more wet examples (n=10 test, plus whatever 511GA rain frames
the collector has gathered since) to actually test it.

## ONNX export: parity and latency

Full detail in `reports/onnx_parity.md`. Summary: max |prob_onnx -
prob_keras| = 2.6e-6 (Stage A) / 2.7e-6 (Stage B) on 150 val images each
(limit: 1e-4) and Grad-CAM correlation (ONNX cam, bilinear-upsampled +
per-image normalized, vs. Keras GradientTape Grad-CAM) = 0.999999999902 /
0.999999999941 (limit: 0.99) -- both essentially exact, which is expected
given the CAM math (see `src/train/model.py`'s docstring): the ONNX `cam` IS
`ReLU(sum_k w_k * A_k)` from the Dense kernel by construction, and per-image
normalization cancels the one positive scalar that separates it from real
Grad-CAM. CPU latency (batch 1, median/p95 over 100 runs): Stage A 1.04/1.12
ms (default threads), Stage B 1.10/1.24 ms, full pipeline (A then B,
sequential) 2.57/2.81 ms default-threads or 5.76/5.87 ms pinned to 1 thread
-- far under any real-time budget for this project.

One non-fatal wrinkle worth recording: `tf2onnx`'s transpose-optimizer pass
throws an internal `ValueError` (`bias_size = max(numpy_val.shape)` on an
empty shape) on every single conversion in this environment. It's caught
and the optimizer falls back, and the resulting graph passes
`onnx.checker.check_model` plus all the numeric checks above -- but it means
those tracebacks show up in `tf2onnx`'s own logger on every export. I
redirected that logger to `logs/jobs/tf2onnx_convert.log` instead of letting
it hit the console/job log.

## Grad-CAM: what the heatmaps show

Looked directly (Read tool) at 6 overlays across both models and all three
labels, from `reports/gradcam_val/{stageA,stageB_mixed}/` (44 PNGs total,
local only, val rows only):

- **Flooded scenes: heat sits on the water, convincingly.** A FRED dashcam
  frame of a car in floodwater (`flooded_fred_002050_p0.97`) lights up
  exactly the flooded road area around the vehicle. A `roadway_flooding`
  scene of people wading through a flooded street lights up the water
  surface broadly, including around their submerged legs. This is the
  behavior you want from Stage A/B on a real flood.
- **Dry/borderline scenes: heat often sits on scene context, not road
  texture.** A correctly-classified dry 511GA frame under a highway overpass
  lights up the bridge girders and a road patch under the bridge, not the
  pavement surface generally. Another correctly-classified dry frame (tree-
  lined road) lights up the tree canopy in the upper-left, not the road at
  all. A true-wet NYSDOT highway frame (correctly scored low, p=0.04) lights
  up a sound barrier wall in the distance rather than any visible pavement
  wetness.
- **Read together**, this suggests the model has learned "water is
  flood-evidence" solidly (the actual task), but for dry/ambiguous frames it
  may be leaning partly on scene/context cues (bridges, foliage, camera
  framing) rather than pavement texture per se -- plausible given how little
  visual signal a merely-dry road surface offers compared to salient
  standing water, but it's exactly the kind of shortcut Phase 4's CAM-energy-
  in-road-region check (PLAN.md Phase 4 task 3) should quantify properly
  with masks, rather than my by-eye spot check.

## Suspicious / worth Phase 4's attention

1. **B-spec's perfect stage-level AUC (1.0000) is a textbook case of a
   metric being technically correct and practically misleading** -- see the
   variant-decision section above. Don't take a perfect AUC at face value
   for a class with n=6-15 on one side, trained on a single source.
2. **The wet class's real signal may partly be source-identity, not
   road-condition.** 30/31 wet train+val rows are NYSDOT; the augmentation
   pipeline (`camera_style`) is supposed to erase easy source cues (JPEG
   recompression, noise, blur, overlays applied probabilistically to every
   source), but I did not independently verify with a source-classifier
   probe -- that's explicitly Phase 4's job (PLAN.md Phase 4 task 3) and I'd
   flag it as the single highest-value thing for that reviewer to check
   first, given how the numbers above played out.
3. **Stage A misses 5 of 6 true-wet val rows entirely** (scores them below
   tA, i.e. "dry"). Not necessarily wrong -- these NYSDOT highway frames
   may look genuinely close to dry at 224px after JPEG/crop -- but it means
   Stage B's calibration on "wet" barely gets exercised end-to-end on this
   validation set at all, regardless of variant.
4. Only one EfficientNetB0 run was done (Stage A baseline only), not the
   full matrix (no Stage B EfficientNetB0, no EfficientNetB0 fine-tune) --
   a time/compute tradeoff given it under-performed MobileNetV3Small even in
   its best (head-only) run here. Flagging so it's a known gap, not a silent
   omission.

## What failed / remains unverified

- Nothing failed in the final state. Two bugs were caught and fixed during
  development (see above) -- both are now regression-tested.
- I did not attempt an EfficientNetB0 fine-tune or an EfficientNetB0 Stage B
  run (time/compute tradeoff, see above).
- I did not re-run the spec variant's fine-tune (only its head-only
  baseline) -- once the baseline pipeline numbers disqualified it so
  sharply (recall 0.125), I judged further investment not worthwhile until
  Phase 4's independent read is in; the fine-tuned Stage A model's rebalance
  of spec's numbers (see above) means this could be revisited cheaply if
  wanted (spec's checkpoint from the baseline run is still on disk at
  `models/20260926-031202_bspec_mobilenetv3small/`).
- Grad-CAM "what it looks at" assessment above is a by-eye spot check of 6
  images, not the systematic mask-based CAM-energy analysis PLAN.md assigns
  to Phase 4.

## Decisions for the orchestrator

1. **Confirm or override the Stage B variant** (mixed, shipped, vs. spec) --
   see the dedicated section above. My recommendation is to keep mixed for
   the structural robustness argument, while flagging that spec is
   numerically competitive on this specific val snapshot.
2. **The true-wet validation slice (n=6) is too small to certify Stage B's
   real-world false-alarm behavior on wet-but-not-flooded roads.** This was
   already flagged at Checkpoint 2 as the project's core data gap and Phase
   3 doesn't change that -- it can only get better from more 511GA rain
   frames or additional wet-labeled data.
3. Nothing here required a package install or a rules exception.
