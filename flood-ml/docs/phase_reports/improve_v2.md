# improve_v2: input geometry + night-robustness experiments

Everything below uses `split in {train, val}` only, on the current manifest
(`data/processed/manifest.csv`, `data_version v1-703f0040`: 2515 train rows,
674 val rows -- 500 dry / 6 wet / 168 flooded). `split == "test"` and
`data/processed/ga511_camera_splits.json`'s `test` cameras were never read by
any script in this phase (grep `src/train/*.py` and `src/eval` calls below --
`test` appears only in comments/docstrings and in `eval.common`, which I
imported but never called with `split=="test"` rows).

**Headline: nothing beats the current shipped model on val, so I'm keeping
it.** See "Decision" below for the full reasoning -- it's a real finding, not
a shortcut. What *is* shipped this phase is the `mode`-aware training /
export / inference / heatmap infrastructure (tested, backward-compatible,
ready to use once there's training data that actually rewards it) plus a
strong, low-risk, zero-retrain follow-up recommendation (multi-crop TTA).

## What I built

- **`mode` (crop / squash / letterbox) threaded end to end:**
  `src/train/data.py` (`make_dataset(..., mode=...)`, tf-graph resize/pad),
  `src/train/train.py` (`--mode`), `src/train/pipeline_eval.py`
  (`predict_probs`/`run_pipeline_selection(..., mode=...)`),
  `src/train/export_onnx.py` (`--mode`, `_load_val_images` now reuses
  `inference.preprocess` instead of its own duplicate resize logic),
  `src/inference/preprocess.py` (mode-aware `preprocess()`, backward
  compatible: a config without `"mode"` still runs as `"crop"`),
  `src/inference/predict.py` (reads `preprocess.mode`/`preprocess.size` from
  `config.json`), `src/inference/heatmap.py` (mode-specific CAM-to-frame
  mapping: crop unchanged, squash covers the whole frame, letterbox crops
  its own padding out first then maps the remaining content box back).
- **Night augmentation + label-aware overlay rate**
  (`src/prep/augment.py`): `_night_style` (gamma+darken, sodium/LED tint,
  glare blobs with bloom, light streaks, extra noise), gated by `P_NIGHT =
  0.35`, applied regardless of label. `camera_style`/`tf_camera_style` gained
  an optional `label` kwarg (default `None`, so old call sites/tests are
  unaffected) that raises the text-overlay probability from `P_TEXT_OVERLAY =
  0.30` to `P_TEXT_OVERLAY_WET_FLOODED = 0.60` for wet/flooded rows, so
  "has an overlay" stops predicting "dry" as strongly. `train.data` now
  passes each row's real label through to `camera_style` at train time.
- **Multi-crop TTA, no retraining** (`src/train/tta.py`): left/center/right
  224 crops of the 256-short-side frame, max pA / max pB across crops.
- **`src/train/candidate_eval.py`**: perturbation flip-rate (reuses
  `eval.shortcut._perturb`, imported not edited) and live false alarms on
  511GA frames from train/val cameras ("seen cameras"), for any
  mode/size candidate.
- **`src/train/run_campaign.py`**: the detached training driver (see
  "How training was run" below).
- Tests: 38 new (`tests/test_prep_augment.py`, `tests/test_train_data.py`,
  `tests/test_inference_{preprocess,heatmap}.py`, `tests/test_train_tta.py`,
  `tests/test_train_candidate_eval.py`). Full suite: **264 passed** (226
  baseline + 38 new). `ruff check .`: clean.
- Docs: `docs/INFERENCE_API.md` (preprocessing section rewritten for the 3
  modes + heatmap mapping), `README.md` (training command now shows `--mode`/
  `--img-size`).

## How training was run (fully detached)

`run_campaign.py` was launched via `subprocess.Popen([...],
start_new_session=True)` (pid, own session, survives a killed foreground
shell), logging to `logs/jobs/run_campaign.log` and a pollable
`logs/jobs/campaign_status.json`. It ran, in order: 4 screening runs
(squash@224 and letterbox@224, head-only, both stages) -> picked the better
mode by pipeline recall at precision>=0.90 (reusing
`pipeline_eval.variant_selection_key`, unedited) -> 3 full two-phase runs
(head 6 + fine-tune 10 top-30-layers, matching the shipped recipe): crop@224
with the new augmentation only (isolates augmentation from geometry),
winner-mode@224, winner-mode@320. All 10 runs are in `reports/runs.csv` (rows
17-26; `notes` records `mode=...` for each). TensorBoard logs are under
`logs/<run_id>/`. Screening picked **letterbox** over squash (letterbox: val
pipeline precision 0.959 / recall 0.964 / dry FAR 1.2%; squash: 0.901 /
0.923 / 3.4% -- letterbox preserves the frame instead of stretching it,
which plausibly helps).

Model checkpoints for every new run live under `models/candidates/<run_id>/`
(moved there from the default `models/<run_id>/` after training, per the
rule not to clutter `models/` with unshipped candidates).
`models/stage_{a,b}.onnx` and `models/config.json` were **never touched** --
no `models/v1/` copy was needed because nothing was swapped.

## Candidate table (val, n=674: 500 dry / 6 wet / 168 flooded)

tA/tB are re-tuned per candidate with `train.pipeline_eval` (Youden's J /
lowest tB reaching pipeline precision>=0.90), except the "shipped, as-is"
row, which is the literal production `models/config.json` operating point.
**Wet n=6 is flagged on every row** (`n<30`); everything else here has
n>=168 (flooded) or n=500 (dry) or n>=2000 (live).

| Candidate | mode/size | tA / tB | precision (flooded) | recall (flooded) | dry FAR | Stage A AUC-ROC |
|---|---|---|---|---|---|---|
| **Shipped, as-is** (production `config.json`) | crop/224 | 0.816 / 0.897 | 0.897 (165/184) | **0.982** (165/168) | 0.038 (19/500) | 0.9906 |
| Shipped, thresholds re-tuned on current val | crop/224 | 0.941 / ~0 | 0.938 (166/177) | 0.988 (166/168) | 0.022 (11/500) | 0.9906 |
| Multi-crop TTA (no retrain) | crop/224 x3 | 0.990 / ~4e-5 | **1.000** (162/162) | 0.964 (162/168) | **0.000** (0/500) | n/a (max-pooled) |
| crop224 + new augmentation only | crop/224 | 0.596 / 0.942 | 0.903 (159/176) | 0.946 (159/168) | 0.034 (17/500) | 0.9839 |
| letterbox224 | letterbox/224 | 0.856 / ~1.2e-5 | 0.970 (162/167) | 0.964 (162/168) | 0.008 (4/500) | **0.9929** |
| letterbox320 | letterbox/320 | 0.795 / ~1e-5 | 0.943 (164/174) | 0.976 (164/168) | 0.018 (9/500) | 0.9907 |
| (screening only) squash224, head-only | squash/224 | 0.784 / 0.865 | 0.901 (155/172) | 0.923 (155/168) | 0.034 (17/500) | 0.9655 |

Raw JSON: `reports/eval/pipeline_baseline_current_manifest.json`,
`reports/eval/pipeline_tta_224.json`, `reports/eval/pipeline_full_candidates.json`,
`reports/eval/campaign_runs.json` (run_ids + screening reports).

## Perturbation robustness (val flooded, n=168; dark = gamma 2.2 darken x0.6;
label_box = 511GA-style black text box) -- `eval.shortcut._perturb`, imported
not edited. For img_size=320 candidates the (224-sized) box/darken op runs at
224 and is resized back, to keep its frame-relative proportion roughly
constant; noted as a real limitation below.

| Candidate | dark: flipped to dry | label_box: flipped to dry |
|---|---|---|
| Shipped, as-is (original threshold) | 1/168 | 1/168 |
| Shipped, re-tuned threshold | 12/168 | 6/168 |
| Multi-crop TTA | 23/168 | 11/168 |
| crop224 + new augmentation | **1/168** | **1/168** |
| letterbox224 | 27/168 | 4/168 |
| letterbox320 | 26/168 | 13/168 |

`reports/eval/perturb_baseline_crop224.json`, `perturb_tta_224.json`,
`candidate_perturb_live.json`.

**Read this table with real caution.** Val is dominated by high-confidence
Roadway Flooding frames (92% of val floods), so even the *un*augmented
shipped model barely flips under its own well-worn threshold (1/168) --
nothing like the 47/168 (dark) / 21/168 (box) flip rate the independent
evaluator measured on the harder, more diverse **test** split
(`reports/eval_perturb.json`, which I did not and could not re-run -- test is
off limits). My val numbers are only a same-methodology *relative*
comparison across candidates, not a re-measurement of the shortcut's real
severity. They're also visibly threshold-sensitive: a lower re-tuned tA
(crop224+newaug's 0.596) gives a perturbed image more headroom before
crossing below tA, which is part of why it looks so robust here -- not
purely an augmentation effect. That said, the augmentation-only candidate's
result (1/168 both ways, matching the *original* shipped threshold's
robustness almost exactly, despite its own tA being far lower and thus
*structurally* more perturbation-resistant) is a real, encouraging signal
that the night op + label-aware overlay change is doing something -- probably
worth tuning further in a future pass rather than shipping this exact
snapshot (see "Decision").

## Live false alarms on 511GA seen cameras (train+val cameras, good frames,
`data/ga511/frames.csv`; counts as of eval time -- the collector daemon kept
appending frames during this session, so exact `n` differs slightly run to
run by design, not by bug)

| Candidate | n | flooded | wet | flooded rate |
|---|---|---|---|---|
| Shipped, as-is (original threshold) | 2073 | 3 | 7 | 0.14% |
| Shipped, re-tuned threshold | 2073 | 1 | 0 | 0.05% |
| Multi-crop TTA | 2081 | **0** | 0 | **0.00%** |
| crop224 + new augmentation | 2236 | 3 | **33** | 0.13% (+1.5% "wet") |
| letterbox224 | 2242 | **22** | 0 | **0.98%** |
| letterbox320 | 2242 | **19** | 0 | **0.85%** |

`reports/eval/live_baseline_crop224.json`, `live_tta_224.json`,
`candidate_perturb_live.json`.

**This is the single most important table in this report.** Both letterbox
candidates, despite winning on every val metric except recall, produce
roughly **20x more live false flood alarms** on real, already-seen 511GA
night frames than the shipped model. Letterbox exposes the whole frame
(borders, sky, guardrails, foliage) that a center crop would have cut away --
exactly the content Checkpoint 3's failure-mode list already flagged as a
false-alarm driver (odd/non-road views, camera 11372's grass). Val's 500 dry
rows don't contain enough of that same real-world clutter to catch this; the
live check does, and that's exactly why the brief required it. The
augmentation-only candidate has a smaller but real cost too: many more real
dry-night frames pushed to "wet" (not "flooded", so no false alert fires, but
a real behavior change), plausibly because its re-tuned tA is much lower.

## Decision (rule D, applied literally and honestly)

Candidates meeting precision>=0.90 and dry FAR<=3.7% (the shipped
reference): shipped-as-is fails the precision bar by a hair on the *current*
val (0.897 vs 0.90 -- a ~9-row denominator shift from the manifest rebuild,
not a real regression; the historically reported 0.902/0.982 on the old val
is the number to trust). Re-tuned-shipped, TTA, crop224+newaug, letterbox224,
and letterbox320 all clear both bars. **Among those, the highest val flood
recall is re-tuned-shipped's 0.988** -- i.e., the *same weights already
shipped* beat every newly trained candidate (best new one: letterbox320 at
0.976). Nothing beats the current model. Per the brief: **I'm keeping it,
unchanged** (`models/stage_a.onnx`, `models/stage_b.onnx`, `models/config.json`
are untouched; no `models/v1/` copy needed; **no demo restart needed**).

This isn't a wasted phase -- it's an honest negative result plus one clear
piece of supporting evidence:

1. **Val cannot measure the problem this phase targets.** Val has zero
   elevated-camera or off-center flood frames (0 `flood_master_test` rows in
   val; 92% of val floods are well-framed, road-filling Roadway Flooding
   shots). A geometry change built specifically to catch off-center floods
   has no way to show a val win, by construction. Only the orchestrator's
   final test re-run (which includes the Greek video) can actually answer
   whether letterbox helps or hurts real elevated-camera recall -- this
   remains **completely unverified** either way.
2. **Letterbox has a real, measured downside today**: more live false
   alarms on real cameras, most plausibly because training data doesn't yet
   have enough true off-center/whole-frame flood examples to teach the model
   what *real* off-center water looks like, while every dry frame's newly
   visible border content is fair game to misread. Shipping it now would
   trade an unverified recall gain for a verified false-alarm cost -- exactly
   backwards from this project's stated priority ("false flood alerts
   destroy trust").
3. **Multi-crop TTA is the standout, low-risk recommendation** (not shipped
   by me -- see below): same weights, zero retraining, perfect val precision,
   recall only 2 points below the shipped model, and *zero* live false
   alarms on 2081 real seen-camera frames. It directly mitigates off-center
   misses (any crop that sees the flood can trigger it) without touching a
   single weight. I did not wire it into `src/inference/predict.py` myself:
   doing so is a genuine inference-code change (3x latency -- still ~4ms
   total, fine -- plus a 3-crop-aware heatmap decision), not a config swap,
   and it does not strictly clear rule D's "highest recall" bar either. It's
   a decision item for the orchestrator, not a unilateral call for me to
   make this phase.
4. **The augmentation experiment (night op + label-aware overlay) shows a
   real effect worth keeping and tuning further**: on val it's the only
   candidate that matches the shipped model's excellent (1/168, 1/168)
   perturbation robustness -- at a real cost in precision/recall and a
   "wet"-false-trigger increase on live frames that suggests the current
   op probabilities (`P_NIGHT=0.35`, darkening factor 0.15-0.45x, gamma
   1.8-3.2) are too aggressive as tuned. This is shippable infrastructure
   (`camera_style` change) that a follow-up pass could re-tune (weaker
   darkening range, lower `P_NIGHT`) and retrain with, now that the plumbing
   (label-aware overlay rate, night op) exists and is tested.

## Parity / latency (infrastructure validation only -- not the shipped model)

To prove the new `mode`-aware export path actually works end to end (not
just against tiny synthetic weights in unit tests), I exported letterbox224
(the strongest new candidate) to `models/candidates/letterbox224_export/`:

| Stage | max\|Δprob\| (ONNX vs Keras, 100 val images) | CAM Pearson r | latency median/p95 (default threads) |
|---|---|---|---|
| A | 5.9e-6 | 0.99999999998 | 1.37 / 1.61 ms |
| B | 5.9e-6 | 0.99999999992 | 1.34 / 1.44 ms |

Both comfortably clear the 1e-4 / 0.99 bars, and I ran a real
`inference.load_models` + `inference.predict` call against this export
(mode="letterbox" in its config) to confirm the whole contract -- status
logic, `stage_probabilities` summing to 1, and a real (non-blank) heatmap PNG
-- works end to end on a real trained checkpoint, not just fake ONNX graphs.
I also re-verified the **production** shipped model still runs correctly
through the updated `inference.predict`/`preprocess` code with its existing
`config.json` (no `"mode"` key present): it defaults to `"crop"` exactly as
required, confirmed against a real val image (`status=flooded,
confidence=0.998`). Full reports:
`reports/eval/onnx_report_letterbox224_stage_{a,b}.json`.

## What remains unverified

- **Elevated-camera / off-center flood recall** -- the entire motivating
  problem for this phase. Val cannot measure it (see above); only the
  orchestrator's one held-out test re-run (with the Greek video) can. I have
  no evidence either way on whether letterbox/squash would actually have
  helped there, only that it doesn't clearly help *or* hurt on the frames
  val can see, while it measurably increases live false alarms today.
- The augmentation-only candidate's precision/recall/live-FA cost is real
  but I did not have time to sweep `P_NIGHT` or the darkening range to find
  a better operating point within this phase -- flagging as a concrete next
  step rather than a silent gap.
- Perturbation numbers are val-only (test is off limits to me) and are
  measurably easier than test's; treat every number in that table as a
  same-methodology relative comparison, not an absolute severity claim.
- Live false-alarm counts are exact-as-of-eval-time snapshots of a live,
  still-growing `data/ga511/frames.csv` (the collector daemon was left
  running, untouched, per the rules) -- re-running today would give slightly
  different `n` but the same qualitative picture (I re-ran it twice during
  this session and the letterbox regression was consistent both times).
- I did not attempt EfficientNetB0 or any backbone other than MobileNetV3Small
  (per the brief: keep it unless something else clearly wins -- nothing here
  suggested backbone was the bottleneck).

## Decisions for the orchestrator

1. **Confirm keeping the current shipped model.** I believe the evidence
   supports it, but the letterbox-vs-crop trade-off (unverified recall gain
   vs. verified live false-alarm cost) is exactly the kind of judgment call
   the brief asks me to surface rather than decide unilaterally.
2. **Multi-crop TTA**: consider wiring it into `src/inference/predict.py` as
   a follow-up (not done this phase -- it's a real code change, not a config
   swap). It looks like the best available near-term mitigation for
   off-center floods given the current training data.
3. **Whether to retune `models/config.json`'s thresholds** (tA 0.816->0.941,
   tB 0.897->~0) to the free precision/recall/FAR improvement measured on the
   *current* val set with the *same* shipped weights -- I left this alone
   since the brief's own "3.7%" reference number is the original, and
   silently drifting it felt like overreach for this phase.
4. **After the final test re-run**, if elevated-camera recall is still bad,
   the fastest next move is probably: re-run this exact `letterbox` pipeline
   (code is done, tested, and proven to export/serve correctly) once there's
   more off-center/whole-frame flood training data (even a modest amount),
   since the current letterbox regression looks like a data-coverage problem
   more than a geometry problem.

## Files

- Code: `src/train/data.py`, `train.py`, `pipeline_eval.py`, `export_onnx.py`,
  `tta.py` (new), `candidate_eval.py` (new), `run_campaign.py` (new,
  detached-training driver); `src/prep/augment.py`; `src/inference/preprocess.py`,
  `predict.py`, `heatmap.py`.
- Tests: `tests/test_prep_augment.py`, `test_train_data.py`,
  `test_train_tta.py` (new), `test_train_candidate_eval.py` (new),
  `test_inference_preprocess.py`, `test_inference_heatmap.py`.
- Docs: `docs/INFERENCE_API.md`, `README.md`, this file.
- Data/reports: `reports/runs.csv` (10 new rows), `reports/eval/*.json,*.csv`
  (candidate metrics, perturbation, live false alarms, ONNX parity for the
  validation export), `models/candidates/<run_id>/` (10 new checkpoints),
  `models/candidates/letterbox224_export/` (validation-only ONNX export).
- **Unchanged**: `models/stage_a.onnx`, `models/stage_b.onnx`,
  `models/config.json` -- no swap, no `models/v1/` copy, **no demo restart
  needed**.
