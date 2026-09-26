"""Unit tests for train.pipeline_eval's pure-numpy logic: the PLAN.md
section 3 status rule, the tB sweep, and small-sample flagging. No models,
no TensorFlow needed at runtime, but importorskip("tensorflow") anyway per
the training-tests convention (this module lives under train/ and is
exercised together with modules that do need it).
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from train.pipeline_eval import (
    SMALL_SAMPLE_WARN_N,
    evaluate_pipeline,
    per_source_breakdown,
    status_from_probs,
    sweep_tb_for_precision,
    variant_selection_key,
)


def test_variant_selection_key_prefers_recall_over_a_precision_sliver_above_target():
    # both clear the 0.90 target; "spec"-like point has marginally higher
    # precision but collapsed recall (the actual B-spec-vs-B-mixed shape
    # found on real val data -- see phase3_modeling.md), "mixed"-like point
    # has slightly lower precision but far higher recall. F1 must prefer it.
    spec_like = variant_selection_key(precision=0.913, recall=0.125, target_precision=0.90)
    mixed_like = variant_selection_key(precision=0.901, recall=0.976, target_precision=0.90)
    assert mixed_like > spec_like


def test_variant_selection_key_prefers_precision_below_target():
    below_target_high_recall = variant_selection_key(precision=0.60, recall=0.99, target_precision=0.90)
    below_target_higher_precision = variant_selection_key(precision=0.75, recall=0.50, target_precision=0.90)
    assert below_target_higher_precision > below_target_high_recall


def test_status_from_probs_matches_plan_status_logic():
    pA = np.array([0.1, 0.9, 0.9, 0.9])
    pB = np.array([0.5, 0.2, 0.8, 0.8])
    tA, tB = 0.5, 0.5
    status = status_from_probs(pA, pB, tA, tB)
    assert list(status) == ["dry", "wet", "flooded", "flooded"]


def test_sweep_tb_for_precision_reaches_target_when_separable():
    true_label = np.array(["dry"] * 10 + ["flooded"] * 10)
    pA = np.concatenate([np.full(10, 0.9), np.full(10, 0.9)])  # stage A always says "wet surface"
    pB = np.concatenate([np.linspace(0, 0.3, 10), np.linspace(0.7, 1.0, 10)])
    tB, note = sweep_tb_for_precision(true_label, pA, pB, tA=0.5, target_precision=0.90)
    status = status_from_probs(pA, pB, 0.5, tB)
    pred_flooded = status == "flooded"
    tp = np.sum(pred_flooded & (true_label == "flooded"))
    fp = np.sum(pred_flooded & (true_label == "dry"))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    assert precision >= 0.90
    assert "reaching" in note


def test_sweep_tb_for_precision_falls_back_when_unreachable():
    # The top score value is tied across several dry AND flooded rows, so no
    # threshold (a scalar cutoff on pB) can ever isolate a pure-positive
    # subset: every candidate that includes the top group includes false
    # positives too. That makes precision >= 0.90 genuinely unreachable,
    # unlike an i.i.d.-noise draw where the single highest-scoring point can
    # spuriously be a true positive and hit precision 1.0 on a denominator
    # of 1 (exactly the small-sample trap this module flags elsewhere).
    true_label = np.array(["dry"] * 5 + ["flooded"] * 5 + ["dry"] * 90 + ["flooded"] * 0, dtype=object)
    pA = np.full(len(true_label), 0.9)
    pB = np.concatenate([np.full(10, 0.99), np.full(90, 0.1)])
    tB, note = sweep_tb_for_precision(true_label, pA, pB, tA=0.5, target_precision=0.90)
    assert "unreachable" in note
    assert 0.0 <= tB <= 1.0


def test_evaluate_pipeline_flags_small_samples():
    n_dry, n_wet, n_flooded = 40, 4, 40
    true_label = np.array(["dry"] * n_dry + ["wet"] * n_wet + ["flooded"] * n_flooded)
    rows = pd.DataFrame({"label": true_label})
    rng = np.random.default_rng(1)
    pA = np.where(true_label == "dry", rng.uniform(0, 0.3, len(true_label)), rng.uniform(0.7, 1.0, len(true_label)))
    pB = np.where(true_label == "flooded", rng.uniform(0.7, 1.0, len(true_label)), rng.uniform(0, 0.3, len(true_label)))

    report = evaluate_pipeline(rows, pA, pB)
    assert report.false_alarm_wet_den == n_wet
    assert n_wet < SMALL_SAMPLE_WARN_N
    assert any("false_alarm_wet_rate" in flag for flag in report.small_sample_flags)
    assert report.pipeline_precision_flooded >= 0.0
    assert set(report.confusion.keys()) == {"dry", "wet", "flooded"}


def test_evaluate_pipeline_excludes_not_flooded_from_stage_a_and_confusion():
    n_dry, n_wet, n_flooded, n_nf = 40, 40, 40, 40
    true_label = np.array(
        ["dry"] * n_dry + ["wet"] * n_wet + ["flooded"] * n_flooded + ["not_flooded"] * n_nf
    )
    rows = pd.DataFrame({"label": true_label})
    rng = np.random.default_rng(2)
    # not_flooded rows get a wildly wrong (high) pA -- if Stage A's AUC/threshold
    # used them, this would tank stage_a_auc_roc; it must not.
    pA = np.where(
        true_label == "dry", rng.uniform(0, 0.3, len(true_label)),
        np.where(true_label == "not_flooded", rng.uniform(0.9, 1.0, len(true_label)),
                 rng.uniform(0.7, 1.0, len(true_label))),
    )
    pB = np.where(true_label == "flooded", rng.uniform(0.7, 1.0, len(true_label)), rng.uniform(0, 0.3, len(true_label)))

    report = evaluate_pipeline(rows, pA, pB)
    assert set(report.confusion.keys()) == {"dry", "wet", "flooded"}
    assert report.stage_a_accuracy > 0.9  # unaffected by not_flooded's adversarial pA
    # false-flood rate on not_flooded is reported, with the right denominator
    assert report.false_alarm_not_flooded_den == n_nf
    assert 0.0 <= report.false_alarm_not_flooded_rate <= 1.0
    # pipeline precision denominator includes any not_flooded false floods
    n_nf_flooded = np.sum((pA[true_label == "not_flooded"] >= report.tA) & (pB[true_label == "not_flooded"] >= report.tB))
    assert report.pipeline_precision_den >= report.pipeline_recall_num + n_nf_flooded


def test_evaluate_pipeline_accepts_fixed_thresholds():
    n = 30
    true_label = np.array(["dry"] * n + ["flooded"] * n)
    rows = pd.DataFrame({"label": true_label})
    pA = np.where(true_label == "dry", 0.1, 0.9)
    pB = np.where(true_label == "flooded", 0.9, 0.1)
    report = evaluate_pipeline(rows, pA, pB, tA=0.5, tB=0.5)
    assert report.tA == 0.5 and report.tB == 0.5
    assert "fixed" in report.tB_note
    assert report.pipeline_recall_flooded == pytest.approx(1.0)


def test_per_source_breakdown_slices_by_source():
    rows = pd.DataFrame({
        "label": ["dry", "flooded", "flooded", "not_flooded"],
        "source": ["a", "a", "b", "b"],
    })
    pA = np.array([0.1, 0.9, 0.9, 0.9])
    pB = np.array([0.1, 0.9, 0.1, 0.9])
    out = per_source_breakdown(rows, pA, pB, tA=0.5, tB=0.5)
    assert set(out.keys()) == {"a", "b"}
    assert out["a"]["n"] == 2
    assert out["a"]["pipeline_precision_num"] == 1 and out["a"]["pipeline_precision_den"] == 1
    assert out["b"]["false_alarm_not_flooded_den"] == 1
    assert out["b"]["false_alarm_not_flooded_num"] == 1  # the not_flooded row scores flooded


def test_evaluate_pipeline_reports_stage_a_wet_recall():
    n_dry, n_wet, n_flooded = 30, 30, 30
    true_label = np.array(["dry"] * n_dry + ["wet"] * n_wet + ["flooded"] * n_flooded)
    rows = pd.DataFrame({"label": true_label})
    pA = np.where(true_label == "dry", 0.1, 0.9)
    pB = np.where(true_label == "flooded", 0.9, 0.1)
    report = evaluate_pipeline(rows, pA, pB)
    assert report.stage_a_wet_recall_den == n_wet
    assert report.stage_a_wet_recall == pytest.approx(1.0)
