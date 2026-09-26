import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("cv2")

from eval.error_gallery import error_type
from eval.evaluate import auc_safe, confusion, set_metrics
from eval.leakage import hamming_pairs, ph_int


def test_confusion_keeps_all_classes():
    cm = confusion(["dry", "dry"], ["dry", "flooded"])
    assert cm.shape == (3, 3) and cm.to_numpy().sum() == 2


def test_auc_safe_single_class():
    assert auc_safe([1, 1], [0.2, 0.3])["auc_roc"] is None
    assert auc_safe([], [])["auc_roc"] is None


def test_set_metrics_counts():
    d = pd.DataFrame({"label": ["dry", "dry", "wet", "flooded", "flooded"],
                      "status": ["dry", "flooded", "dry", "flooded", "wet"],
                      "pA": [0.1, 0.9, 0.2, 0.95, 0.9], "pB": [0.0, 0.95, 0.1, 0.99, 0.5],
                      "boot_group": list("abcde")})
    m = set_metrics(d, 0.8, 0.9)
    assert m["false_flood"]["dry"]["k"] == 1 and m["false_flood"]["dry"]["n"] == 2
    assert m["per_class"]["flooded"]["recall"]["k"] == 1
    assert m["missed_flood"]["wet|flooded"]["k"] == 1
    assert m["stage_a"]["confusion"] == [[1, 1], [1, 2]]


def test_error_type():
    assert error_type("wet", "flooded").startswith("wet -> flooded")
    assert error_type("flooded", "dry").endswith("(missed flood)")
    assert error_type("dry", "dry") is None


def test_hamming_pairs():
    a = ph_int(["0000000000000000", "ffffffffffffffff"])
    b = ph_int(["000000000000003f", "0000000000000fff"])
    pairs = hamming_pairs(a, b, thr=6)
    assert pairs == [(0, 0, 6)]
    assert np.bitwise_count(a[1] ^ b[1]) == 52
