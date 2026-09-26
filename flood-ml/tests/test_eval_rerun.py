# v3 rerun: model-aware eval (letterbox parity, not_flooded metrics, smoothing, stress edits)
import numpy as np
import pytest
from PIL import Image

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("cv2")

from eval.common import CROP224, paired_delta, preproc_spec, preprocess, preprocess_pil_legacy
from eval.error_gallery import error_type
from eval.evaluate import set_metrics, smoothed_alerts
from eval.evaluate import test_sets as make_sets
from eval.gradcam_eval import to_letterbox
from eval.stress import crop_area, crop_aspect, edit
from inference.preprocess import preprocess as deployed

LB320 = {"mode": "letterbox", "size": 320, "jpeg": True}


def _img(w, h, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8))


def test_preproc_spec_reads_config():
    assert preproc_spec({"input": {"size": 224}}) == CROP224
    cfg = {"preprocess": {"mode": "letterbox", "size": 320, "jpeg_roundtrip": True}}
    assert preproc_spec(cfg) == LB320


@pytest.mark.parametrize("size", [(450, 253), (640, 480), (1920, 1080), (200, 300)])
def test_letterbox_is_the_deployed_function(size):
    im = _img(*size)
    x = preprocess(im, LB320)
    ref, _ = deployed(im, mode="letterbox", size=320, do_jpeg_roundtrip=True)
    assert x.shape == (320, 320, 3) and np.array_equal(x, ref)


def test_letterbox_pads_gray_on_wide_frames():
    x = preprocess(Image.new("RGB", (450, 253), (10, 200, 30)), LB320)
    assert np.allclose(x[0, 0], 128) and np.allclose(x[-1, -1], 128)
    assert np.allclose(x[160, 160], [10, 200, 30], atol=3)


def test_crop_spec_and_legacy_loader_shapes():
    im = _img(450, 253)
    assert preprocess(im, CROP224).shape == (224, 224, 3)
    assert preprocess_pil_legacy(im).shape == (224, 224, 3)


def test_paired_delta():
    den = np.ones(4, bool)
    r = paired_delta([0, 0, 1, 1], [1, 1, 1, 1], den, list("abcd"))
    assert r["delta"] == 0.5 and r["n"] == 4 and r["n_groups"] == 4
    one = paired_delta([0, 1], [1, 1], [True, True], ["g", "g"])
    assert one["ci_group"][0] == one["ci_group"][1] == 0.5


def test_set_metrics_not_flooded_only_counts_as_flood_negative():
    d = pd.DataFrame({"label": ["dry", "not_flooded", "not_flooded", "flooded", "wet"],
                      "status": ["dry", "flooded", "dry", "flooded", "flooded"],
                      "pA": [0.1, 0.9, 0.1, 0.95, 0.9], "pB": [0.0, 0.9, 0.0, 0.99, 0.95],
                      "boot_group": list("abcde")})
    m = set_metrics(d, 0.8, 0.5)
    assert m["per_class"]["flooded"]["precision"]["n"] == 3  # nf row called flooded counts
    assert m["per_class"]["dry"]["precision"]["n"] == 1  # nf row called dry does not
    assert m["false_flood"]["not_flooded"]["k"] == 1 and m["false_flood"]["not_flooded"]["n"] == 2
    assert m["false_flood"]["wet"]["k"] == 1
    assert m["stage_a"]["n"] == 3  # nf rows excluded from stage A
    assert m["confusion"]["not_flooded"] == {"dry": 1, "wet": 0, "flooded": 1}


def test_test_sets_partition():
    src = ["ga511", "fred", "roadway_flooding", "flood_master_test", "nysdot_road_surface",
           "iowa_rwis", "eu_flood_2013", "alleyfloodnet"]
    df = pd.DataFrame({"source": src})
    s = make_sets(df)
    assert len(s["d_legacy_all"]) == 5 and len(s["d_legacy_external"]) == 4
    assert len(s["d_new_sources"]) == 3 and len(s["d_all_test"]) == 8


def test_smoothed_alerts_uses_deployed_smoother():
    st = ["flooded", "flooded", "flooded", "flooded", "dry", "flooded", "flooded"]
    f = pd.DataFrame({"camera_id": ["1"] * 7 + ["2"] * 2, "timestamp_utc": list(range(9)),
                      "status": st + ["flooded", "flooded"], "local_time": [str(i) for i in range(9)]})
    a = smoothed_alerts(f).set_index("camera_id")
    assert a.loc["1", "alerts"] == 1 and a.loc["1", "longest_run"] == 4
    assert a.loc["2", "alerts"] == 0 and a.loc["2", "raw_flooded"] == 2


def test_to_letterbox_mask_geometry():
    m = np.zeros((256, 455), bool)
    m[:, 228:] = True
    out, content = to_letterbox(m, (455, 256), 320)
    assert out.shape == content.shape == (320, 320)
    assert content[:70].sum() == 0 and content[70:250].all()
    assert out[:, :150].sum() == 0 and out[100, 200]


def test_stress_edits():
    im = _img(640, 480)
    assert abs(np.divide(*crop_aspect(im, 16 / 9).size) - 16 / 9) < 0.01
    c = crop_area(im, 0.75)
    assert abs(c.size[0] * c.size[1] / (640 * 480) - 0.75) < 0.01 and abs(c.size[0] / c.size[1] - 4 / 3) < 0.01
    ov = np.asarray(edit(im, "ga511_overlay"))
    assert ov[0, 0].sum() == 0 and ov.shape == (480, 640, 3)
    bb = np.asarray(edit(im, "band_bottom"))
    assert bb[-1, 0].sum() == 0 and not np.array_equal(bb[0, 0], [0, 0, 0])
    assert np.asarray(edit(im, "dark")).mean() < np.asarray(im).mean()


def test_error_type_not_flooded():
    assert error_type("not_flooded", "flooded").startswith("not_flooded -> flooded")
    assert error_type("not_flooded", "wet") is None
