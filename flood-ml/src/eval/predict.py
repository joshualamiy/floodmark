from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from eval.common import (
    CAM_SPLITS,
    FRAMES_CSV,
    LOCAL,
    MODELS,
    REPORTS,
    ROOT,
    Pipeline,
    load_config,
    load_image,
    load_test,
    predict_paths,
)


def run_test() -> pd.DataFrame:
    df = load_test()
    pipe = Pipeline()
    t0 = time.time()
    r = predict_paths(pipe, df["path"])
    df["pA"], df["pB"], df["status"], df["confidence"] = r["pA"], r["pB"], r["status"], r["confidence"]
    LOCAL.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCAL / "test_preds.csv", index=False)
    np.savez_compressed(LOCAL / "test_cams.npz", camA=r["camA"], camB=r["camB"])
    print(f"test: {len(df)} rows in {time.time() - t0:.1f}s")
    print(pd.crosstab(df["label"], df["status"]))
    return df


def live_frames(snapshot_ts: int | None = None) -> pd.DataFrame:
    f = pd.read_csv(FRAMES_CSV)
    f = f[f["dead_reason"].isna() & f["path"].notna()].copy()
    if snapshot_ts:
        f = f[f["timestamp_utc"] <= snapshot_ts]
    sp = json.loads(CAM_SPLITS.read_text())
    f["cam_split"] = f["camera_id"].astype(str).map(sp).fillna("unmapped")
    f["abs_path"] = "data/ga511/" + f["path"]
    lab = pd.read_csv(ROOT / "data/ga511/labels.csv").drop_duplicates("frame_id", keep="last")
    f = f.merge(lab[["frame_id", "label"]].rename(columns={"label": "user_label"}), on="frame_id", how="left")
    ai = pd.read_csv(ROOT / "data/ga511/ai_review_labels.csv").drop_duplicates("frame_id", keep="last")
    f = f.merge(ai[["frame_id", "label"]].rename(columns={"label": "ai_label"}), on="frame_id", how="left")
    return f.reset_index(drop=True)


def run_live(snapshot_ts: int | None = None) -> pd.DataFrame:
    f = live_frames(snapshot_ts)
    pipe = Pipeline()
    t0 = time.time()
    r = predict_paths(pipe, f["abs_path"])
    f["pA"], f["pB"], f["status"], f["confidence"] = r["pA"], r["pB"], r["status"], r["confidence"]
    LOCAL.mkdir(parents=True, exist_ok=True)
    f.to_csv(LOCAL / "live_preds.csv", index=False)
    np.savez_compressed(LOCAL / "live_cams.npz", camA=r["camA"], camB=r["camB"])
    print(f"live: {len(f)} frames in {time.time() - t0:.1f}s, max ts {f['timestamp_utc'].max()}")
    print(pd.crosstab(f["cam_split"], f["status"]))
    return f


def tf_preprocess(path: str) -> np.ndarray:
    # replica of the training-time val pipeline (tf bilinear, uint8 cast, center crop)
    import tensorflow as tf

    img = tf.io.decode_jpeg(tf.io.read_file(str(ROOT / path)), channels=3)
    h, w = tf.cast(tf.shape(img)[0], tf.float32), tf.cast(tf.shape(img)[1], tf.float32)
    s = 256.0 / tf.minimum(h, w)
    img = tf.image.resize(img, [tf.cast(tf.round(h * s), tf.int32), tf.cast(tf.round(w * s), tf.int32)])
    img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)
    img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    return tf.cast(img, tf.float32).numpy()


def run_parity(n: int = 200, seed: int = 0) -> dict:
    import keras

    cfg = load_config()
    df = pd.read_csv(LOCAL / "test_preds.csv")
    samp = df.groupby("source", group_keys=False).apply(
        lambda d: d.sample(min(len(d), max(20, n // 5)), random_state=seed))
    x = np.stack([load_image(p) for p in samp["path"]])
    pipe = Pipeline()
    onnx = pipe.predict(x)
    out: dict = {"n": len(samp)}
    for st, key in (("stage_a", "pA"), ("stage_b", "pB")):
        m = keras.models.load_model(MODELS / cfg[st]["run_id"] / "model.keras")
        pk = m.predict(x, batch_size=64, verbose=0)[:, 0]
        feat = m.get_layer("backbone").predict(x, batch_size=64, verbose=0)
        w = m.get_layer("prob").get_weights()[0][:, 0]
        cam_k = np.maximum(np.einsum("nhwc,c->nhw", feat, w), 0)
        cam_o = onnx["camA" if st == "stage_a" else "camB"]
        out[st] = {"run_id": cfg[st]["run_id"],
                   "max_abs_dprob": float(np.abs(pk - onnx[key]).max()),
                   "max_abs_dcam_rel": float(np.abs(cam_k - cam_o).max() / (np.abs(cam_k).max() + 1e-9))}
    # train/serve skew: PIL bilinear vs training tf pipeline
    xt = np.stack([tf_preprocess(p) for p in samp["path"]])
    rt = pipe.predict(xt)
    out["preproc_skew"] = {
        "max_abs_dpA": float(np.abs(rt["pA"] - onnx["pA"]).max()),
        "max_abs_dpB": float(np.abs(rt["pB"] - onnx["pB"]).max()),
        "mean_abs_dpA": float(np.abs(rt["pA"] - onnx["pA"]).mean()),
        "status_flips": int((rt["status"] != onnx["status"]).sum()),
        "max_abs_pixel": float(np.abs(xt - x).max()),
    }
    (REPORTS / "eval_parity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def run_tf_sensitivity(which: str = "test") -> pd.DataFrame:
    # rerun a whole set with the training-time tf preprocessing
    f = pd.read_csv(LOCAL / f"{which}_preds.csv")
    col = "path" if which == "test" else "abs_path"
    r = predict_paths(Pipeline(), f[col], loader=tf_preprocess)
    f["pA_tf"], f["pB_tf"], f["status_tf"] = r["pA"], r["pB"], r["status"]
    f.to_csv(LOCAL / f"{which}_preds.csv", index=False)
    print(which, "status flips:", int((f["status"] != f["status_tf"]).sum()), "of", len(f))
    print(pd.crosstab(f["status"], f["status_tf"]))
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["test", "live", "parity", "tf_test", "tf_live", "all"])
    ap.add_argument("--snapshot-ts", type=int, default=None)
    a = ap.parse_args()
    if a.what in ("test", "all"):
        run_test()
    if a.what in ("live", "all"):
        run_live(a.snapshot_ts)
    if a.what in ("parity", "all"):
        run_parity()
    if a.what in ("tf_test", "all"):
        run_tf_sensitivity("test")
    if a.what in ("tf_live", "all"):
        run_tf_sensitivity("live")
