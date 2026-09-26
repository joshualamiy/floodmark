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
    REPORTS,
    ROOT,
    Pipeline,
    abs_path,
    load_pil,
    load_test,
    local,
    model_dir,
    predict_paths,
    preprocess_pil_legacy,
)


def input_paths(df: pd.DataFrame, raw: bool) -> pd.Series:
    # raw = the original file; nysdot raw has a label-leaking header, so keep its crop
    if not raw:
        return df["path"]
    return df["orig_path"].where(df["source"] != "nysdot_road_surface", df["path"])


def run_test(tag: str, raw: bool = False) -> pd.DataFrame:
    df = load_test()
    pipe = Pipeline(model_dir(tag))
    t0 = time.time()
    df["input_path"] = input_paths(df, raw)
    r = predict_paths(pipe, df["input_path"])
    df["pA"], df["pB"], df["status"], df["confidence"] = r["pA"], r["pB"], r["status"], r["confidence"]
    out = local(tag)
    name = "test_preds_raw" if raw else "test_preds"
    df.to_csv(out / f"{name}.csv", index=False)
    if not raw:
        np.savez_compressed(out / "test_cams.npz", camA=r["camA"], camB=r["camB"])
    print(f"{tag} test{' raw' if raw else ''}: {len(df)} rows in {time.time() - t0:.1f}s, spec {pipe.spec}")
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
    ts = pd.to_datetime(f["timestamp_utc"], unit="s", utc=True).dt.tz_convert("America/New_York")
    f["local_time"] = ts.astype(str)
    mins = ts.dt.hour * 60 + ts.dt.minute
    f["day"] = ((mins >= 450) & (mins < 1170)).to_numpy()  # 07:30-19:30 EDT
    return f.reset_index(drop=True)


def run_live(tag: str, snapshot_ts: int | None = None) -> pd.DataFrame:
    f = live_frames(snapshot_ts)
    pipe = Pipeline(model_dir(tag))
    t0 = time.time()
    r = predict_paths(pipe, f["abs_path"])
    f["pA"], f["pB"], f["status"], f["confidence"] = r["pA"], r["pB"], r["status"], r["confidence"]
    out = local(tag)
    f.to_csv(out / "live_preds.csv", index=False)
    np.savez_compressed(out / "live_cams.npz", camA=r["camA"], camB=r["camB"])
    print(f"{tag} live: {len(f)} frames in {time.time() - t0:.1f}s, max ts {f['timestamp_utc'].max()}")
    print(pd.crosstab([f["cam_split"], f["day"]], f["status"]))
    return f


def tf_input(path: str, spec: dict) -> np.ndarray:
    # training-time val pipeline (train.data), run on the processed copy
    import tensorflow as tf

    from train.data import _letterbox_resize, _resize_short_side, _squash_resize

    img = tf.io.decode_jpeg(tf.io.read_file(str(abs_path(path))), channels=3)
    if spec["mode"] == "crop":
        img = _resize_short_side(img, short_side=max(spec["size"], round(spec["size"] * 256 / 224)))
        img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)
        img = tf.image.resize_with_crop_or_pad(img, spec["size"], spec["size"])
    else:
        fn = _squash_resize if spec["mode"] == "squash" else _letterbox_resize
        img = tf.cast(tf.clip_by_value(fn(img, spec["size"]), 0.0, 255.0), tf.uint8)
    return tf.cast(img, tf.float32).numpy()


def run_tf(tag: str) -> pd.DataFrame:
    # sensitivity: same rows through the training-time tf path
    out = local(tag)
    f = pd.read_csv(out / "test_preds.csv", dtype={"camera_id": str, "group_id": str})
    pipe = Pipeline(model_dir(tag))
    r = predict_paths(pipe, f["path"], loader=lambda p: tf_input(p, pipe.spec), workers=1)
    f["pA_tf"], f["pB_tf"], f["status_tf"] = r["pA"], r["pB"], r["status"]
    f.to_csv(out / "test_preds.csv", index=False)
    print(tag, "tf status flips:", int((f["status"] != f["status_tf"]).sum()), "of", len(f),
          "max|dpA|", float(np.abs(f["pA"] - f["pA_tf"]).max()), "max|dpB|", float(np.abs(f["pB"] - f["pB_tf"]).max()))
    return f


def run_parity(tag: str, n: int = 56, seed: int = 0) -> dict:
    # eval loader vs the deployed predict_batch; onnx vs keras
    from inference import load_models, predict_batch

    out = local(tag)
    df = pd.read_csv(out / "test_preds.csv", dtype={"camera_id": str, "group_id": str})
    k = max(1, n // df["source"].nunique())
    samp = df.groupby("source", group_keys=False).apply(lambda d: d.sample(min(len(d), k), random_state=seed))
    pipe = Pipeline(model_dir(tag))
    x = np.stack([pipe.load(p) for p in samp["input_path"]])
    ev = pipe.predict(x)
    m = load_models(model_dir(tag))
    dep = predict_batch([load_pil(p) for p in samp["input_path"]], models=m, heatmap=False)
    pa_d = np.array([p.stage_a_probs["wet"] for p in dep])
    pb_d = np.array([p.stage_b_probs["flooded"] for p in dep])
    res: dict = {"tag": tag, "n": len(samp), "spec": pipe.spec,
                 "eval_vs_deployed_predict": {
                     "max_abs_dpA": float(np.abs(pa_d - ev["pA"]).max()),
                     "max_abs_dpB": float(np.abs(pb_d - ev["pB"]).max()),
                     "status_flips": int(sum(p.status != s for p, s in zip(dep, ev["status"])))}}
    try:
        import keras

        for st, key in (("stage_a", "pA"), ("stage_b", "pB")):
            km = keras.models.load_model(ROOT / "models" / pipe.cfg[st]["run_id"] / "model.keras")
            pk = km.predict(x, batch_size=32, verbose=0)[:, 0]
            res[f"onnx_vs_keras_{st}"] = {"run_id": pipe.cfg[st]["run_id"],
                                          "max_abs_dprob": float(np.abs(pk - ev[key]).max())}
    except (ImportError, OSError, ValueError) as e:
        res["onnx_vs_keras"] = f"skipped: {e}"
    return res


def v1_reproduce() -> dict:
    # does models/v1 + new loader reproduce the first phase 4 run?
    old = pd.read_csv(LOCAL / "test_preds.csv")
    new = pd.read_csv(local("v1") / "test_preds.csv")
    m = old.merge(new, on="orig_path", suffixes=("_old", "_new"))
    pipe = Pipeline(model_dir("v1"))
    leg = predict_paths(pipe, m["path_old"], loader=lambda p: preprocess_pil_legacy(load_pil(p)))
    return {
        "rows_matched": len(m), "rows_old": len(old), "rows_new": len(new),
        "same_label": bool((m["label_old"] == m["label_new"]).all()),
        "legacy_loader_on_models_v1": {"max_abs_dpA": float(np.abs(leg["pA"] - m["pA_old"]).max()),
                                       "max_abs_dpB": float(np.abs(leg["pB"] - m["pB_old"]).max()),
                                       "status_flips": int((leg["status"] != m["status_old"]).sum())},
        "deployed_loader_vs_old": {"max_abs_dpA": float(np.abs(m["pA_new"] - m["pA_old"]).max()),
                                   "mean_abs_dpA": float(np.abs(m["pA_new"] - m["pA_old"]).mean()),
                                   "max_abs_dpB": float(np.abs(m["pB_new"] - m["pB_old"]).max()),
                                   "status_flips": int((m["status_new"] != m["status_old"]).sum()),
                                   "flips": m.loc[m["status_new"] != m["status_old"],
                                                  ["source_old", "label_old", "status_old", "status_new"]]
                                   .astype(str).to_dict("records")},
    }


def parity_all() -> dict:
    res = {t: run_parity(t) for t in ("v1", "v3")}
    res["v1_reproduces_first_run"] = v1_reproduce()
    for t in ("v1", "v3"):
        p = local(t) / "test_preds.csv"
        f = pd.read_csv(p)
        if "status_tf" in f:
            res[t]["tf_training_path_vs_deployed"] = {
                "status_flips": int((f["status"] != f["status_tf"]).sum()), "n": len(f),
                "max_abs_dpA": float(np.abs(f["pA"] - f["pA_tf"]).max()),
                "max_abs_dpB": float(np.abs(f["pB"] - f["pB_tf"]).max())}
        rp = local(t) / "test_preds_raw.csv"
        if rp.exists():
            r = pd.read_csv(rp)
            res[t]["raw_orig_vs_processed_copy"] = {
                "status_flips": int((f["status"] != r["status"]).sum()), "n": len(f),
                "max_abs_dpA": float(np.abs(f["pA"] - r["pA"]).max()),
                "max_abs_dpB": float(np.abs(f["pB"] - r["pB"]).max()),
                "flips_by_source": f[f["status"] != r["status"]]["source"].value_counts().to_dict()}
    (REPORTS / "eval_v3_parity.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str)[:4000])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["test", "raw", "live", "tf", "parity"])
    ap.add_argument("--tag", default="v3", choices=["v1", "v3"])
    ap.add_argument("--snapshot-ts", type=int, default=None)
    a = ap.parse_args()
    if a.what == "test":
        run_test(a.tag)
    elif a.what == "raw":
        run_test(a.tag, raw=True)
    elif a.what == "live":
        run_live(a.tag, a.snapshot_ts)
    elif a.what == "tf":
        run_tf(a.tag)
    else:
        parity_all()
