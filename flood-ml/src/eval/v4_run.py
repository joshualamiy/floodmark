# v4 vs v3: the one test run (manifest test, fresh live test cams, smoothing, leakage, sheets)
from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
from PIL import Image

from eval.common import (
    CAM_SPLITS,
    LOCAL,
    MANIFEST,
    MODELS,
    REPORTS,
    ROOT,
    Pipeline,
    load_config,
    load_pil,
    load_test,
    predict_paths,
    preproc_spec,
    rate,
    thresholds,
)

OUT = LOCAL / "v4"
DIRS = {"v3": MODELS / "v3", "v4": MODELS}
PRED = {"v3": OUT / "v3_rerun", "v4": OUT}
TAGS = ("v3", "v4")
FRAMES = OUT / "frames_snapshot.csv"
OLD_MANIFEST = ROOT / "data/processed/manifest_v1-7251bbd2.csv"
SNAP_TS = 1790454779  # frames.csv frozen, 16:32:59 EDT 9/26
PREV_TS = 1790438064  # previous review cutoff, 11:54:24 EDT
BLOCK = ("11372", "17397", "13750")  # phase4_rerun recommendation
# phash candidates checked by eye (sheets/same_feed_matched_frames.jpg): same feed under two ids
SAME_FEED = ("13547", "14075", "17408")



def _rd(p, **kw) -> pd.DataFrame:
    return pd.read_csv(p, dtype={"camera_id": str, "group_id": str}, **kw)


def snapshot() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src in ("labels.csv", "ai_review_labels.csv"):
        dst = OUT / f"{src[:-4]}_snapshot.csv"
        if not dst.exists():
            shutil.copy2(ROOT / "data/ga511" / src, dst)


# ---------- manifest test ----------

def run_test(tag: str, raw: bool = False) -> pd.DataFrame:
    from eval.predict import input_paths, tf_input

    df = load_test()
    pipe = Pipeline(DIRS[tag])
    t0 = time.time()
    df["input_path"] = input_paths(df, raw)
    r = predict_paths(pipe, df["input_path"])
    for k in ("pA", "pB", "status", "confidence"):
        df[k] = r[k]
    out = PRED[tag]
    out.mkdir(parents=True, exist_ok=True)
    if raw:
        df.to_csv(out / "test_preds_raw.csv", index=False)
    else:
        rt = predict_paths(pipe, df["path"], loader=lambda p: tf_input(p, pipe.spec), workers=1)
        df["pA_tf"], df["pB_tf"], df["status_tf"] = rt["pA"], rt["pB"], rt["status"]
        df.to_csv(out / "test_preds.csv", index=False)
        np.savez_compressed(out / "test_cams.npz", camA=r["camA"], camB=r["camB"])
    print(f"{tag} test{' raw' if raw else ''}: {len(df)} rows {time.time() - t0:.0f}s spec {pipe.spec} "
          f"tA {pipe.ta:.4f} tB {pipe.tb:.4f}", flush=True)
    print(pd.crosstab(df["label"], df["status"]), flush=True)
    return df


def v3_repro() -> dict:
    # models/v3 on today's files vs the v3 run from the previous review
    old, new = _rd(LOCAL / "v3/test_preds.csv"), _rd(PRED["v3"] / "test_preds.csv")
    m = old.merge(new, on="orig_path", suffixes=("_old", "_new"))
    same_order = bool((old["orig_path"].to_numpy() == new["orig_path"].to_numpy()).all())
    res = {"rows_old": len(old), "rows_new": len(new), "rows_matched": len(m), "same_row_order": same_order,
           "same_label": bool((m["label_old"] == m["label_new"]).all()),
           "max_abs_dpA": float(np.abs(m["pA_old"] - m["pA_new"]).max()),
           "max_abs_dpB": float(np.abs(m["pB_old"] - m["pB_new"]).max()),
           "status_flips": int((m["status_old"] != m["status_new"]).sum()),
           "tf_status_flips": int((m["status_tf_old"] != m["status_tf_new"]).sum()),
           "rows_with_new_processed_path": int((m["path_old"] != m["path_new"]).sum()),
           "new_path_rows_by_source": m.loc[m["path_old"] != m["path_new"], "source_new"].value_counts().to_dict()}
    if same_order:
        oc, nc = np.load(LOCAL / "v3/test_cams.npz"), np.load(PRED["v3"] / "test_cams.npz")
        res["max_abs_dcamA"] = float(np.abs(oc["camA"] - nc["camA"]).max())
        res["max_abs_dcamB"] = float(np.abs(oc["camB"] - nc["camB"]).max())
    return res


# ---------- live frames ----------

def live_frames() -> pd.DataFrame:
    f = _rd(FRAMES)
    f = f[f["dead_reason"].isna() & f["path"].notna() & (f["timestamp_utc"] <= SNAP_TS)].copy()
    sp = json.loads(CAM_SPLITS.read_text())
    f["cam_split"] = f["camera_id"].map(sp).fillna("unmapped")
    f["abs_path"] = "data/ga511/" + f["path"]
    lab = pd.read_csv(OUT / "labels_snapshot.csv").drop_duplicates("frame_id", keep="last")
    f = f.merge(lab[["frame_id", "label"]].rename(columns={"label": "user_label"}), on="frame_id", how="left")
    ai = pd.read_csv(OUT / "ai_review_labels_snapshot.csv").drop_duplicates("frame_id", keep="last")
    f = f.merge(ai[["frame_id", "label"]].rename(columns={"label": "ai_label"}), on="frame_id", how="left")
    man = pd.read_csv(MANIFEST, usecols=["orig_path", "split"])
    f = f.merge(man.rename(columns={"orig_path": "abs_path", "split": "manifest_split"}), on="abs_path", how="left")
    f["in_manifest"] = f["manifest_split"].notna()
    ts = pd.to_datetime(f["timestamp_utc"], unit="s", utc=True).dt.tz_convert("America/New_York")
    f["local_time"] = ts.astype(str)
    mins = ts.dt.hour * 60 + ts.dt.minute
    f["day"] = ((mins >= 450) & (mins < 1170)).to_numpy()
    f["fresh"] = (f["timestamp_utc"] > PREV_TS).to_numpy()
    return f.reset_index(drop=True)


def moves(f: pd.DataFrame):
    # deployed CameraMoveDetector, per camera in time order, same as inference.cli
    from inference import CameraMoveDetector

    det = CameraMoveDetector()
    mv, sc = np.zeros(len(f), bool), np.ones(len(f))
    for i in f.sort_values("timestamp_utc").index:
        mv[i], sc[i] = det.update(str(f.at[i, "camera_id"]), load_pil(f.at[i, "abs_path"]))
    return mv, sc


def run_live() -> None:
    f = live_frames()
    t0 = time.time()
    f["moved"], f["move_score"] = moves(f)
    print(f"moves: {int(f['moved'].sum())} of {len(f)} in {time.time() - t0:.0f}s", flush=True)
    for tag in TAGS:
        pipe = Pipeline(DIRS[tag])
        t0 = time.time()
        r = predict_paths(pipe, f["abs_path"])
        g = f.copy()
        for k in ("pA", "pB", "status", "confidence"):
            g[k] = r[k]
        out = PRED[tag]
        out.mkdir(parents=True, exist_ok=True)
        g.to_csv(out / "live_preds_all.csv", index=False)
        np.savez_compressed(out / "live_cams_all.npz", camA=r["camA"], camB=r["camB"])
        keep = ~g["in_manifest"].to_numpy()
        g[keep].to_csv(out / "live_preds.csv", index=False)
        np.savez_compressed(out / "live_cams.npz", camA=r["camA"][keep], camB=r["camB"][keep])
        print(f"{tag} live: {len(g)} frames ({keep.sum()} not in manifest) {time.time() - t0:.0f}s", flush=True)
        print(pd.crosstab([g["cam_split"], g["in_manifest"], g["day"]], g["status"]), flush=True)


def parity(tag: str, n: int = 56, seed: int = 0) -> dict:
    from inference import load_models, predict_batch

    d = _rd(PRED[tag] / "test_preds.csv")
    k = max(1, n // d["source"].nunique())
    s = d.groupby("source", group_keys=False).apply(lambda x: x.sample(min(len(x), k), random_state=seed))
    lv = _rd(PRED[tag] / "live_preds.csv")
    lt = lv[lv["cam_split"] == "test"]
    ls = pd.concat([lt[lt["status"] == "flooded"], lt.sample(30, random_state=seed)])
    paths = list(s["path"]) + list(ls["abs_path"])
    stored_pa = np.concatenate([s["pA"].to_numpy(), ls["pA"].to_numpy()])
    stored_st = np.concatenate([s["status"].to_numpy(), ls["status"].to_numpy()])
    m = load_models(DIRS[tag])
    dep = predict_batch([load_pil(p) for p in paths], models=m, heatmap=False)
    pa = np.array([p.stage_a_probs["wet"] for p in dep])
    return {"n_test": len(s), "n_live": len(ls), "model_version": dep[0].model_version,
            "thresholds": dep[0].thresholds,
            "max_abs_dpA_vs_stored": float(np.abs(pa - stored_pa).max()),
            "status_flips_vs_stored": int(sum(p.status != st for p, st in zip(dep, stored_st)))}


# ---------- metrics ----------

def smooth(f: pd.DataFrame, n: int = 3, block=(), use_moves: bool = False) -> pd.DataFrame:
    from inference.smoothing import TemporalSmoother

    rows = []
    for cam, g in f.sort_values("timestamp_utc").groupby("camera_id"):
        sm, prev, alerts, onsets, skipped = TemporalSmoother(n=n, blocklist=block), None, 0, [], 0
        run = best = 0
        for r in g.itertuples():
            if use_moves and r.moved:
                sm.skip(str(cam))
                skipped += 1
                continue
            s = sm.update(str(cam), r.status).status
            if s == "flooded" and prev != "flooded":
                alerts += 1
                onsets.append(r.local_time)
            prev = s
            run = run + 1 if r.status == "flooded" else 0
            best = max(best, run)
        rows.append({"camera_id": cam, "frames": len(g), "raw_flooded": int((g["status"] == "flooded").sum()),
                     "raw_wet": int((g["status"] == "wet").sum()), "longest_run": best,
                     "skipped_moved": skipped, "alerts": alerts, "alert_onsets": onsets})
    return pd.DataFrame(rows)


def smooth_summary(f: pd.DataFrame, **kw) -> dict:
    sa = smooth(f, **kw)
    if sa.empty:
        return {"config": {k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()}, "cameras": 0}
    return {"config": {"n": kw.get("n", 3), "blocklist": list(kw.get("block", ())),
                       "camera_move_skip": kw.get("use_moves", False)},
            "cameras": len(sa), "frames": int(sa["frames"].sum()),
            "frames_skipped_moved": int(sa["skipped_moved"].sum()),
            "cams_with_any_flooded": int((sa["raw_flooded"] > 0).sum()),
            "cams_with_3plus_flooded": int((sa["raw_flooded"] >= 3).sum()),
            "alerts_total": int(sa["alerts"].sum()), "cams_alerting": int((sa["alerts"] > 0).sum()),
            "alerting": sa[sa["alerts"] > 0].to_dict("records"),
            "top_raw": sa.sort_values("raw_flooded", ascending=False).head(8).to_dict("records")}


def view(f: pd.DataFrame, m) -> dict:
    m = np.asarray(m, bool)
    g = f["camera_id"].astype(str)
    return {"frames": int(m.sum()), "cameras": int(g[m].nunique()),
            "flooded": rate(f["status"] == "flooded", m, g), "wet": rate(f["status"] == "wet", m, g),
            "not_dry": rate(f["status"] != "dry", m, g)}


def live_block(tag: str, same_feed: list[str]) -> dict:
    from eval.evaluate import live_metrics

    f = _rd(PRED[tag] / "live_preds.csv")  # not in manifest
    out = live_metrics(f)
    t = (f["cam_split"] == "test").to_numpy()
    fr, day = f["fresh"].to_numpy(bool), f["day"].to_numpy(bool)
    for k, m in (("fresh_day", t & fr & day), ("fresh_night", t & fr & ~day), ("fresh_all", t & fr),
                 ("prefresh_day", t & ~fr & day), ("prefresh_night", t & ~fr & ~day)):
        out[f"test_cams|{k}"] = view(f, m)
    sf = f["camera_id"].isin(same_feed).to_numpy()
    for k, m in (("day", t & day), ("fresh_day", t & fr & day)):
        out[f"test_cams_excl_same_feed|{k}"] = view(f, m & ~sf)
    unu = (f["user_label"] == "unusable").to_numpy()
    out["test_cams_excl_user_unusable|fresh_day"] = view(f, t & fr & day & ~unu)
    ft = f[t]
    a = _rd(PRED[tag] / "live_preds_all.csv")
    at = a[a["cam_split"] == "test"]
    out["smoothing"] = {
        "test_fresh_set|n3_as_prev_replay": smooth_summary(ft),
        "test_fresh_set|n3_blocklist_moves": smooth_summary(ft, block=BLOCK, use_moves=True),
        "test_fresh_set|n3_moves_no_blocklist": smooth_summary(ft, use_moves=True),
        "test_full_sequence_incl_manifest|n3_as_prev_replay": smooth_summary(at),
        "test_full_sequence_incl_manifest|n3_blocklist_moves": smooth_summary(at, block=BLOCK, use_moves=True),
        "all_cams_fresh_set|n3_blocklist_moves": {
            k: v for k, v in smooth_summary(f, block=BLOCK, use_moves=True).items() if k != "top_raw"},
    }
    out["smoothed_alerts"] = {k: v.get("alerts_total") for k, v in out["smoothing"].items()}
    out["moved_frames_test_fresh_set"] = int(ft["moved"].sum())
    out["snapshot_ts"] = SNAP_TS
    out["fresh_after_ts"] = PREV_TS
    return out


def prev_set_repro() -> dict:
    # rebuild the previous review's 1,602 test-cam frames and score both models on them
    old = _rd(LOCAL / "v3/live_preds.csv")
    ot = old[old["cam_split"] == "test"]
    res = {"prev_frames": len(ot), "prev_v3_flooded_day": int(((ot["status"] == "flooded") & ot["day"]).sum()),
           "prev_v3_flooded_night": int(((ot["status"] == "flooded") & ~ot["day"]).sum())}
    for tag in TAGS:
        a = _rd(PRED[tag] / "live_preds_all.csv")
        m = ot[["frame_id", "pA", "pB", "status"]].merge(a, on="frame_id", suffixes=("_prev", ""))
        res[tag] = {"matched": len(m), "flooded_day": int(((m["status"] == "flooded") & m["day"]).sum()),
                    "flooded_night": int(((m["status"] == "flooded") & ~m["day"]).sum()),
                    "wet_day": int(((m["status"] == "wet") & m["day"]).sum()),
                    "wet_night": int(((m["status"] == "wet") & ~m["day"]).sum())}
        if tag == "v3":
            res[tag]["max_abs_dpA_vs_prev"] = float(np.abs(m["pA"] - m["pA_prev"]).max())
            res[tag]["status_flips_vs_prev"] = int((m["status"] != m["status_prev"]).sum())
    return res


def greek(preds: dict, lives: dict) -> dict:
    out = {}
    for tag in TAGS:
        d = preds[tag][preds[tag]["source"] == "flood_master_test"]
        out[tag] = {"n": len(d), "status": d["status"].value_counts().to_dict(),
                    "median_pA": float(d["pA"].median()), "median_pB": float(d["pB"].median()),
                    "median_pApB": float((d["pA"] * d["pB"]).median())}
    # ranking at matched live false floods (sensitivity only, not a threshold choice)
    rows = []
    for k in (2, 5, 9, 20):
        r = {"live_test_false_floods": k}
        for tag in TAGS:
            d, lv = preds[tag], lives[tag]
            ls = np.sort((lv.loc[lv["cam_split"] == "test", "pA"] * lv.loc[lv["cam_split"] == "test", "pB"]).to_numpy())[::-1]
            thr = ls[k] if k < len(ls) else 0.0
            g = (d.loc[d["source"] == "flood_master_test", "pA"] * d.loc[d["source"] == "flood_master_test", "pB"]).to_numpy()
            r[f"{tag}_greek_recall"] = f"{int((g > thr).sum())}/{len(g)}"
        rows.append(r)
    out["matched_live_false_floods"] = rows
    return out


def flips(p3: pd.DataFrame, p4: pd.DataFrame) -> dict:
    assert (p3["orig_path"].to_numpy() == p4["orig_path"].to_numpy()).all()
    d = p3[["source", "label"]].copy()
    d["v3"], d["v4"] = p3["status"].to_numpy(), p4["status"].to_numpy()
    ch = d[d["v3"] != d["v4"]]
    fl = d["label"] == "flooded"
    return {"n_changed": len(ch),
            "by_source_label_v3_to_v4": {f"{a}|{b}|{c}->{e}": int(v) for (a, b, c, e), v in
                                         ch.groupby(["source", "label", "v3", "v4"]).size().items()},
            "floods_lost": int((fl & (d["v3"] == "flooded") & (d["v4"] != "flooded")).sum()),
            "floods_gained": int((fl & (d["v3"] != "flooded") & (d["v4"] == "flooded")).sum()),
            "false_floods_removed": int((~fl & (d["v3"] == "flooded") & (d["v4"] != "flooded")).sum()),
            "false_floods_added": int((~fl & (d["v3"] != "flooded") & (d["v4"] == "flooded")).sum())}


def metrics() -> dict:
    from eval.evaluate import paired, set_metrics, test_sets, threshold_whatif

    res: dict = {"sets": {}, "thresholds": {}, "preprocess": {}}
    preds = {t: _rd(PRED[t] / "test_preds.csv") for t in TAGS}
    for t in TAGS:
        cfg = load_config(DIRS[t])
        ta, tb = thresholds(cfg)
        res["thresholds"][t] = {"tA": ta, "tB": tb, "run_a": cfg["stage_a"]["run_id"],
                                "run_b": cfg["stage_b"]["run_id"], "data_version": cfg.get("data_version"),
                                "model_dir": str(DIRS[t].relative_to(ROOT))}
        res["preprocess"][t] = cfg.get("preprocess")
        for k, v in test_sets(preds[t]).items():
            res["sets"].setdefault(k, {})[t] = set_metrics(v, ta, tb)
    s3, s4 = test_sets(preds["v3"]), test_sets(preds["v4"])
    res["paired_v4_minus_v3"] = {k: paired(s3[k], s4[k]) for k in s3}
    res["test_flips_v3_to_v4"] = flips(preds["v3"], preds["v4"])
    same_feed = list(SAME_FEED)
    res["live"] = {t: live_block(t, same_feed) for t in TAGS}
    res["live_same_feed_cams_excluded"] = same_feed
    res["live_prev_1602_set"] = prev_set_repro()
    lives = {t: _rd(PRED[t] / "live_preds.csv") for t in TAGS}
    res["greek"] = greek(preds, lives)
    for t in TAGS:
        th = res["thresholds"][t]
        res[f"threshold_whatif_{t}"] = threshold_whatif(preds[t], lives[t], th["tA"], th["tB"])
    res["threshold_whatif_note"] = "sensitivity only; test is never used to pick thresholds"
    for t in TAGS:
        rp = PRED[t] / "test_preds_raw.csv"
        if rp.exists():
            d = _rd(rp)
            th = res["thresholds"][t]
            res.setdefault("raw_input_sensitivity", {})[t] = {
                k: {"flood_recall": set_metrics(v, th["tA"], th["tB"])["per_class"]["flooded"]["recall"]}
                for k, v in test_sets(d).items() if k.startswith("d_")}
            res["raw_input_sensitivity"][t]["status_flips_vs_processed"] = int(
                (d["status"].to_numpy() != preds[t]["status"].to_numpy()).sum())
    res["tf_path_status_flips"] = {t: int((preds[t]["status"] != preds[t]["status_tf"]).sum()) for t in TAGS}
    res["v3_reproduces_previous_review"] = v3_repro()
    res["parity_deployed_predict_batch"] = {t: parity(t) for t in TAGS}
    res["figure"] = None
    (OUT / "eval_v4_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    return res


# ---------- stress (511GA overlay, darkening) ----------

def real_overlays(k: int = 8, seed: int = 0) -> list:
    # cut the real title bar + logo box out of train-camera day frames
    f = live_frames()
    f = f[(f["cam_split"] == "train") & f["day"] & (f["width"] == 450)].sample(300, random_state=seed)
    outs = []
    for p in f["abs_path"]:
        a = np.asarray(load_pil(p), np.int16)
        h, w, _ = a.shape
        mx = a.max(-1)
        bright = a.min(-1) > 190
        # bar = dark rows at top-left (text strokes are thin, so medians stay dark)
        bh = 0
        while bh < h // 4 and np.median(mx[bh, :120]) < 60:
            bh += 1
        if not 12 <= bh <= 30:
            continue
        col_dark = np.median(mx[1:bh - 1], axis=0) < 60
        bw = 0
        while bw < w - 8 and col_dark[bw:bw + 8].any():
            bw += 1
        if not 150 <= bw <= int(0.85 * w):
            continue
        lg = bright[: h // 5, int(0.8 * w):]
        rows, cols = np.flatnonzero(lg.mean(1) > 0.4), np.flatnonzero(lg.mean(0) > 0.4)
        if not len(rows) or not len(cols):
            continue
        box = (int(0.8 * w) + cols[0], rows[0], int(0.8 * w) + cols[-1] + 1, rows[-1] + 1)
        im = Image.fromarray(a.astype(np.uint8))
        outs.append({"w": w, "bar": im.crop((0, 0, bw, bh)), "logo": im.crop(box), "logo_xy": box[:2], "src": p})
        if len(outs) >= k:
            break
    return outs


def paste_real(im: Image.Image, ov: dict) -> Image.Image:
    im = im.copy()
    s = im.size[0] / ov["w"]
    bar = ov["bar"].resize((max(1, round(ov["bar"].size[0] * s)), max(1, round(ov["bar"].size[1] * s))))
    logo = ov["logo"].resize((max(1, round(ov["logo"].size[0] * s)), max(1, round(ov["logo"].size[1] * s))))
    im.paste(bar, (0, 0))
    im.paste(logo, (round(ov["logo_xy"][0] * s), round(ov["logo_xy"][1] * s)))
    return im


def run_stress() -> dict:
    from eval.stress import darken, run_set, summarize
    from eval.viz import sheet

    ovs = real_overlays()
    peek = [paste_real(load_pil(p), ovs[i % len(ovs)]).resize((480, 270)) for i, p in enumerate(
        _rd(PRED["v4"] / "test_preds.csv").query("source == 'flood_master_test'")["path"].iloc[:4])]
    sheet(peek, OUT / "sheets/real_overlay_peek.jpg", cols=2)
    res: dict = {"real_overlay_sources": [o["src"] for o in ovs]}
    for tag in TAGS:
        pipe = Pipeline(DIRS[tag])
        d = _rd(PRED[tag] / "test_preds.csv")
        r: dict = {}
        for name, sub, kinds in (
                ("flooded", d[d["label"] == "flooded"],
                 ("dark", "ga511_overlay", "dark+overlay", "band_top", "band_bottom", "gray_band_top", "logo")),
                ("dry_non_ga511", d[(d["label"] == "dry") & (d["source"] != "ga511")], ("dark", "ga511_overlay")),
                ("not_flooded", d[d["label"] == "not_flooded"], ("dark", "ga511_overlay")),
                ("wet", d[d["label"] == "wet"], ("dark", "ga511_overlay"))):
            out = run_set(pipe, sub["path"], kinds)
            ims = [load_pil(p) for p in sub["path"]]
            out["real_511_overlay"] = pipe.predict(np.stack(
                [pipe.prep(paste_real(im, ovs[i % len(ovs)])) for i, im in enumerate(ims)]))
            out["real_511_overlay+dark"] = pipe.predict(np.stack(
                [pipe.prep(paste_real(darken(im), ovs[i % len(ovs)])) for i, im in enumerate(ims)]))
            r[name] = summarize(sub, out)
        res[tag] = r
        print(tag, json.dumps({s: {k: (v["flooded"], v["dry"]) for k, v in x.items()} for s, x in r.items()}),
              flush=True)
    (OUT / "eval_v4_perturb.json").write_text(json.dumps(res, indent=2))
    return res


def energy() -> dict:
    from eval import gradcam_eval

    gradcam_eval.REPORTS = OUT  # keep reports/ untouched
    return gradcam_eval.mask_energy("v4")


# ---------- leakage ----------

def _ph(paths) -> np.ndarray:
    import imagehash

    from eval.leakage import ph_int

    return ph_int([str(imagehash.phash(load_pil(p))) for p in paths])


def leakage() -> dict:
    from eval.leakage import hamming_pairs, ph_int

    new, old = _rd(MANIFEST), _rd(OLD_MANIFEST)
    sp = json.loads(CAM_SPLITS.read_text())
    test_cams = {c for c, s in sp.items() if s == "test"}
    seen = new[new["split"].isin(["train", "val"])]
    added = seen[~seen["orig_path"].isin(old["orig_path"])].copy()
    fr = _rd(FRAMES)
    fr = fr[fr["dead_reason"].isna() & fr["path"].notna() & (fr["timestamp_utc"] <= SNAP_TS)].copy()
    fr["abs_path"] = "data/ga511/" + fr["path"]
    fr["cs"] = fr["camera_id"].map(sp).fillna("unmapped")
    added = added.merge(fr[["abs_path", "timestamp_utc", "view_id"]].rename(columns={"abs_path": "orig_path"}),
                        on="orig_path", how="left")
    ts = pd.to_datetime(added["timestamp_utc"], unit="s", utc=True).dt.tz_convert("America/New_York")
    mins = ts.dt.hour * 60 + ts.dt.minute
    added["day"] = ((mins >= 450) & (mins < 1170)).to_numpy()
    res: dict = {"added_rows": len(added),
                 "added_by_source_split": {f"{a}|{b}": int(v) for (a, b), v in
                                           added.groupby(["source", "split"]).size().items()},
                 "added_day": int(added["day"].sum()),
                 "added_by_label_source": added["label_source"].value_counts().to_dict()}
    tv_cams = set(seen["camera_id"].dropna())
    tv_views = set(fr.loc[fr["cs"].isin(["train", "val"]), "view_id"].astype(str))
    te_views = set(fr.loc[fr["cs"] == "test", "view_id"].astype(str))
    res["camera_overlap"] = {
        "trainval_manifest_cams_that_are_test_cams": sorted(tv_cams & test_cams),
        "added_rows_on_test_cams": int(added["camera_id"].isin(test_cams).sum()),
        "test_manifest_cams_in_trainval_manifest": sorted(set(new.loc[new["split"] == "test", "camera_id"].dropna())
                                                          & tv_cams),
        "view_ids_shared_test_vs_trainval_cams": sorted(tv_views & te_views),
        "manifest_camera_ids_spanning_splits": int((new.groupby("camera_id")["split"].nunique() > 1).sum()),
    }
    # phash, one implementation for all images
    t0 = time.time()
    a_ph = _ph(added["orig_path"])
    tm = new[new["split"] == "test"].reset_index(drop=True)
    tm_paths = tm["orig_path"].where(tm["source"] == "ga511", tm["path"])
    tm_ph = _ph(tm_paths)
    lv = _rd(PRED["v4"] / "live_preds.csv")
    lt = lv[lv["cam_split"] == "test"].reset_index(drop=True)
    lt_ph = _ph(lt["abs_path"])
    print(f"phash {len(a_ph) + len(tm_ph) + len(lt_ph)} imgs {time.time() - t0:.0f}s", flush=True)
    for name, b, bdf, pcol in (("manifest_test", tm_ph, tm, "orig_path"), ("fresh_test_cam_frames", lt_ph, lt, "abs_path")):
        for thr in (6, 10):
            pairs = hamming_pairs(a_ph, b, thr=thr)
            ex = [{"train": added.iloc[i]["orig_path"], "train_cam": added.iloc[i]["camera_id"],
                   "train_day": bool(added.iloc[i]["day"]), "test": bdf.iloc[j][pcol],
                   "test_cam": bdf.iloc[j]["camera_id"], "ham": dd} for i, j, dd in pairs]
            e = pd.DataFrame(ex)
            res[f"new_trainval_vs_{name}|ham<={thr}"] = {
                "pairs": len(pairs), "pairs_day_train": int(e["train_day"].sum()) if len(e) else 0,
                "distinct_test": int(e["test"].nunique()) if len(e) else 0,
                "test_cams": sorted(e["test_cam"].dropna().unique().tolist()) if len(e) else [],
                "cam_pairs": (e["train_cam"] + "~" + e["test_cam"]).value_counts().head(20).to_dict() if len(e) else {},
                "examples": ex[:10]}
    # same feed under two ids: collector phash, every test-cam frame vs every train/val-cam frame
    t, s = fr[fr["cs"] == "test"].reset_index(drop=True), fr[fr["cs"].isin(["train", "val"])].reset_index(drop=True)
    pairs = hamming_pairs(ph_int(t["phash"]), ph_int(s["phash"]), thr=6)
    e = pd.DataFrame([{"test_cam": t.at[i, "camera_id"], "other_cam": s.at[j, "camera_id"],
                       "other_split": s.at[j, "cs"], "ham": dd,
                       "dt_min": abs(int(t.at[i, "timestamp_utc"]) - int(s.at[j, "timestamp_utc"])) / 60}
                      for i, j, dd in pairs])
    cp = (e.groupby(["test_cam", "other_cam", "other_split"]).agg(n=("ham", "size"), min_ham=("ham", "min"),
                                                                  min_dt_min=("dt_min", "min")).reset_index()
          if len(e) else pd.DataFrame())
    # a real shared feed repeats: same-time matches, several of them
    feed = cp[(cp["n"] >= 2) & (cp["min_dt_min"] <= 30)] if len(cp) else cp
    res["cross_camera_phash_le6"] = {"frame_pairs": len(e), "camera_pairs": cp.to_dict("records"),
                                     "camera_pairs_total": len(cp)}
    res["same_feed_candidates_auto"] = sorted(feed["test_cam"].unique().tolist()) if len(feed) else []
    res["same_feed_test_cams"] = list(SAME_FEED)
    res["same_feed_note"] = ("auto candidates checked by eye: 13547/17556, 14075/14259, 17408/13479 share a feed; "
                             "15327, 16067, 16439 are dawn look-alikes; 13351 is a 'Camera error' screen")
    cams = fr.groupby("camera_id").agg(lat=("lat", "first"), lon=("lon", "first"), cs=("cs", "first"))
    tc, oc = cams[cams["cs"] == "test"], cams[cams["cs"].isin(["train", "val"])]
    dlat = (tc["lat"].to_numpy()[:, None] - oc["lat"].to_numpy()[None]) * 111_000
    dlon = (tc["lon"].to_numpy()[:, None] - oc["lon"].to_numpy()[None]) * 111_000 * np.cos(np.radians(33.75))
    close = np.argwhere(np.hypot(dlat, dlon) < 30)
    res["test_cams_within_30m_of_trainval_cam"] = [
        {"test_cam": tc.index[i], "other_cam": oc.index[j], "other_split": oc.iloc[j]["cs"]} for i, j in close]
    (OUT / "eval_v4_leakage.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "examples"})
                      for k, v in res.items()}, default=str)[:4000], flush=True)
    return res


# ---------- visual check ----------

def _strip(path: str, cam_a, cam_b, spec: dict, cap: str, h: int = 240) -> Image.Image:
    from PIL import ImageDraw

    from eval.common import preprocess_geom
    from eval.viz import overlay

    raw = load_pil(path)
    arr, geom = preprocess_geom(raw, spec)
    l, t, r, b = geom["content_box_canvas"]
    img = arr.astype(np.uint8)
    parts = [np.asarray(raw.resize((round(raw.size[0] * h / raw.size[1]), h)))]
    for cam in (cam_a, cam_b):
        o = Image.fromarray(overlay(img, cam)[t:b, l:r])
        parts.append(np.asarray(o.resize((round(o.size[0] * h / o.size[1]), h))))
    strip = Image.fromarray(np.concatenate(parts, axis=1))
    c = Image.new("RGB", (strip.width, h + 18), (250, 250, 250))
    c.paste(strip, (0, 0))
    ImageDraw.Draw(c).text((4, h + 3), cap[:190], fill=(0, 0, 0))
    return c


def sheets(per: int = 6) -> dict:
    from eval.viz import sheet

    out = {}
    for tag in TAGS:
        f = _rd(PRED[tag] / "live_preds.csv")
        cams = np.load(PRED[tag] / "live_cams.npz")
        spec = preproc_spec(load_config(DIRS[tag]))
        for st in ("flooded", "wet"):
            sub = f[(f["cam_split"] == "test") & (f["status"] == st)].sort_values(["camera_id", "timestamp_utc"])
            out[f"{tag}_{st}"] = len(sub)
            tl = [_strip(r["abs_path"], cams["camA"][i], cams["camB"][i], spec,
                         f"{tag} {st} cam {r['camera_id']} {r['local_time'][5:16]} {'DAY' if r['day'] else 'night'}"
                         f"{' FRESH' if r['fresh'] else ''} pA {r['pA']:.2f} pB {r['pB']:.2f} "
                         f"user {r['user_label']} moved {r['moved']}  [raw | CAM A | CAM B]")
                  for i, r in sub.iterrows()]
            for n, s in enumerate(range(0, len(tl), per)):
                sheet(tl[s:s + per], OUT / f"sheets/{tag}_live_test_{st}_{n}.jpg", cols=1)
        # test-split false floods on dry/wet ga511 + not_flooded, for the record
        d = _rd(PRED[tag] / "test_preds.csv")
        tc = np.load(PRED[tag] / "test_cams.npz")
        sub = d[(d["status"] == "flooded") & (d["label"] != "flooded")]
        out[f"{tag}_test_false_floods"] = len(sub)
        tl = [_strip(r["path"], tc["camA"][i], tc["camB"][i], spec,
                     f"{tag} {r['source']} {r['label']}->flooded grp {r['boot_group']} pA {r['pA']:.2f} pB {r['pB']:.2f}")
              for i, r in sub.iterrows()]
        for n, s in enumerate(range(0, len(tl), per)):
            sheet(tl[s:s + per], OUT / f"sheets/{tag}_test_false_floods_{n}.jpg", cols=1)
    print(out, flush=True)
    return out


def gallery() -> dict:
    from eval import error_gallery

    return error_gallery.build(out=REPORTS / "errors.html", tag="v4")


STEPS = {"test": lambda: [run_test(t) for t in TAGS], "raw": lambda: [run_test(t, raw=True) for t in TAGS],
         "live": run_live, "leakage": leakage, "metrics": metrics, "stress": run_stress, "energy": energy,
         "sheets": sheets, "gallery": gallery}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=list(STEPS))
    a = ap.parse_args()
    snapshot()
    for st in a.steps:
        t0 = time.time()
        print(f"== {st}", flush=True)
        STEPS[st]()
        print(f"== {st} done {time.time() - t0:.0f}s", flush=True)
