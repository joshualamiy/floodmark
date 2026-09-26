from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval.common import LOCAL, MANIFEST, MODELS, REPORTS, load_config, load_image
from eval.cues import image_cues, jpeg_quality

CUE_COLS = ["brightness", "dark_frac", "clip_frac", "lap_var", "saturation", "green_frac",
            "text_score", "letterbox", "jpeg_q_orig", "is_png_orig", "aspect", "short_side"]


def manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST, dtype={"camera_id": str, "group_id": str})
    seq = df["orig_path"].astype(str).str.split("/").str[-3]
    blk = df.groupby("source").cumcount() // 13
    unit = np.select([df["source"] == "fred", df["source"] == "flood_master_test"],
                     [seq, "blk" + blk.astype(str)], df["group_id"].astype(str))
    df["cv_group"] = df["source"] + ":" + unit
    return df


def compute_cues(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, o in zip(df["path"], df["orig_path"]):
        c = image_cues(load_image(p))
        c["jpeg_q_orig"] = jpeg_quality(o)
        c["is_png_orig"] = str(o).lower().endswith(".png")
        rows.append(c)
    c = pd.DataFrame(rows, index=df.index)
    c["aspect"] = df["width"] / df["height"]
    c["short_side"] = df[["width", "height"]].min(axis=1)
    return c


def embed_stage_a(paths, batch: int = 64) -> np.ndarray:
    import keras

    m = keras.models.load_model(MODELS / load_config()["stage_a"]["run_id"] / "model.keras")
    bb = m.get_layer("backbone")
    out = []
    for i in range(0, len(paths), batch):
        x = np.stack([load_image(p) for p in paths[i:i + batch]])
        out.append(bb.predict(x, verbose=0).mean(axis=(1, 2)))
    return np.concatenate(out)


def embed_clip(paths, batch: int = 64) -> np.ndarray:
    import open_clip
    import torch
    from PIL import Image

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            ims = [prep(Image.fromarray(load_image(p).astype(np.uint8))) for p in paths[i:i + batch]]
            f = model.encode_image(torch.stack(ims).to(dev))
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)


def source_probe(x: np.ndarray, df: pd.DataFrame, seed: int = 0) -> dict:
    y = df["source"].to_numpy()
    pred = np.empty_like(y)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(x, y, df["cv_group"]):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5,
                                                                 class_weight="balanced"))
        clf.fit(x[tr], y[tr])
        pred[te] = clf.predict(x[te])
    labs = sorted(set(y))
    cm = confusion_matrix(y, pred, labels=labs)
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "per_source_recall": {s: float(cm[i, i] / cm[i].sum()) for i, s in enumerate(labs)},
            "labels": labs, "confusion": cm.tolist(), "n": len(y)}


def auc(y, s):
    y = np.asarray(y, int)
    return None if y.min() == y.max() else float(roc_auc_score(y, s))


def tracks_source(preds: pd.DataFrame, cues: pd.DataFrame) -> dict:
    d = preds.join(cues, how="left")
    out: dict = {}
    out["pA_by_source_label"] = d.groupby(["source", "label"])["pA"].agg(
        ["count", "median", "mean"]).round(4).reset_index().to_dict("records")
    fred = d[d["source"] == "fred"]
    fseq = fred[fred["boot_group"].str.contains("20250811")]
    nys = d[d["source"] == "nysdot_road_surface"]
    out["within_source"] = {
        "fred_all_test_auc_pA_flooded_vs_dry": auc(fred["label"] == "flooded", fred["pA"]),
        "fred_flood_day_sequence_only": {
            "auc_pA": auc(fseq["label"] == "flooded", fseq["pA"]), "n_dry": int((fseq["label"] == "dry").sum()),
            "n_flooded": int((fseq["label"] == "flooded").sum()),
            "dry_called_flooded": int(((fseq["label"] == "dry") & (fseq["status"] == "flooded")).sum()),
            "dry_called_not_dry": int(((fseq["label"] == "dry") & (fseq["status"] != "dry")).sum())},
        "fred_dry_day_sequences_not_dry": int(((fred["label"] == "dry") & ~fred["boot_group"].str.contains(
            "20250811") & (fred["status"] != "dry")).sum()),
        "nysdot_one_camera_auc_pA_wet_vs_dry": auc(nys["label"] == "wet", nys["pA"]),
        "nysdot_wet_called_not_dry": int(((nys["label"] == "wet") & (nys["status"] != "dry")).sum()),
    }
    corr = {}
    for (src, lab), g in d.groupby(["source", "label"]):
        if len(g) < 15:
            continue
        corr[f"{src}|{lab}"] = {c: round(float(spearmanr(g[c].astype(float), g["pA"])[0]), 3)
                                for c in ["brightness", "clip_frac", "lap_var", "saturation",
                                          "green_frac", "text_score"] if g[c].astype(float).std() > 0}
    out["spearman_cue_vs_pA_within_source_label"] = corr
    return out


def cue_tables(df: pd.DataFrame, cues: pd.DataFrame) -> dict:
    d = df.join(cues)
    med = d.groupby("source")[CUE_COLS].median(numeric_only=True).round(3)
    med["letterbox_frac"] = d.groupby("source")["letterbox"].mean().round(3)
    med["png_frac"] = d.groupby("source")["is_png_orig"].mean().round(3)
    tr = d[d["split"] == "train"]
    avail = {}
    for c in CUE_COLS:
        v = tr[c].astype(float)
        if v.notna().all() and v.std() > 0:
            avail[c] = {"auc_flooded": auc(tr["label"] == "flooded", v),
                        "auc_not_dry": auc(tr["label"] != "dry", v)}
    return {"median_by_source": med.to_dict("index"), "train_single_cue_auc": avail}


def source_lookup_baseline(df: pd.DataFrame) -> dict:
    tr = df[df["split"] == "train"]
    maj = tr.groupby("source")["label"].agg(lambda s: s.value_counts().index[0]).to_dict()
    te = df[df["split"] == "test"]
    guess = te["source"].map(maj).fillna("unknown_source")
    return {"majority_label_by_train_source": maj,
            "test_accuracy_known_sources": float((guess == te["label"])[guess != "unknown_source"].mean()),
            "test_rows_from_sources_absent_in_train": int((guess == "unknown_source").sum())}


def main(skip_embed: bool = False) -> dict:
    df = manifest()
    LOCAL.mkdir(parents=True, exist_ok=True)
    cp = LOCAL / "cues_manifest.csv"
    if cp.exists():
        cues = pd.read_csv(cp, index_col=0)
    else:
        cues = compute_cues(df)
        cues.to_csv(cp)
    res: dict = {"n_rows": len(df), "cv": "StratifiedGroupKFold(5) on cv_group (fred sequence, "
                                           "nysdot camera, fmd 13-frame blocks, 511 camera, "
                                           "roadway dup cluster)"}
    for name, fn in (("stageA_gap", embed_stage_a), ("clip_vitb32", embed_clip)):
        ep = LOCAL / f"emb_{name}.npy"
        if not ep.exists():
            if skip_embed:
                continue
            np.save(ep, fn(list(df["path"])))
        res[f"source_probe_{name}"] = source_probe(np.load(ep), df)
    cx = cues[CUE_COLS].astype(float).fillna(-1).to_numpy()
    res["source_probe_simple_cues"] = source_probe(cx, df)
    res["cue_tables"] = cue_tables(df, cues)
    res["source_lookup_baseline"] = source_lookup_baseline(df)
    preds = pd.read_csv(LOCAL / "test_preds.csv")
    test_idx = df.index[df["split"] == "test"]
    tc = cues.loc[test_idx].reset_index(drop=True)
    res["tracks_source"] = tracks_source(preds, tc)
    (REPORTS / "eval_shortcut.json").write_text(json.dumps(res, indent=2, default=float))
    for k, v in res.items():
        if k.startswith("source_probe"):
            print(k, round(v["balanced_accuracy"], 3), {s: round(r, 2) for s, r in v["per_source_recall"].items()})
    print(json.dumps(res["source_lookup_baseline"], indent=1))
    print(json.dumps(res["tracks_source"]["within_source"], indent=1))
    return res




def _perturb(x: np.ndarray, kind: str) -> np.ndarray:
    import io

    from PIL import Image

    if kind == "dark":
        return 255 * (x / 255) ** 2.2 * 0.6
    if kind == "gray":
        g = x @ np.array([0.299, 0.587, 0.114])
        return np.repeat(g[..., None], 3, axis=2)
    if kind == "label_box":
        y = x.copy()
        y[:28, :190] = 0
        y[6:22, 8:182:5] = 255
        return y
    if kind == "ga511_jpeg":
        im = Image.fromarray(x.astype(np.uint8)).resize((253, 253)).resize((224, 224))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=60)
        return np.asarray(Image.open(buf).convert("RGB"), np.float32)
    raise ValueError(kind)


def perturb(kinds=("dark", "gray", "label_box", "ga511_jpeg")) -> dict:
    # synthetic stress test: do true floods survive 511-night-like edits?
    from eval.common import Pipeline

    d = pd.read_csv(LOCAL / "test_preds.csv")
    pipe = Pipeline()
    out = {}
    for lab in ("flooded", "dry"):
        sub = d[d["label"] == lab]
        if lab == "dry":
            sub = sub[sub["source"] != "ga511"]
        x = np.stack([load_image(p) for p in sub["path"]])
        base = pipe.predict(x)["status"]
        for k in kinds:
            st = pipe.predict(np.stack([_perturb(i, k) for i in x]).astype(np.float32))["status"]
            out[f"{lab}|{k}"] = {"n": len(sub), "base_flooded": int((base == "flooded").sum()),
                                 "pert_flooded": int((st == "flooded").sum()),
                                 "base_dry": int((base == "dry").sum()), "pert_dry": int((st == "dry").sum()),
                                 "by_source_pert_flooded": sub.assign(s=st).groupby("source")["s"].apply(
                                     lambda s: int((s == "flooded").sum())).to_dict()}
    (REPORTS / "eval_perturb.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--perturb", action="store_true")
    a = ap.parse_args()
    perturb() if a.perturb else main(a.skip_embed)
