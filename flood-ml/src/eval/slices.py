# failure slices: night, blur, glare, moved cameras
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from eval.common import LOCAL, MANIFEST, REPORTS, load_image, rate
from eval.cues import image_cues, tag_slices
from eval.viz import sheet, tile

ROAD = "a traffic camera view of a road or highway"
ODD = ["a blurry out-of-focus image", "a close-up of trees, bushes or grass",
       "a view of mostly sky", "a camera pointed at a pole, wall or bridge underside",
       "a blank gray or black screen", "a field with a house and no road"]
TAGS = ["night", "glare", "blurry", "grayscale", "overlay_text", "odd_view", "view_changed"]


def clip_odd(paths=None, emb: np.ndarray | None = None) -> np.ndarray:
    import open_clip
    import torch
    from PIL import Image

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(dev).eval()
    with torch.no_grad():
        t = model.encode_text(tok([ROAD] + ODD).to(dev))
        t = torch.nn.functional.normalize(t, dim=-1).cpu().numpy()
        if emb is None:
            out = []
            for i in range(0, len(paths), 64):
                ims = [prep(Image.fromarray(load_image(p).astype(np.uint8))) for p in paths[i:i + 64]]
                f = model.encode_image(torch.stack(ims).to(dev))
                out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
            emb = np.concatenate(out)
    logits = 100 * emb @ t.T
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return p[:, 0]


def tag_frame(df: pd.DataFrame, cues: pd.DataFrame, p_road: np.ndarray) -> pd.DataFrame:
    t = pd.DataFrame([tag_slices(r) for r in cues.to_dict("records")], index=df.index)
    t["odd_view"] = p_road < 0.5
    ga = (df["source"] == "ga511") if "source" in df else pd.Series(True, index=df.index)
    t.loc[ga, "night"] = True
    t["glare"] = (cues["clip_frac"] > 0.005) & t["night"]
    t.loc[~ga, "overlay_text"] = False
    return t


def slice_table(d: pd.DataFrame, tags: pd.DataFrame, group_col: str, true_col: str | None) -> dict:
    out = {}
    g = d[group_col].astype(str)
    for tag in [c for c in TAGS if c in tags]:
        for on in (True, False):
            m = (tags[tag] == on).to_numpy()
            key = f"{tag}={'yes' if on else 'no'}"
            r = {"n": int(m.sum())}
            if true_col is None:
                r["flooded"] = rate(d["status"] == "flooded", m, g)
                r["not_dry"] = rate(d["status"] != "dry", m, g)
            else:
                y = d[true_col]
                r["fa_flooded|dry"] = rate(d["status"] == "flooded", m & (y == "dry").to_numpy(), g)
                r["fa_notdry|dry"] = rate(d["status"] != "dry", m & (y == "dry").to_numpy(), g)
                r["recall_flooded"] = rate(d["status"] == "flooded", m & (y == "flooded").to_numpy(), g)
                r["wet_to_notdry"] = rate(d["status"] != "dry", m & (y == "wet").to_numpy(), g)
                r["by_source_label"] = d[m].groupby(["source", true_col]).size().astype(int).to_dict()
                r["by_source_label"] = {f"{a}|{b}": v for (a, b), v in r["by_source_label"].items()}
            out[key] = r
    return out


def view_changed(f: pd.DataFrame, thr: int = 22) -> pd.Series:
    first = f.sort_values("timestamp_utc").groupby("camera_id")["phash"].first()
    ref = f["camera_id"].map(first)
    ham = [(int(a, 16) ^ int(b, 16)).bit_count() for a, b in zip(f["phash"], ref)]
    return pd.Series(np.array(ham) >= thr, index=f.index)


def main() -> dict:
    res: dict = {}
    man = pd.read_csv(MANIFEST, dtype={"camera_id": str})
    test_idx = man.index[man["split"] == "test"]
    cues = pd.read_csv(LOCAL / "cues_manifest.csv", index_col=0).loc[test_idx].reset_index(drop=True)
    d = pd.read_csv(LOCAL / "test_preds.csv")
    emb = np.load(LOCAL / "emb_clip_vitb32.npy")[test_idx]
    pr = clip_odd(emb=emb)
    tags = tag_frame(d, cues, pr)
    tags["p_road_clip"] = pr
    tags.join(d[["path", "source", "label", "status"]]).to_csv(LOCAL / "test_slices.csv")
    res["test_prevalence"] = {t: {k: int(v) for k, v in tags.groupby(d["source"])[t].sum().items()}
                              for t in TAGS if t in tags}
    res["test"] = slice_table(d, tags, "boot_group", "label")
    f = pd.read_csv(LOCAL / "live_preds.csv")
    lc = LOCAL / "cues_live.csv"
    fc = pd.read_csv(lc, index_col=0) if lc.exists() else None
    if fc is None or fc.get("frame_id", pd.Series()).tolist() != f["frame_id"].tolist():
        fc = pd.DataFrame([image_cues(load_image(p)) for p in f["abs_path"]], index=f.index)
        fc["p_road_clip"] = clip_odd(paths=list(f["abs_path"]))
        fc["frame_id"] = f["frame_id"]
        fc.to_csv(lc)
    fc = fc.drop(columns=["frame_id"])
    ft = tag_frame(f, fc.drop(columns=["p_road_clip"]), fc["p_road_clip"].to_numpy())
    ft["view_changed"] = view_changed(f)
    ft.join(f[["abs_path", "camera_id", "cam_split", "status"]]).to_csv(LOCAL / "live_slices.csv")
    res["live_prevalence"] = {t: int(ft[t].sum()) for t in TAGS}
    res["live_all"] = slice_table(f, ft, "camera_id", None)
    tm = (f["cam_split"] == "test").to_numpy()
    res["live_test_cams"] = slice_table(f[tm], ft[tm], "camera_id", None)
    res["cams_with_multiple_frames"] = int((f.groupby("camera_id").size() > 1).sum())
    (REPORTS / "eval_slices.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res["test_prevalence"]), json.dumps(res["live_prevalence"]))
    return res


def eye_check(n: int = 8, seed: int = 0) -> None:
    for name, csv, pcol in (("test", "test_slices.csv", "path"), ("live", "live_slices.csv", "abs_path")):
        t = pd.read_csv(LOCAL / csv, index_col=0)
        for tag in TAGS:
            if tag not in t:
                continue
            for on in (True, False):
                s = t[t[tag] == on]
                if not len(s):
                    continue
                s = s.sample(min(n, len(s)), random_state=seed)
                tl = [tile(r[pcol], None, f"{tag}={on} {r.get('source', r.get('cam_split', ''))}")
                      for _, r in s.iterrows()]
                sheet(tl, LOCAL / f"slices/{name}_{tag}_{'yes' if on else 'no'}.jpg", cols=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye", action="store_true")
    a = ap.parse_args()
    main()
    if a.eye:
        eye_check()

