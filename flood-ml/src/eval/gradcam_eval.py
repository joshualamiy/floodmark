from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from eval.common import LOCAL, REPORTS, ROOT
from eval.viz import sheet, tile

REVIEW_STRATA = {("ga511", "dry"): 12, ("ga511", "wet"): 1, ("fred", "dry"): 8,
                 ("fred", "flooded"): 6, ("flood_master_test", "flooded"): 10,
                 ("roadway_flooding", "flooded"): 12, ("nysdot_road_surface", "dry"): 4,
                 ("nysdot_road_surface", "wet"): 6}


def load_masks(path: str, source: str) -> tuple[np.ndarray, np.ndarray | None]:
    m = np.array(Image.open(ROOT / path))
    if m.ndim == 3:
        m = m[..., 0]
    if source == "fred":
        return m == 2, (m == 1) | (m == 2)
    return m > 0, None


def to_crop(mask: np.ndarray, size: int = 224, short: int = 256) -> np.ndarray:
    h, w = mask.shape
    s = short / min(h, w)
    nw, nh = round(w * s), round(h * s)
    r = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    top, left = (nh - size) // 2, (nw - size) // 2
    return r[top:top + size, left:left + size].astype(bool)


def energy_in(cam: np.ndarray, region: np.ndarray) -> float:
    c = cv2.resize(cam.astype(np.float32), region.shape[::-1], interpolation=cv2.INTER_LINEAR)
    c = np.maximum(c, 0)
    tot = c.sum()
    return float((c * region).sum() / tot) if tot > 0 else float("nan")


def peak_in(cam: np.ndarray, region: np.ndarray) -> bool:
    c = cv2.resize(cam.astype(np.float32), region.shape[::-1], interpolation=cv2.INTER_LINEAR)
    y, x = np.unravel_index(np.argmax(c), c.shape)
    return bool(region[y, x])


def mask_energy() -> dict:
    d = pd.read_csv(LOCAL / "test_preds.csv")
    cams = np.load(LOCAL / "test_cams.npz")
    rows = d[d["mask_path"].notna()]
    recs = []
    regions = {}
    for i, r in rows.iterrows():
        w, road = load_masks(r["mask_path"], r["source"])
        regions[i] = (to_crop(w), None if road is None else to_crop(road))
    rng = np.random.default_rng(0)
    for i, r in rows.iterrows():
        wat, road = regions[i]
        same = [j for j in regions if j != i and d.at[j, "source"] == r["source"]]
        others = rng.choice(same, size=min(20, len(same)), replace=False) if same else []
        for st in ("A", "B"):
            cam = cams[f"cam{st}"][i]
            rec = {"idx": i, "source": r["source"], "label": r["label"], "status": r["status"],
                   "stage": st, "water_area": float(wat.mean()),
                   "water_energy": energy_in(cam, wat) if wat.any() else float("nan"),
                   "water_peak": peak_in(cam, wat) if wat.any() else None,
                   "water_energy_shuffled": float(np.nanmean(
                       [energy_in(cams[f"cam{st}"][j], wat) for j in others])) if len(others) else None}
            if road is not None:
                rec["road_area"] = float(road.mean())
                rec["road_energy"] = energy_in(cam, road) if road.any() else float("nan")
                rec["road_energy_shuffled"] = float(np.nanmean(
                    [energy_in(cams[f"cam{st}"][j], road) for j in others])) if len(others) else None
            recs.append(rec)
    e = pd.DataFrame(recs)
    e.to_csv(LOCAL / "cam_mask_energy.csv", index=False)
    out = {}
    for (src, st), g in e[e["label"] == "flooded"].groupby(["source", "stage"]):
        g = g.dropna(subset=["water_energy"])
        out[f"{src}|stage{st}"] = {
            "n": len(g),
            "median_water_area": float(g["water_area"].median()),
            "median_water_energy": float(g["water_energy"].median()),
            "median_energy_over_area": float((g["water_energy"] / g["water_area"]).median()),
            "median_shuffled_energy": float(g["water_energy_shuffled"].median()),
            "frac_energy_gt_area": float((g["water_energy"] > g["water_area"]).mean()),
            "frac_energy_gt_shuffled": float((g["water_energy"] > g["water_energy_shuffled"]).mean()),
            "pointing_game": float(g["water_peak"].astype(float).mean()),
            "correct_vs_missed_median_energy": {
                "correct": float(g[g["status"] == "flooded"]["water_energy"].median()),
                "missed": float(g[g["status"] != "flooded"]["water_energy"].median())},
        }
    fr = e[(e["source"] == "fred") & e["road_area"].notna()]
    for st, g in fr.groupby("stage"):
        g = g.dropna(subset=["road_energy"])
        out[f"fred_road|stage{st}"] = {
            "n": len(g), "labels": g["label"].value_counts().to_dict(),
            "median_road_area": float(g["road_area"].median()),
            "median_road_energy": float(g["road_energy"].median()),
            "median_shuffled_energy": float(g["road_energy_shuffled"].median())}
    (REPORTS / "eval_gradcam_energy.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def review_sample(seed: int = 0) -> pd.DataFrame:
    d = pd.read_csv(LOCAL / "test_preds.csv")
    parts = [d[(d["source"] == s) & (d["label"] == lab)].sample(k, random_state=seed)
             for (s, lab), k in REVIEW_STRATA.items()]
    samp = pd.concat(parts)
    samp.to_csv(LOCAL / "gradcam_review_sample.csv")
    cams = np.load(LOCAL / "test_cams.npz")
    for n, start in enumerate(range(0, len(samp), 10)):
        tl = []
        for i, r in samp.iloc[start:start + 10].iterrows():
            cap = f"#{i} {r['source'][:10]} {r['label']} pA={r['pA']:.2f} pB={r['pB']:.2f} {r['status']}"
            tl.append(tile(r["path"], cams["camA"][i], cap + " [A]"))
            tl.append(tile(r["path"], cams["camB"][i], "[B]"))
        sheet(tl, LOCAL / f"gradcam_review/sheet_{n}.jpg", cols=4)
    print(len(samp), "sampled")
    return samp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["energy", "sample"])
    a = ap.parse_args()
    mask_energy() if a.what == "energy" else review_sample()
