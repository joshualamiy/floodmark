from __future__ import annotations

import json

import numpy as np
import pandas as pd

from eval.common import CAM_SPLITS, FRAMES_CSV, LOCAL, MANIFEST, REPORTS, ROOT, TRAIN_MANIFEST

HAM = 6


def ph_int(h) -> np.ndarray:
    return np.array([int(str(x), 16) for x in h], dtype=np.uint64)


def hamming_pairs(a: np.ndarray, b: np.ndarray, thr: int = HAM, chunk: int = 512):
    # all (i, j, dist) with hamming(a[i], b[j]) <= thr
    out = []
    for s in range(0, len(a), chunk):
        x = a[s:s + chunk, None] ^ b[None, :]
        dist = np.bitwise_count(x)
        i, j = np.nonzero(dist <= thr)
        out.extend(zip((i + s).tolist(), j.tolist(), dist[i, j].tolist()))
    return out


def shared(col: str, a: pd.DataFrame, b: pd.DataFrame) -> list:
    va = set(a[col].dropna().astype(str))
    vb = set(b[col].dropna().astype(str))
    return sorted(va & vb)


def split_pairs(df: pd.DataFrame, tag: str, cols=("group_id", "camera_id", "dup_cluster", "orig_path")):
    res = {}
    splits = ["train", "val", "test"]
    ph = ph_int(df["phash"])
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            a, b = df[df["split"] == s1], df[df["split"] == s2]
            r = {c: shared(c, a, b) for c in cols}
            pairs = hamming_pairs(ph[a.index.to_numpy()], ph[b.index.to_numpy()])
            ex = [{"a": a.iloc[p]["path"], "b": b.iloc[q]["path"], "a_src": a.iloc[p]["source"],
                   "b_src": b.iloc[q]["source"], "ham": d} for p, q, d in pairs]
            res[f"{s1}~{s2}"] = {**{c: {"count": len(v), "examples": v[:5]} for c, v in r.items()},
                                 "phash_le6": {"count": len(pairs), "examples": ex[:10]}}
    return {tag: res}


def test_vs_training(cur: pd.DataFrame, old: pd.DataFrame) -> dict:
    t = cur[cur["split"] == "test"]
    seen = old[old["split"].isin(["train", "val"])]
    r = {c: shared(c, t, seen) for c in ("group_id", "camera_id", "orig_path")}
    pairs = hamming_pairs(ph_int(t["phash"]), ph_int(seen["phash"]))
    ex = [{"test": t.iloc[p]["path"], "test_orig": t.iloc[p]["orig_path"],
           "seen": seen.iloc[q]["orig_path"], "seen_split": seen.iloc[q]["split"],
           "src": f"{t.iloc[p]['source']}~{seen.iloc[q]['source']}", "ham": d} for p, q, d in pairs]
    by = pd.Series([e["src"] for e in ex]).value_counts().to_dict() if ex else {}
    return {**{c: {"count": len(v), "examples": v[:5]} for c, v in r.items()},
            "phash_le6": {"count": len(pairs), "by_source_pair": by, "examples": ex[:15]}}


def fmd_vs_public(cur: pd.DataFrame) -> dict:
    f = cur[cur["source"] == "flood_master_test"]
    p = cur[cur["source"] != "flood_master_test"]
    pairs = hamming_pairs(ph_int(f["phash"]), ph_int(p["phash"]))
    out = {"manifest_phash_le6": len(pairs),
           "examples": [{"fmd": f.iloc[a]["orig_path"], "public": p.iloc[b]["orig_path"], "ham": d}
                        for a, b, d in pairs[:10]]}
    idx = ROOT / "data/restricted/flood_master/index.csv"
    if idx.exists():
        ix = pd.read_csv(idx)
        out["fmd_index_columns"] = list(ix.columns)
        col = next((c for c in ix.columns if "local" in c.lower() or "public" in c.lower()), None)
        if col:
            used = set(cur["orig_path"].astype(str))
            hits = ix[ix[col].astype(str).isin(used)]
            out["fmd_trainval_images_resolving_to_manifest_rows"] = len(hits)
            m = cur.set_index("orig_path")
            out["of_which_in_test"] = int(sum(m.at[h, "split"] == "test" for h in hits[col]
                                              if h in m.index))
    return out


def live_cross_split() -> dict:
    # different camera ids that look at the same scene across splits
    f = pd.read_csv(FRAMES_CSV)
    f = f[f["dead_reason"].isna() & f["phash"].notna()].copy()
    sp = json.loads(CAM_SPLITS.read_text())
    f["cs"] = f["camera_id"].astype(str).map(sp)
    f = f[f["cs"].notna()].reset_index(drop=True)
    t, s = f[f["cs"] == "test"], f[f["cs"] != "test"]
    pairs = hamming_pairs(ph_int(t["phash"]), ph_int(s["phash"]))
    pairs = [(a, b, d) for a, b, d in pairs if t.iloc[a]["camera_id"] != s.iloc[b]["camera_id"]]
    ex = [{"test_frame": t.iloc[a]["frame_id"], "other_frame": s.iloc[b]["frame_id"],
           "other_split": s.iloc[b]["cs"], "ham": d} for a, b, d in pairs]
    cams = f.groupby("camera_id").agg(lat=("lat", "first"), lon=("lon", "first"), cs=("cs", "first"))
    tc, oc = cams[cams["cs"] == "test"], cams[cams["cs"] != "test"]
    dlat = (tc["lat"].to_numpy()[:, None] - oc["lat"].to_numpy()[None]) * 111_000
    dlon = (tc["lon"].to_numpy()[:, None] - oc["lon"].to_numpy()[None]) * 111_000 * np.cos(np.radians(33.75))
    close = np.argwhere(np.hypot(dlat, dlon) < 30)
    near = [{"test_cam": int(tc.index[a]), "other_cam": int(oc.index[b]), "other_split": oc.iloc[b]["cs"]}
            for a, b in close]
    return {"frames_phash_le6_diff_camera": len(pairs),
            "distinct_test_frames_matched": len({e["test_frame"] for e in ex}),
            "examples": ex[:10], "camera_pairs_within_30m": len(near), "near_examples": near[:10]}


def main() -> dict:
    cur = pd.read_csv(MANIFEST, dtype={"camera_id": str, "group_id": str})
    old = pd.read_csv(TRAIN_MANIFEST, dtype={"camera_id": str, "group_id": str})
    res = {}
    res.update(split_pairs(cur, "current_manifest"))
    res.update(split_pairs(old, "training_manifest"))
    res["test_vs_training_trainval"] = test_vs_training(cur, old)
    res["fmd_vs_public"] = fmd_vs_public(cur)
    res["live_511_test_cams_vs_other_cams"] = live_cross_split()
    LOCAL.mkdir(parents=True, exist_ok=True)
    (REPORTS / "eval_leakage.json").write_text(json.dumps(res, indent=2, default=str))
    for k, v in res.items():
        if k.endswith("manifest"):
            for sp, r in v.items():
                print(k, sp, {c: r[c]["count"] for c in r})
        else:
            print(k, json.dumps(v, default=str)[:600])
    return res


if __name__ == "__main__":
    main()
