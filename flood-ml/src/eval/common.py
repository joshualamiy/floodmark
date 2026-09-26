from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
LOCAL = REPORTS / "eval"  # gitignored: per-row outputs, cams, overlays
MANIFEST = ROOT / "data/processed/manifest.csv"
TRAIN_MANIFEST = ROOT / "data/processed/manifest_train_v1-c7dea35e.csv"
FRAMES_CSV = ROOT / "data/ga511/frames.csv"
CAM_SPLITS = ROOT / "data/processed/ga511_camera_splits.json"

LABELS = ("dry", "wet", "flooded")
SMALL_N, SMALL_G = 30, 5


def load_config() -> dict:
    return json.loads((MODELS / "config.json").read_text())


def thresholds(cfg: dict | None = None) -> tuple[float, float]:
    cfg = cfg or load_config()
    return float(cfg["stage_a"]["threshold_tA"]), float(cfg["stage_b"]["threshold_tB"])


def load_test(manifest: Path = MANIFEST) -> pd.DataFrame:
    df = pd.read_csv(manifest, dtype={"camera_id": str, "group_id": str})
    df = df[df["split"] == "test"].reset_index(drop=True)
    df["boot_group"] = boot_groups(df)
    return df


def boot_groups(df: pd.DataFrame) -> pd.Series:
    # fred has one location in test, so resample by sequence (still correlated)
    seq = df["orig_path"].astype(str).str.split("/").str[-3]
    unit = np.where(df["source"] == "fred", seq, df["group_id"].astype(str))
    return df["source"].astype(str) + ":" + pd.Series(unit, index=df.index)


def preprocess(img: Image.Image, short: int = 256, size: int = 224) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    s = short / min(w, h)
    nw, nh = round(w * s), round(h * s)
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - size) // 2, (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype=np.float32)


def load_image(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with Image.open(p) as im:
        return preprocess(im)


def status_of(pa, pb, ta: float, tb: float) -> np.ndarray:
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    return np.where(pa < ta, "dry", np.where(pb >= tb, "flooded", "wet"))


def stage_probs(pa, pb) -> dict[str, np.ndarray]:
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    return {"dry": 1 - pa, "wet": pa * (1 - pb), "flooded": pa * pb}


def confidence(pa, pb, status) -> np.ndarray:
    sp = stage_probs(pa, pb)
    status = np.asarray(status)
    out = np.zeros(len(status))
    for k, v in sp.items():
        out[status == k] = v[status == k]
    return out


class Pipeline:
    def __init__(self, models_dir: Path = MODELS, threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.a = ort.InferenceSession(str(models_dir / "stage_a.onnx"), so, providers=prov)
        self.b = ort.InferenceSession(str(models_dir / "stage_b.onnx"), so, providers=prov)
        self.ta, self.tb = thresholds(json.loads((models_dir / "config.json").read_text()))

    @staticmethod
    def _run(sess, x):
        # outputs come back as (cam, prob): always fetch by name
        cam, prob = sess.run(["cam", "prob"], {"image": x})
        return prob[:, 0], cam

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        pa, cam_a = self._run(self.a, x)
        pb, cam_b = self._run(self.b, x)
        st = status_of(pa, pb, self.ta, self.tb)
        return {"pA": pa, "pB": pb, "camA": cam_a, "camB": cam_b, "status": st,
                "confidence": confidence(pa, pb, st)}


def predict_paths(pipe: Pipeline, paths, batch: int = 64, loader=load_image) -> dict:
    out: dict[str, list] = {}
    paths = list(paths)
    for i in range(0, len(paths), batch):
        x = np.stack([loader(p) for p in paths[i:i + batch]])
        r = pipe.predict(x)
        for k, v in r.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ratio_boot(num: np.ndarray, den: np.ndarray, groups, n_boot: int = 2000, seed: int = 0):
    # cluster bootstrap of sum(num)/sum(den); returns (lo, hi, n_groups_with_den)
    num, den = np.asarray(num, float), np.asarray(den, float)
    g = pd.Series(np.asarray(groups)).astype(str)
    keep = den > 0
    if keep.sum() == 0:
        return float("nan"), float("nan"), 0
    agg = pd.DataFrame({"g": g[keep].values, "n": num[keep], "d": den[keep]}).groupby("g").sum()
    gn, gd = agg["n"].to_numpy(), agg["d"].to_numpy()
    k = len(agg)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    sn, sd = gn[idx].sum(1), gd[idx].sum(1)
    r = sn[sd > 0] / sd[sd > 0]
    lo, hi = np.percentile(r, [2.5, 97.5]) if len(r) else (float("nan"),) * 2
    return float(lo), float(hi), k


def rate(num_mask, den_mask, groups, seed: int = 0) -> dict:
    num_mask, den_mask = np.asarray(num_mask, bool), np.asarray(den_mask, bool)
    k, n = int((num_mask & den_mask).sum()), int(den_mask.sum())
    lo, hi, ng = ratio_boot(num_mask & den_mask, den_mask, groups, seed=seed)
    wl, wh = wilson(k, n)
    flags = []
    if n < SMALL_N:
        flags.append(f"n<{SMALL_N}")
    if ng < SMALL_G:
        flags.append(f"groups<{SMALL_G}")
    return {"k": k, "n": n, "value": (k / n) if n else float("nan"), "ci_group": [lo, hi],
            "ci_wilson_row": [wl, wh], "n_groups": ng, "flags": flags}


def fmt_rate(r: dict) -> str:
    v = "n/a" if not r["n"] else f"{r['value']:.3f}"
    lo, hi = r["ci_group"]
    ci = "" if np.isnan(lo) else f" [{lo:.3f}, {hi:.3f}]"
    fl = f" **{', '.join(r['flags'])}**" if r["flags"] else ""
    return f"{v} ({r['k']}/{r['n']}, {r['n_groups']} grp){ci}{fl}"
