from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
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

# v1 = the model of the first phase 4 run; v3 = shipped letterbox320
MODEL_DIRS = {"v1": MODELS / "v1", "v3": MODELS}
LEGACY_SOURCES = ("ga511", "fred", "roadway_flooding", "flood_master_test", "nysdot_road_surface")
NEW_SOURCES = ("iowa_rwis", "eu_flood_2013", "alleyfloodnet")

LABELS = ("dry", "wet", "flooded")
TRUE_LABELS = ("dry", "wet", "flooded", "not_flooded")
SMALL_N, SMALL_G = 30, 5
CROP224 = {"mode": "crop", "size": 224, "jpeg": True}


def model_dir(tag: str) -> Path:
    return MODEL_DIRS[tag]


def local(tag: str | None = None) -> Path:
    # per-model outputs; None = the original v1 run's files
    p = LOCAL / tag if tag else LOCAL
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(models_dir: Path = MODELS) -> dict:
    return json.loads((Path(models_dir) / "config.json").read_text())


def thresholds(cfg: dict | None = None) -> tuple[float, float]:
    cfg = cfg or load_config()
    return float(cfg["stage_a"]["threshold_tA"]), float(cfg["stage_b"]["threshold_tB"])


def preproc_spec(cfg: dict) -> dict:
    # same keys/defaults as inference.predict_batch
    pp = cfg.get("preprocess", {})
    return {"mode": pp.get("mode", "crop"),
            "size": int(pp.get("size", cfg.get("input", {}).get("size", 224))),
            "jpeg": bool(pp.get("jpeg_roundtrip", True))}


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


def preprocess_geom(img, spec: dict | None = None):
    # the deployed function itself, so eval input == deployed input
    from inference.preprocess import preprocess as deployed

    s = spec or CROP224
    return deployed(img, mode=s["mode"], size=s["size"], do_jpeg_roundtrip=s["jpeg"])


def preprocess(img, spec: dict | None = None) -> np.ndarray:
    return preprocess_geom(img, spec)[0]


def preprocess_pil_legacy(img: Image.Image, short: int = 256, size: int = 224) -> np.ndarray:
    # first phase 4 loader (PIL bilinear, no jpeg); only to check v1 reproduces
    img = img.convert("RGB")
    w, h = img.size
    s = short / min(w, h)
    nw, nh = round(w * s), round(h * s)
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - size) // 2, (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype=np.float32)


def abs_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_pil(path: str | Path) -> Image.Image:
    with Image.open(abs_path(path)) as im:
        im.load()
        return im.convert("RGB")


def load_image(path: str | Path, spec: dict | None = None) -> np.ndarray:
    return preprocess(load_pil(path), spec)


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
        self.dir = Path(models_dir)
        self.cfg = load_config(self.dir)
        self.a = ort.InferenceSession(str(self.dir / self.cfg["stage_a"]["onnx_path"]), so, providers=prov)
        self.b = ort.InferenceSession(str(self.dir / self.cfg["stage_b"]["onnx_path"]), so, providers=prov)
        self.ta, self.tb = thresholds(self.cfg)
        self.spec = preproc_spec(self.cfg)

    def load(self, path) -> np.ndarray:
        return load_image(path, self.spec)

    def prep(self, img: Image.Image) -> np.ndarray:
        return preprocess(img, self.spec)

    @staticmethod
    def _run(sess, x):
        # outputs come back as (cam, prob): always fetch by name
        cam, prob = sess.run(["cam", "prob"], {"image": x})
        return prob[:, 0], cam

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x, np.float32)
        pa, cam_a = self._run(self.a, x)
        pb, cam_b = self._run(self.b, x)
        st = status_of(pa, pb, self.ta, self.tb)
        return {"pA": pa, "pB": pb, "camA": cam_a, "camB": cam_b, "status": st,
                "confidence": confidence(pa, pb, st)}


def predict_paths(pipe: Pipeline, paths, batch: int = 64, loader=None, workers: int = 8) -> dict:
    loader = loader or pipe.load
    out: dict[str, list] = {}
    paths = list(paths)
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(paths), batch):
            x = np.stack(list(ex.map(loader, paths[i:i + batch])))
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


def paired_delta(num_a, num_b, den, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    # rate_b - rate_a on the same rows, clusters resampled jointly
    num_a, num_b, den = (np.asarray(v, float) for v in (num_a, num_b, den))
    keep = den > 0
    if keep.sum() == 0:
        return {"delta": float("nan"), "ci_group": [float("nan")] * 2, "n": 0, "n_groups": 0}
    g = pd.Series(np.asarray(groups)).astype(str)[keep].values
    agg = pd.DataFrame({"g": g, "a": num_a[keep] * den[keep], "b": num_b[keep] * den[keep],
                        "d": den[keep]}).groupby("g").sum()
    ga, gb, gd = agg["a"].to_numpy(), agg["b"].to_numpy(), agg["d"].to_numpy()
    k = len(agg)
    idx = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    sd = gd[idx].sum(1)
    dl = (gb[idx].sum(1) - ga[idx].sum(1)) / sd
    lo, hi = np.percentile(dl, [2.5, 97.5])
    return {"delta": float((gb.sum() - ga.sum()) / gd.sum()), "ci_group": [float(lo), float(hi)],
            "n": int(gd.sum()), "n_groups": k}


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
