"""Image-space stress tests for both models (v3 rerun).

Edits go on the full frame before each model's own deployed preprocess, so a
511GA-style overlay lands on image content, not on letterbox padding.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from eval.common import REPORTS, Pipeline, load_pil, local, model_dir

TAGS = ("v1", "v3")


def darken(im: Image.Image) -> Image.Image:
    x = np.asarray(im, np.float32)
    return Image.fromarray(np.clip(255 * (x / 255) ** 2.2 * 0.6, 0, 255).astype(np.uint8))


def ga511_overlay(im: Image.Image) -> Image.Image:
    # black title band top-left + white logo box top-right, sized like real 511GA frames
    im = im.copy()
    w, h = im.size
    bh = max(10, round(0.07 * h))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, round(0.70 * w), bh], fill=(0, 0, 0))
    step = max(3, round(bh * 0.45))
    for x in range(4, round(0.68 * w), step):
        d.rectangle([x, round(bh * 0.2), x + max(1, step // 3), round(bh * 0.8)], fill=(255, 255, 255))
    d.rectangle([round(0.89 * w), 2, w - 3, round(0.15 * h)], fill=(245, 245, 250))
    return im


def overlay_part(im: Image.Image, part: str) -> Image.Image:
    # decompose the overlay; "band_bottom" = same band moved to the bottom (occlusion control)
    im = im.copy()
    w, h = im.size
    bh = max(10, round(0.07 * h))
    d = ImageDraw.Draw(im)
    if part == "band_top":
        d.rectangle([0, 0, round(0.70 * w), bh], fill=(0, 0, 0))
    elif part == "band_bottom":
        d.rectangle([0, h - bh - 1, round(0.70 * w), h - 1], fill=(0, 0, 0))
    elif part == "logo":
        d.rectangle([round(0.89 * w), 2, w - 3, round(0.15 * h)], fill=(245, 245, 250))
    elif part == "gray_band_top":
        d.rectangle([0, 0, round(0.70 * w), bh], fill=(128, 128, 128))
    return im


def crop_aspect(im: Image.Image, aspect: float) -> Image.Image:
    # largest centered crop with w/h == aspect
    w, h = im.size
    if w / h > aspect:
        nw = round(h * aspect)
        return im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    nh = round(w / aspect)
    return im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))


def crop_area(im: Image.Image, frac: float) -> Image.Image:
    # control: same area removed, aspect kept
    w, h = im.size
    s = np.sqrt(frac)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return im.crop(((w - nw) // 2, (h - nh) // 2, (w - nw) // 2 + nw, (h - nh) // 2 + nh))


def edit(im: Image.Image, kind: str) -> Image.Image:
    if kind == "dark":
        return darken(im)
    if kind == "ga511_overlay":
        return ga511_overlay(im)
    if kind == "dark+overlay":
        return ga511_overlay(darken(im))
    if kind in ("band_top", "band_bottom", "logo", "gray_band_top"):
        return overlay_part(im, kind)
    if kind in ("to_16x9", "to_4x3"):
        return crop_aspect(im, 16 / 9 if kind == "to_16x9" else 4 / 3)
    if kind in ("to_16x9_ctrl", "to_4x3_ctrl"):
        a = 16 / 9 if kind == "to_16x9_ctrl" else 4 / 3
        c = crop_aspect(im, a)
        return crop_area(im, (c.size[0] * c.size[1]) / (im.size[0] * im.size[1]))
    raise ValueError(kind)


def run_set(pipe: Pipeline, paths, kinds) -> dict[str, np.ndarray]:
    ims = [load_pil(p) for p in paths]
    out = {"base": pipe.predict(np.stack([pipe.prep(i) for i in ims]))}
    for k in kinds:
        out[k] = pipe.predict(np.stack([pipe.prep(edit(i, k)) for i in ims]))
    return out


def summarize(sub: pd.DataFrame, res: dict) -> dict:
    base = res["base"]["status"]
    out = {}
    for k, r in res.items():
        st = r["status"]
        out[k] = {"n": len(sub), "flooded": int((st == "flooded").sum()), "dry": int((st == "dry").sum()),
                  "flooded_to_not": int(((base == "flooded") & (st != "flooded")).sum()),
                  "not_to_flooded": int(((base != "flooded") & (st == "flooded")).sum()),
                  "mean_pA": float(r["pA"].mean()), "mean_pB": float(r["pB"].mean()),
                  "flooded_by_source": sub.assign(s=st).groupby("source")["s"].apply(
                      lambda s: int((s == "flooded").sum())).to_dict()}
    return out


def main() -> dict:
    res: dict = {}
    for tag in TAGS:
        pipe = Pipeline(model_dir(tag))
        d = pd.read_csv(local(tag) / "test_preds.csv")
        r: dict = {}
        fl = d[d["label"] == "flooded"]
        r["flooded"] = summarize(fl, run_set(pipe, fl["path"], (
            "dark", "ga511_overlay", "dark+overlay", "band_top", "band_bottom", "gray_band_top", "logo")))
        dry = d[(d["label"] == "dry") & (d["source"] != "ga511")]
        r["dry_non_ga511"] = summarize(dry, run_set(pipe, dry["path"], ("dark", "ga511_overlay")))
        # aspect: letterbox bars encode source aspect ratio
        asp = d["width"] / d["height"]
        f43 = fl[(asp[fl.index] < 1.5).to_numpy()]
        r["flooded_non16x9"] = summarize(f43, run_set(pipe, f43["path"], ("to_16x9", "to_16x9_ctrl")))
        ga = d[(d["source"] == "ga511") & (d["label"] == "dry")]
        r["ga511_dry"] = summarize(ga, run_set(pipe, ga["path"], ("to_4x3", "to_4x3_ctrl")))
        nf = d[d["label"] == "not_flooded"]
        r["not_flooded"] = summarize(nf, run_set(pipe, nf["path"], ("dark", "ga511_overlay")))
        res[tag] = r
        print(tag, json.dumps({s: {k: (v["flooded"], v["dry"]) for k, v in x.items()} for s, x in r.items()}))
    (REPORTS / "eval_v3_perturb.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
