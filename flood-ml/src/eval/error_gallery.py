from __future__ import annotations

import argparse
import html

import numpy as np
import pandas as pd
from PIL import Image

from eval.common import (
    LOCAL,
    REPORTS,
    load_config,
    load_image,
    local,
    model_dir,
    preproc_spec,
    stage_probs,
)
from eval.viz import b64_jpeg, overlay

ERR_TYPES = [
    ("dry", "wet", "dry -> wet"),
    ("dry", "flooded", "dry -> flooded (false flood alarm on dry)"),
    ("wet", "dry", "wet -> dry"),
    ("wet", "flooded", "wet -> flooded (false flood alarm on wet)"),
    ("flooded", "dry", "flooded -> dry (missed flood)"),
    ("flooded", "wet", "flooded -> wet (missed flood)"),
    ("not_flooded", "flooded", "not_flooded -> flooded (false flood alarm, dry or rain-wet)"),
]


def error_type(true: str, pred: str) -> str | None:
    for t, p, name in ERR_TYPES:
        if t == true and p == pred:
            return name
    return None


def card(path: str, cam_a, cam_b, meta: dict, spec: dict | None = None) -> str:
    img = load_image(path, spec)
    parts = [img.astype(np.uint8), overlay(img, cam_a), overlay(img, cam_b)]
    strip = Image.fromarray(np.concatenate(parts, axis=1)).resize((3 * 160, 160))
    sp = stage_probs(meta["pA"], meta["pB"])
    rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
                   for k, v in meta.items() if k not in ("pA", "pB"))
    probs = " / ".join(f"{k} {float(v):.3f}" for k, v in sp.items())
    return (f'<div class="card"><img src="data:image/jpeg;base64,{b64_jpeg(strip)}">'
            f'<table>{rows}<tr><td>pA / pB</td><td>{meta["pA"]:.3f} / {meta["pB"]:.3f}</td></tr>'
            f"<tr><td>stage probs</td><td>{probs}</td></tr></table></div>")


CSS = """body{font-family:system-ui,sans-serif;margin:16px;background:#fcfcfb;color:#0b0b0b}
.card{display:inline-block;vertical-align:top;margin:6px;padding:6px;border:1px solid #ddd;
background:#fff;width:490px}.card img{width:480px}table{font-size:11px;border-collapse:collapse}
td{padding:1px 6px 1px 0}h2{margin-top:28px}.note{color:#52514e;font-size:13px}"""


def build(out=REPORTS / "errors.html", live_wet: bool = True, tag: str | None = None,
          live_splits=("test",)) -> dict:
    # tag None = first run (v1); else reports/eval/<tag>/
    base = local(tag) if tag else LOCAL
    spec = preproc_spec(load_config(model_dir(tag))) if tag else None
    d = pd.read_csv(base / "test_preds.csv")
    cams = np.load(base / "test_cams.npz")
    d["err"] = [error_type(t, p) for t, p in zip(d["label"], d["status"])]
    counts: dict = {}
    geo = f"{spec['mode']} {spec['size']}" if spec else "crop 224"
    body = [f"<h1>Test-split errors ({tag or 'v1'}, ONNX pipeline)</h1>",
            (f'<p class="note">Local only. Each strip: model input ({geo}), Stage A CAM, '
             "Stage B CAM. CAMs are min-max normalized per image, so a strip always shows some "
             "red even when evidence is weak.</p>")]
    for _, _, name in ERR_TYPES:
        sub = d[d["err"] == name]
        counts[name] = {"total": len(sub), **sub["source"].value_counts().to_dict()}
        body.append(f"<h2>{html.escape(name)}: {len(sub)}</h2>")
        for i, r in sub.sort_values(["source", "confidence"], ascending=[True, False]).iterrows():
            meta = {"path": r["path"], "source": r["source"], "group": r["boot_group"],
                    "true": r["label"], "pred": r["status"], "conf": f"{r['confidence']:.3f}",
                    "pA": r["pA"], "pB": r["pB"]}
            body.append(card(r["path"], cams["camA"][i], cams["camB"][i], meta, spec))
    lp = base / "live_preds.csv"
    if lp.exists():
        f = pd.read_csv(lp)
        lc = np.load(base / "live_cams.npz")
        if tag:
            f = f[f["cam_split"].isin(live_splits)]
        keep = ["flooded", "wet"] if live_wet else ["flooded"]
        for st in keep:
            sub = f[f["status"] == st].sort_values("timestamp_utc")
            counts[f"live_{st}"] = len(sub)
            body.append(f"<h2>(c) live 511GA frames ({', '.join(live_splits) if tag else 'all'} cameras) "
                        f"called {st}: {len(sub)} (presumed dry: 0 mm rain)</h2>")
            for i, r in sub.iterrows():
                meta = {"frame": r["frame_id"], "camera": r["camera_id"], "cam split": r["cam_split"],
                        "local time": r.get("local_time"), "user label": r.get("user_label"),
                        "pred": r["status"], "conf": f"{r['confidence']:.3f}", "pA": r["pA"], "pB": r["pB"]}
                body.append(card(r["abs_path"], lc["camA"][i], lc["camB"][i], meta, spec))
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>Test errors</title>"
           f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>")
    out.write_text(doc)
    print(out, counts)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-live-wet", action="store_true")
    ap.add_argument("--tag", default=None, choices=["v1", "v3"])
    a = ap.parse_args()
    build(live_wet=not a.no_live_wet, tag=a.tag)
