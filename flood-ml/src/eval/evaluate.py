from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from eval.common import FIGS, LABELS, LOCAL, REPORTS, fmt_rate, rate, thresholds

# reference palette slots 1-3 + ink
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def confusion(true, pred, labels=LABELS) -> pd.DataFrame:
    return pd.crosstab(pd.Categorical(true, labels), pd.Categorical(pred, labels), dropna=False)


def auc_safe(y, s) -> dict:
    y, s = np.asarray(y, int), np.asarray(s, float)
    if len(y) == 0 or y.min() == y.max():
        return {"auc_roc": None, "auc_pr": None, "n_pos": int(y.sum()), "n": len(y)}
    return {"auc_roc": float(roc_auc_score(y, s)), "auc_pr": float(average_precision_score(y, s)),
            "n_pos": int(y.sum()), "n": len(y)}


def set_metrics(d: pd.DataFrame, ta: float, tb: float) -> dict:
    g, y, st = d["boot_group"], d["label"].to_numpy(), d["status"].to_numpy()
    out: dict = {"n": len(d), "n_groups": int(g.nunique()),
                 "labels": d["label"].value_counts().to_dict()}
    out["confusion_3class"] = confusion(y, st).to_dict()
    per = {}
    for c in LABELS:
        per[c] = {"precision": rate(y == c, st == c, g), "recall": rate(st == c, y == c, g)}
    out["per_class"] = per
    out["false_alarm"] = {
        "flooded|wet": rate(st == "flooded", y == "wet", g),
        "flooded|dry": rate(st == "flooded", y == "dry", g),
        "wet|dry": rate(st == "wet", y == "dry", g),
        "notdry|dry": rate(st != "dry", y == "dry", g),
    }
    out["missed_flood"] = {
        "dry|flooded": rate(st == "dry", y == "flooded", g),
        "wet|flooded": rate(st == "wet", y == "flooded", g),
    }
    a_true, a_pred = y != "dry", d["pA"].to_numpy() >= ta
    out["stage_a"] = {"confusion": pd.crosstab(pd.Categorical(a_true, [False, True]),
                                               pd.Categorical(a_pred, [False, True]),
                                               dropna=False).to_numpy().tolist(),
                      **auc_safe(a_true, d["pA"])}
    b_true, b_pred = y == "flooded", d["pB"].to_numpy() >= tb
    gate = a_pred
    wf = y != "dry"
    pb = d["pB"].to_numpy()
    out["stage_b_all_rows"] = {"confusion": _cm(b_true, b_pred), **auc_safe(b_true, d["pB"])}
    out["stage_b_true_wet_or_flooded"] = {"confusion": _cm(b_true[wf], b_pred[wf]),
                                          **auc_safe(b_true[wf], pb[wf])}
    out["stage_b_after_gate"] = {"confusion": _cm(b_true[gate], b_pred[gate]),
                                 **auc_safe(b_true[gate], pb[gate])}
    out["flooded_score_pApB"] = auc_safe(b_true, d["pA"] * d["pB"])
    return out


def _cm(t, p) -> list:
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    return [[int((~t & ~p).sum()), int((~t & p).sum())], [int((t & ~p).sum()), int((t & p).sum())]]


def test_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sets = {"a_ga511": df[df["source"] == "ga511"]}
    ext = df[df["source"] != "ga511"]
    for s in sorted(ext["source"].unique()):
        sets[f"b_{s}"] = ext[ext["source"] == s]
    sets["b_external_pooled"] = ext
    sets["all_test"] = df
    return sets


def live_metrics(f: pd.DataFrame) -> dict:
    out = {}
    views = {"test_cams_all_good": f["cam_split"] == "test",
             "test_cams_unlabeled_only": (f["cam_split"] == "test") & f["user_label"].isna(),
             "test_cams_excl_user_unusable": (f["cam_split"] == "test") & (f["user_label"] != "unusable"),
             "val_cams_SEEN_IN_SELECTION": f["cam_split"] == "val",
             "train_cams_SEEN_IN_TRAINING": f["cam_split"] == "train",
             "unmapped_cams": f["cam_split"] == "unmapped",
             "all_cams": pd.Series(True, index=f.index)}
    g = f["camera_id"].astype(str)
    for k, m in views.items():
        m = m.to_numpy()
        out[k] = {"frames": int(m.sum()), "cameras": int(g[m].nunique()),
                  "flooded": rate(f["status"] == "flooded", m, g),
                  "wet": rate(f["status"] == "wet", m, g),
                  "not_dry": rate(f["status"] != "dry", m, g),
                  "status_flips_tf_preproc": int((f.loc[m, "status"] != f.loc[m, "status_tf"]).sum())
                  if "status_tf" in f else None}
    ts = pd.to_datetime(f["timestamp_utc"], unit="s", utc=True).dt.tz_convert("America/New_York")
    out["window_local"] = [str(ts.min()), str(ts.max())]
    out["precip_1h_mm"] = {"n_with_value": int(f["precip_1h_mm"].notna().sum()),
                           "max": float(f["precip_1h_mm"].max()),
                           "n_missing": int(f["precip_1h_mm"].isna().sum())}
    return out


def _style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall", color=INK2, fontsize=9)
    ax.set_ylabel("precision", color=INK2, fontsize=9)


def pr_panel(ax, curves, title):
    _style(ax, title)
    ops = []
    for (name, y, s, op), col in zip(curves, (C1, C2, C3)):
        y = np.asarray(y, int)
        if y.min() == y.max():
            continue
        p, r, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(r, p, color=col, lw=2, label=f"{name} (AP {ap:.2f}, {y.sum()} pos / {len(y)})")
        if op is not None:
            yp = np.asarray(op, bool)
            tp = (yp & (y == 1)).sum()
            if yp.sum() == 0:
                ax.text(0.98, 0.3, f"{name}: 0 predicted positive\nat the shipped threshold", color=INK2, fontsize=8, ha="right")
                continue
            ops.append((tp / max(y.sum(), 1), tp / yp.sum(), col))
    # later dots smaller so overlapping operating points stay visible
    for k, (rc, pr_, col) in enumerate(ops):
        ax.plot([rc], [pr_], "o", ms=11 - 3 * k, color=col, mec=SURF, mew=1.5, zorder=5 + k)
    ax.legend(fontsize=7, frameon=False, loc="lower left", labelcolor=INK)


def pr_figures(df: pd.DataFrame, ta: float, tb: float) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    ext = df[df["source"] != "ga511"]
    fred = df[df["source"] == "fred"]
    nys = df[df["source"] == "nysdot_road_surface"]
    paths = []

    def cur(d, target, score, op):
        return (d[target], d[score], op(d))

    notdry = lambda d: (d["label"] != "dry").astype(int)
    fl = lambda d: (d["label"] == "flooded").astype(int)
    opA = lambda d: d["pA"] >= ta
    opF = lambda d: d["status"] == "flooded"

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURF)
    pr_panel(axs[0], [("all test", notdry(df), df["pA"], opA(df)),
                      ("external", notdry(ext), ext["pA"], opA(ext)),
                      ("fred", notdry(fred), fred["pA"], opA(fred))],
             "Stage A: wet-or-flooded vs dry (score pA)")
    pr_panel(axs[1], [("nysdot (dry vs wet)", notdry(nys), nys["pA"], opA(nys))],
             "Stage A within NYSDOT: n=16, 1 camera")
    fig.text(0.01, 0.01, "dot = shipped threshold tA. test split only.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = FIGS / "pr_stage_a.png"
    fig.savefig(p, dpi=130, facecolor=SURF)
    plt.close(fig)
    paths.append(str(p))

    df = df.assign(pf=df["pA"] * df["pB"])
    ext, fred = df[df["source"] != "ga511"], df[df["source"] == "fred"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURF)
    pr_panel(axs[0], [("all test", fl(df), df["pf"], opF(df)),
                      ("external", fl(ext), ext["pf"], opF(ext)),
                      ("fred", fl(fred), fred["pf"], opF(fred))],
             "Pipeline: flooded vs not (score pA*pB)")
    wf = df[df["label"] != "dry"]
    pr_panel(axs[1], [("B on true wet+flooded", fl(wf), wf["pB"], wf["pB"] >= tb),
                      ("B on all rows", fl(df), df["pB"], df["pB"] >= tb)],
             "Stage B alone (score pB); only 10 wet rows")
    fig.text(0.01, 0.01, "dot = shipped operating point. test split only.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = FIGS / "pr_flooded.png"
    fig.savefig(p, dpi=130, facecolor=SURF)
    plt.close(fig)
    paths.append(str(p))
    return paths


def threshold_sensitivity(df: pd.DataFrame, live: pd.DataFrame | None) -> pd.DataFrame:
    # what-if only: never pick thresholds from this table
    from eval.common import status_of

    rows = []
    fl = (df["label"] == "flooded").to_numpy()
    for ta in (0.3, 0.5, 0.7, 0.816):
        for tb in (0.5, 0.7, 0.897):
            st = status_of(df["pA"], df["pB"], ta, tb)
            r = {"tA": ta, "tB": tb, "recall_all": round(float((st[fl] == "flooded").mean()), 3)}
            for src in ("flood_master_test", "fred", "roadway_flooding"):
                m = fl & (df["source"] == src).to_numpy()
                r[f"recall_{src}"] = round(float((st[m] == "flooded").mean()), 3)
            r["dry_called_flooded"] = int((st[(df["label"] == "dry").to_numpy()] == "flooded").sum())
            r["wet_called_not_dry"] = int((st[(df["label"] == "wet").to_numpy()] != "dry").sum())
            if live is not None:
                sl = status_of(live["pA"], live["pB"], ta, tb)
                r["live_flooded"], r["live_not_dry"] = int((sl == "flooded").sum()), int((sl != "dry").sum())
            rows.append(r)
    return pd.DataFrame(rows)


def main(sensitivity: bool = False) -> dict:
    ta, tb = thresholds()
    df = pd.read_csv(LOCAL / "test_preds.csv", dtype={"camera_id": str, "group_id": str})
    if sensitivity:
        df = df.assign(pA=df["pA_tf"], pB=df["pB_tf"], status=df["status_tf"])
    res = {"thresholds": {"tA": ta, "tB": tb}, "preprocessing": "tf" if sensitivity else "pil",
           "sets": {k: set_metrics(v, ta, tb) for k, v in test_sets(df).items()}}
    live = LOCAL / "live_preds.csv"
    if live.exists() and not sensitivity:
        res["live_c"] = live_metrics(pd.read_csv(live))
    if not sensitivity:
        res["figures"] = pr_figures(df, ta, tb)
        lv = pd.read_csv(live) if live.exists() else None
        threshold_sensitivity(df, lv).to_csv(REPORTS / "eval_threshold_sensitivity.csv", index=False)
    name = "eval_metrics_tfpreproc.json" if sensitivity else "eval_metrics.json"
    (REPORTS / name).write_text(json.dumps(res, indent=2, default=float))
    for k, v in res["sets"].items():
        fa = v["false_alarm"]
        print(f"{k:28s} n={v['n']:4d} rec_fl={fmt_rate(v['per_class']['flooded']['recall'])}")
        print(f"{'':28s} FA fl|dry={fmt_rate(fa['flooded|dry'])} fl|wet={fmt_rate(fa['flooded|wet'])}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-preproc", action="store_true")
    main(ap.parse_args().tf_preproc)
