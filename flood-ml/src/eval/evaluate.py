from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from eval.common import (
    FIGS,
    LABELS,
    LEGACY_SOURCES,
    NEW_SOURCES,
    REPORTS,
    TRUE_LABELS,
    fmt_rate,
    load_config,
    local,
    model_dir,
    paired_delta,
    rate,
    status_of,
    thresholds,
)

# reference palette slots 1-2 + ink
C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
TAGS = ("v1", "v3")


def confusion(true, pred, rows=LABELS, cols=LABELS) -> pd.DataFrame:
    return pd.crosstab(pd.Categorical(true, rows), pd.Categorical(pred, cols), dropna=False)


def auc_safe(y, s) -> dict:
    y, s = np.asarray(y, int), np.asarray(s, float)
    if len(y) == 0 or y.min() == y.max():
        return {"auc_roc": None, "auc_pr": None, "n_pos": int(y.sum()), "n": len(y)}
    return {"auc_roc": float(roc_auc_score(y, s)), "auc_pr": float(average_precision_score(y, s)),
            "n_pos": int(y.sum()), "n": len(y)}


def _cm(t, p) -> list:
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    return [[int((~t & ~p).sum()), int((~t & p).sum())], [int((t & ~p).sum()), int((t & p).sum())]]


def set_metrics(d: pd.DataFrame, ta: float, tb: float) -> dict:
    # not_flooded rows only count toward "called flooded or not"
    g, y, st = d["boot_group"], d["label"].to_numpy(), d["status"].to_numpy()
    pa, pb = d["pA"].to_numpy(), d["pB"].to_numpy()
    known = y != "not_flooded"
    out: dict = {"n": len(d), "n_groups": int(g.nunique()), "labels": d["label"].value_counts().to_dict()}
    cm = confusion(y, st, rows=TRUE_LABELS)
    out["confusion"] = {r: {c: int(cm.loc[r, c]) for c in LABELS} for r in TRUE_LABELS}
    per = {}
    for c in LABELS:
        den = (st == c) if c == "flooded" else (st == c) & known
        per[c] = {"precision": rate(y == c, den, g), "recall": rate(st == c, y == c, g)}
    out["per_class"] = per
    fl = st == "flooded"
    out["false_flood"] = {k: rate(fl, y == k, g) for k in ("dry", "wet", "not_flooded")}
    out["false_flood"]["any_non_flood"] = rate(fl, y != "flooded", g)
    out["false_alarm"] = {"wet|dry": rate(st == "wet", y == "dry", g),
                          "notdry|dry": rate(st != "dry", y == "dry", g)}
    out["wet"] = {"stageA_recall": rate(pa >= ta, y == "wet", g),
                  "status_wet_recall": rate(st == "wet", y == "wet", g)}
    out["missed_flood"] = {"dry|flooded": rate(st == "dry", y == "flooded", g),
                           "wet|flooded": rate(st == "wet", y == "flooded", g)}
    kt, kp = y[known] != "dry", pa[known] >= ta
    out["stage_a"] = {"confusion": _cm(kt, kp), **auc_safe(kt, pa[known])}
    wd = (y == "wet") | (y == "dry")
    out["stage_a_wet_vs_dry"] = auc_safe(y[wd] == "wet", pa[wd])
    bt, bp = y == "flooded", pb >= tb
    gate = pa >= ta
    out["stage_b_all_rows"] = {"confusion": _cm(bt, bp), **auc_safe(bt, pb)}
    out["stage_b_after_gate"] = {"confusion": _cm(bt[gate], bp[gate]), **auc_safe(bt[gate], pb[gate])}
    out["flooded_score_pApB"] = auc_safe(bt, pa * pb)
    return out


def test_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    src = df["source"]
    sets = {"a_ga511": df[src == "ga511"]}
    for s in LEGACY_SOURCES[1:]:
        sets[f"b_{s}"] = df[src == s]
    for s in NEW_SOURCES:
        sets[f"c_{s}"] = df[src == s]
    leg = df[src.isin(LEGACY_SOURCES)]
    sets["d_legacy_external"] = leg[leg["source"] != "ga511"]
    sets["d_legacy_all"] = leg
    sets["d_new_sources"] = df[src.isin(NEW_SOURCES)]
    sets["d_all_test"] = df
    return sets


def load_preds(tag: str, name: str = "test_preds.csv") -> pd.DataFrame:
    return pd.read_csv(local(tag) / name, dtype={"camera_id": str, "group_id": str})


def paired(d1: pd.DataFrame, d3: pd.DataFrame) -> dict:
    # v3 - v1 on identical rows
    assert (d1["orig_path"].to_numpy() == d3["orig_path"].to_numpy()).all()
    y, g = d1["label"].to_numpy(), d1["boot_group"]
    f1, f3 = d1["status"].to_numpy() == "flooded", d3["status"].to_numpy() == "flooded"
    out = {"flood_recall": paired_delta(f1, f3, y == "flooded", g)}
    for k in ("dry", "wet", "not_flooded"):
        out[f"false_flood|{k}"] = paired_delta(f1, f3, y == k, g)
    return out


# live frames -----------------------------------------------------------

def smoothed_alerts(f: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    # replay each camera through the deployed TemporalSmoother, in time order
    from inference.smoothing import TemporalSmoother

    rows = []
    for cam, g in f.sort_values("timestamp_utc").groupby("camera_id"):
        sm, prev, alerts, onsets = TemporalSmoother(n=n), None, 0, []
        run = best = 0
        for _, r in g.iterrows():
            s = sm.update(str(cam), r["status"]).status
            if s == "flooded" and prev != "flooded":
                alerts += 1
                onsets.append(r["local_time"])
            prev = s
            run = run + 1 if r["status"] == "flooded" else 0
            best = max(best, run)
        rows.append({"camera_id": cam, "frames": len(g), "raw_flooded": int((g["status"] == "flooded").sum()),
                     "longest_run": best, "alerts": alerts, "alert_onsets": onsets})
    return pd.DataFrame(rows)


def live_metrics(f: pd.DataFrame) -> dict:
    out: dict = {}
    g = f["camera_id"].astype(str)
    views = {"test_cams": f["cam_split"] == "test",
             "test_cams_excl_user_unusable": (f["cam_split"] == "test") & (f["user_label"] != "unusable"),
             "unmapped_cams": f["cam_split"] == "unmapped",
             "val_cams_SEEN": f["cam_split"] == "val",
             "train_cams_SEEN": f["cam_split"] == "train",
             "all_cams": pd.Series(True, index=f.index)}
    day = f["day"].astype(bool)
    for k, m in views.items():
        for tod, tm in (("all", pd.Series(True, index=f.index)), ("day", day), ("night", ~day)):
            mm = (m & tm).to_numpy()
            out[f"{k}|{tod}"] = {"frames": int(mm.sum()), "cameras": int(g[mm].nunique()),
                                 "flooded": rate(f["status"] == "flooded", mm, g),
                                 "wet": rate(f["status"] == "wet", mm, g),
                                 "not_dry": rate(f["status"] != "dry", mm, g)}
    t = f[f["cam_split"] == "test"]
    sa = smoothed_alerts(t)
    out["test_cams_smoothing_n3"] = {
        "cameras": len(sa), "cams_with_any_flooded": int((sa["raw_flooded"] > 0).sum()),
        "cams_with_3plus_flooded": int((sa["raw_flooded"] >= 3).sum()),
        "alerts_total": int(sa["alerts"].sum()), "cams_alerting": int((sa["alerts"] > 0).sum()),
        "alerting": sa[sa["alerts"] > 0].to_dict("records"),
        "top_raw": sa.sort_values("raw_flooded", ascending=False).head(8).to_dict("records")}
    sa_all = smoothed_alerts(f)
    out["all_cams_smoothing_n3"] = {"alerts_total": int(sa_all["alerts"].sum()),
                                    "cams_alerting": int((sa_all["alerts"] > 0).sum())}
    gap = t.sort_values("timestamp_utc").groupby("camera_id")["timestamp_utc"].diff().dropna() / 60
    out["test_cams_sampling_gap_min"] = {"median": float(gap.median()), "p25": float(gap.quantile(0.25)),
                                         "p75": float(gap.quantile(0.75))}
    out["window_local"] = [str(f["local_time"].min()), str(f["local_time"].max())]
    out["max_timestamp_utc"] = int(f["timestamp_utc"].max())
    out["precip_1h_mm"] = {"n_with_value": int(f["precip_1h_mm"].notna().sum()),
                           "max": float(f["precip_1h_mm"].max()),
                           "n_missing": int(f["precip_1h_mm"].isna().sum())}
    return out


def threshold_whatif(d: pd.DataFrame, live: pd.DataFrame | None, ta0: float, tb0: float) -> list:
    # sensitivity only: never pick thresholds from this
    rows = []
    y = d["label"].to_numpy()
    for ta in sorted({0.5, 0.7, round(ta0, 3)}):
        for tb in sorted({round(tb0, 3), 0.5, 0.7, 0.9}):
            st = status_of(d["pA"], d["pB"], ta, tb)
            r = {"tA": ta, "tB": tb, "recall_all": round(float((st[y == "flooded"] == "flooded").mean()), 3)}
            for s in ("flood_master_test", "fred", "roadway_flooding", "eu_flood_2013", "alleyfloodnet"):
                m = (y == "flooded") & (d["source"] == s).to_numpy()
                r[f"recall_{s}"] = round(float((st[m] == "flooded").mean()), 3)
            for k in ("dry", "wet", "not_flooded"):
                r[f"flooded|{k}"] = int((st[y == k] == "flooded").sum())
            if live is not None:
                lt = (live["cam_split"] == "test").to_numpy()
                r["live_test_flooded"] = int((status_of(live["pA"], live["pB"], ta, tb)[lt] == "flooded").sum())
            rows.append(r)
    return rows


# figure ----------------------------------------------------------------

def _axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def pr_figure(preds: dict[str, pd.DataFrame], live: dict[str, pd.DataFrame] | None = None) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURF)
    for ax, (title, sel) in zip((axs[0], axs[2]), (
            ("Legacy test (1,108 rows)", lambda d: d["source"].isin(LEGACY_SOURCES)),
            ("New sources (534 rows)", lambda d: d["source"].isin(NEW_SOURCES)))):
        _axes(ax, title, "flood recall", "flood precision")
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.02)
        for tag, col in zip(TAGS, (C1, C2)):
            d = preds[tag]
            d = d[sel(d).to_numpy()]
            y = (d["label"] == "flooded").astype(int).to_numpy()
            sc = (d["pA"] * d["pB"]).to_numpy()
            op = (d["status"] == "flooded").to_numpy()
            tp = int((op & (y == 1)).sum())
            p, r, _ = precision_recall_curve(y, sc)
            ax.plot(r, p, color=col, lw=2, label=f"{tag} (AP {average_precision_score(y, sc):.2f})")
            ax.plot([tp / y.sum()], [tp / max(op.sum(), 1)], "o", ms=9, color=col, mec=SURF, mew=1.5, zorder=5)
        ax.legend(fontsize=8, frameon=False, loc="lower left", labelcolor=INK)
    # greek recall vs live false floods, same score sweep
    ax = axs[1]
    _axes(ax, "Greek video recall vs live false floods", "live test-camera frames called flooded (of 1,602)",
          "Greek video recall (62 frames)")
    for tag, col in zip(TAGS, (C1, C2)):
        if not live or tag not in live:
            continue
        d, lv = preds[tag], live[tag]
        g = (d.loc[d["source"] == "flood_master_test", "pA"] * d.loc[d["source"] == "flood_master_test", "pB"]).to_numpy()
        ls = (lv.loc[lv["cam_split"] == "test", "pA"] * lv.loc[lv["cam_split"] == "test", "pB"]).to_numpy()
        thr = np.unique(np.concatenate([g, ls]))[::-1]
        fp = np.array([(ls >= t).sum() for t in thr])
        rc = np.array([(g >= t).mean() for t in thr])
        keep = fp <= 60
        ax.step(fp[keep], rc[keep], where="post", color=col, lw=2, label=tag)
        gd = d[d["source"] == "flood_master_test"]
        ax.plot([(lv.loc[lv["cam_split"] == "test", "status"] == "flooded").sum()],
                [(gd["status"] == "flooded").mean()], "o", ms=9, color=col, mec=SURF, mew=1.5, zorder=5)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 60)
    ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK)
    fig.text(0.01, 0.01, "curves sweep one threshold on pA*pB; dots = shipped two-threshold rule (tA, tB). test split only. "
             "Middle: one video = one cluster, live frames are presumed dry (0 mm rain).", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = FIGS / "v3_vs_v1_pr_flooded.png"
    fig.savefig(p, dpi=130, facecolor=SURF)
    plt.close(fig)
    return str(p)


def main() -> dict:
    res: dict = {"sets": {}, "thresholds": {}, "preprocess": {}}
    preds = {t: load_preds(t) for t in TAGS}
    for t in TAGS:
        cfg = load_config(model_dir(t))
        ta, tb = thresholds(cfg)
        res["thresholds"][t] = {"tA": ta, "tB": tb, "run_a": cfg["stage_a"]["run_id"],
                                "run_b": cfg["stage_b"]["run_id"], "data_version": cfg.get("data_version")}
        res["preprocess"][t] = cfg.get("preprocess")
        for k, v in test_sets(preds[t]).items():
            res["sets"].setdefault(k, {})[t] = set_metrics(v, ta, tb)
    s1, s3 = test_sets(preds["v1"]), test_sets(preds["v3"])
    res["paired_v3_minus_v1"] = {k: paired(s1[k], s3[k]) for k in s1}
    res["live"] = {}
    for t in TAGS:
        lp = local(t) / "live_preds.csv"
        if lp.exists():
            res["live"][t] = live_metrics(pd.read_csv(lp, dtype={"camera_id": str}))
    lv = local("v3") / "live_preds.csv"
    ta, tb = res["thresholds"]["v3"]["tA"], res["thresholds"]["v3"]["tB"]
    res["threshold_whatif_v3"] = threshold_whatif(preds["v3"], pd.read_csv(lv) if lv.exists() else None, ta, tb)
    for t in TAGS:
        rp = local(t) / "test_preds_raw.csv"
        if rp.exists():
            d = pd.read_csv(rp, dtype={"camera_id": str, "group_id": str})
            th = res["thresholds"][t]
            res.setdefault("raw_input_sensitivity", {})[t] = {
                k: {"flood_recall": set_metrics(v, th["tA"], th["tB"])["per_class"]["flooded"]["recall"]}
                for k, v in test_sets(d).items() if k.startswith("d_")}
    lives = {t: pd.read_csv(local(t) / "live_preds.csv") for t in TAGS if (local(t) / "live_preds.csv").exists()}
    res["figure"] = pr_figure(preds, lives)
    (REPORTS / "eval_v3_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    for k, v in res["sets"].items():
        for t in TAGS:
            m = v[t]
            print(f"{k:24s} {t} n={m['n']:4d} rec={fmt_rate(m['per_class']['flooded']['recall'])}")
            print(f"{'':28s} prec={fmt_rate(m['per_class']['flooded']['precision'])}")
            print(f"{'':28s} ff dry={fmt_rate(m['false_flood']['dry'])} wet={fmt_rate(m['false_flood']['wet'])}"
                  f" nf={fmt_rate(m['false_flood']['not_flooded'])}")
    return res


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
