# contact sheets for spot-checking labels
from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

from prep.build_manifest import apply_clip_filter, compute_labels, gather_rows
from prep.mask_rules import nysdot_crop
from prep.reports import contact_sheet

REPORTS = Path("reports")
SEED = 20260925


def _caption(r: dict) -> str:
    wf = r.get("water_frac_road")
    wf_s = f"{wf:.2f}" if isinstance(wf, (int, float)) and not math.isnan(wf) else "n/a"
    outcome = "DROP" if r.get("_drop") else r.get("label", "?")
    return f"{r['source']} | {outcome} | wf={wf_s} | {Path(r['orig_path']).name}"


def _entry_item(r: dict):
    if r.get("needs_nysdot_crop"):
        from PIL import Image
        return nysdot_crop(Image.open(r["orig_path"]).convert("RGB"))
    return r["orig_path"]


def mask_label_spotcheck(rows: list[dict], per_stratum: int = 15) -> dict[str, int]:
    rng = random.Random(SEED)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["source"] == "ga511":
            continue
        outcome = "DROP" if r.get("_drop") else r.get("label", "?")
        strata[(r["source"], outcome)].append(r)

    counts = {}
    for (source, outcome), items in sorted(strata.items()):
        sample = items if len(items) <= per_stratum else rng.sample(items, per_stratum)
        entries = [(_entry_item(r), _caption(r)) for r in sample]
        out_path = REPORTS / f"spotcheck_mask_{source}_{outcome}.png"
        contact_sheet(entries, out_path, cols=5, title=f"{source} / {outcome} (n={len(items)}, showing {len(sample)})")
        counts[str(out_path)] = len(sample)
    return counts


def clip_spotcheck(rows: list[dict], per_stratum: int = 20) -> dict[str, int]:
    rng = random.Random(SEED + 1)
    by_source_kept: dict[str, list[dict]] = defaultdict(list)
    by_source_removed: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["source"] == "ga511" or not r.get("_clip_scored"):
            continue
        if r.get("_drop") and r.get("_clip_reason"):
            by_source_removed[r["source"]].append(r)
        elif not r.get("_drop"):
            by_source_kept[r["source"]].append(r)

    counts = {}
    for source, items in sorted(by_source_removed.items()):
        sample = items if len(items) <= per_stratum else rng.sample(items, per_stratum)
        entries = [(r["orig_path"], f"REMOVED | {r.get('_clip_reason', '')} | {Path(r['orig_path']).name}") for r in sample]
        out_path = REPORTS / f"clip_removed_{source}.png"
        contact_sheet(entries, out_path, cols=5, title=f"CLIP removed: {source} (n={len(items)}, showing {len(sample)})")
        counts[str(out_path)] = len(sample)
    for source, items in sorted(by_source_kept.items()):
        sample = items if len(items) <= per_stratum else rng.sample(items, per_stratum)
        entries = [(r["orig_path"], f"KEPT | {Path(r['orig_path']).name}") for r in sample]
        out_path = REPORTS / f"clip_kept_{source}.png"
        contact_sheet(entries, out_path, cols=5, title=f"CLIP kept sample: {source} (n={len(items)}, showing {len(sample)})")
        counts[str(out_path)] = len(sample)
    return counts


def main() -> None:
    rows = gather_rows()
    compute_labels(rows)
    mask_counts = mask_label_spotcheck(rows)
    print("mask spotcheck sheets:")
    for k, v in mask_counts.items():
        print(f"  {k}: {v}")

    from prep.build_manifest import CLIP_FILTERED_SOURCES

    clip_stats = apply_clip_filter(rows, skip=False)
    for r in rows:
        if r["source"] in CLIP_FILTERED_SOURCES:
            r["_clip_scored"] = True
    clip_counts = clip_spotcheck(rows)
    print("clip spotcheck sheets:")
    for k, v in clip_counts.items():
        print(f"  {k}: {v}")
    print("clip stats:", clip_stats)


if __name__ == "__main__":
    main()

