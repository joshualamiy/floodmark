"""Report generation: contact sheets (spot-check/CLIP/sample grids), the
tracked class-counts report, and the augmentation before/after grid.

Image-bearing outputs go to `reports/*.png` (gitignored, local only, per the
repo's public-data rule). Only `reports/class_counts.md` and
`reports/figures/class_counts.png` (a bar chart, no dataset pixels) are
tracked.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL_W, CELL_H = 220, 170
THUMB_H = 130
FONT = None


def _font():
    global FONT
    if FONT is None:
        try:
            FONT = ImageFont.load_default()
        except Exception:  # noqa: BLE001
            FONT = None
    return FONT


def _load_thumb(path_or_array, thumb_w: int, thumb_h: int) -> Image.Image:
    if isinstance(path_or_array, np.ndarray):
        img = Image.fromarray(path_or_array)
    elif isinstance(path_or_array, Image.Image):
        img = path_or_array
    else:
        img = Image.open(path_or_array).convert("RGB")
    img = img.copy()
    img.thumbnail((thumb_w, thumb_h))
    canvas = Image.new("RGB", (thumb_w, thumb_h), (30, 30, 30))
    x = (thumb_w - img.width) // 2
    y = (thumb_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def contact_sheet(entries: list[tuple], out_path: Path, cols: int = 6,
                   cell_w: int = CELL_W, cell_h: int = CELL_H, thumb_h: int = THUMB_H,
                   title: str | None = None) -> Path:
    """`entries`: list of (path_or_ndarray, caption_str). Missing/unreadable
    images are rendered as a red placeholder tile with the caption, so one
    bad path never kills the whole sheet.
    """
    n = len(entries)
    rows = max(1, (n + cols - 1) // cols)
    title_h = 30 if title else 0
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((8, 6), title, fill=(0, 0, 0), font=_font())

    for i, (item, caption) in enumerate(entries):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, title_h + r * cell_h
        try:
            thumb = _load_thumb(item, cell_w - 8, thumb_h)
        except Exception:  # noqa: BLE001
            thumb = Image.new("RGB", (cell_w - 8, thumb_h), (140, 30, 30))
        sheet.paste(thumb, (x0 + 4, y0 + 4))
        # wrap caption crudely at ~34 chars/line, up to 3 lines
        words = str(caption).split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 34:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)
        for li, line in enumerate(lines[:3]):
            draw.text((x0 + 4, y0 + thumb_h + 6 + li * 11), line, fill=(0, 0, 0), font=_font())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def write_class_counts_report(rows: list[dict], out_md: Path, out_png: Path) -> None:
    by_split_label: Counter = Counter()
    by_source_label: Counter = Counter()
    by_split_source: Counter = Counter()
    label_source_counts: Counter = Counter()
    water_fracs_by_label: dict[str, list[float]] = {"dry": [], "wet": [], "flooded": []}

    for r in rows:
        split = r.get("split", "?")
        label = r.get("label", "?")
        source = r.get("source", "?")
        by_split_label[(split, label)] += 1
        by_source_label[(source, label)] += 1
        by_split_source[(split, source)] += 1
        label_source_counts[r.get("label_source", "?")] += 1
        wf = r.get("water_frac_road")
        if isinstance(wf, (int, float)) and not math.isnan(wf) and label in water_fracs_by_label:
            water_fracs_by_label[label].append(float(wf))

    lines = ["# Class counts", ""]
    lines.append("## Per split x label")
    lines.append("")
    lines.append("| split | dry | wet | flooded | total |")
    lines.append("|---|---|---|---|---|")
    for split in ("train", "val", "test"):
        d = by_split_label.get((split, "dry"), 0)
        w = by_split_label.get((split, "wet"), 0)
        f = by_split_label.get((split, "flooded"), 0)
        lines.append(f"| {split} | {d} | {w} | {f} | {d + w + f} |")
    lines.append("")

    lines.append("## Per source x label")
    lines.append("")
    sources = sorted({s for (s, _l) in by_source_label})
    lines.append("| source | dry | wet | flooded | total |")
    lines.append("|---|---|---|---|---|")
    for s in sources:
        d = by_source_label.get((s, "dry"), 0)
        w = by_source_label.get((s, "wet"), 0)
        f = by_source_label.get((s, "flooded"), 0)
        lines.append(f"| {s} | {d} | {w} | {f} | {d + w + f} |")
    lines.append("")

    lines.append("## Per split x source")
    lines.append("")
    lines.append("| split | " + " | ".join(sources) + " |")
    lines.append("|---|" + "---|" * len(sources))
    for split in ("train", "val", "test"):
        vals = [str(by_split_source.get((split, s), 0)) for s in sources]
        lines.append(f"| {split} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("## label_source counts")
    lines.append("")
    lines.append("| label_source | count |")
    lines.append("|---|---|")
    for k, v in sorted(label_source_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## water_frac_road stats by assigned label")
    lines.append("")
    lines.append("| label | n | mean | median | p10 | p90 |")
    lines.append("|---|---|---|---|---|---|")
    for label, vals in water_fracs_by_label.items():
        if not vals:
            lines.append(f"| {label} | 0 | - | - | - | - |")
            continue
        arr = np.array(vals)
        lines.append(
            f"| {label} | {len(arr)} | {arr.mean():.3f} | {np.median(arr):.3f} | "
            f"{np.percentile(arr, 10):.3f} | {np.percentile(arr, 90):.3f} |"
        )
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))

    # bar chart: counts per split, stacked by label. No dataset pixels.
    fig, ax = plt.subplots(figsize=(6, 4))
    splits = ("train", "val", "test")
    labels = ("dry", "wet", "flooded")
    colors = {"dry": "#8c8c8c", "wet": "#4C9BE8", "flooded": "#D64545"}
    bottoms = np.zeros(len(splits))
    x = np.arange(len(splits))
    for label in labels:
        vals = np.array([by_split_label.get((s, label), 0) for s in splits], dtype=float)
        ax.bar(x, vals, bottom=bottoms, label=label, color=colors[label])
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("images")
    ax.set_title("Class counts per split")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-(source, label) spot-check table (orchestrator ruling, Phase 2 resume):
# the "wet band" and "dry-at-zero" mask rules are only trusted per source
# once >= 20 images of that bucket have been viewed with >= 80% agreement.
# These rows are the record of that manual review (see
# docs/phase_reports/phase2_prep.md for the narrative and the contact sheets
# under reports/spotcheck_*.png for the images themselves). Recomputing the
# viewing/agreement numbers isn't possible from the manifest alone (it's a
# human judgment), so this table is a static record, refreshed by hand only
# when the rule set or population sizes change enough to warrant a re-check.
# ---------------------------------------------------------------------------
SPOTCHECK_TABLE = [
    # (source, label, rule, population, n_viewed, agreement, verdict)
    ("roadway_flooding", "flooded", "mask f>=0.10", 429, 20, "20/20 (100%)", "kept"),
    ("roadway_flooding", "wet", "mask wet-band (unverified)", 8, 8, "~6/8 (75%)", "EXCLUDED (n<20)"),
    ("roadway_flooding", "dry", "mask dry-at-zero (unverified)", 1, 1, "0/1 (0%; road-closed/flood-adjacent scene, not dry)", "EXCLUDED (n<20)"),
    ("fred", "flooded", "mask road+water-hazard, f>=0.10", 1865, 20, "20/20 (100%)", "kept"),
    ("fred", "wet", "mask wet-band (verified)", 164, 20, "~17/20 (85%; dry near-field, flooded road visible ahead)", "kept"),
    ("fred", "dry", "mask dry-at-zero, within flooded sequences (verified)", 199, 20, "20/20 (100%)", "kept"),
    ("fred", "dry", "sequence_condition (always allowed)", 2616, 20, "20/20 (100%)", "kept"),
    ("flood_master_test", "flooded", "mask bottom-band (Greek video only)", 567, 20, "20/20 (100%)", "kept (Italian video excluded, aerial/drone news footage)"),
    ("nysdot_road_surface", "dry", "dataset_label, agreement>=0.67, cropped", 29, 15, "~13/15 (87%)", "kept"),
    ("nysdot_road_surface", "wet", "dataset_label, agreement>=0.67, cropped", 30, 15, "~14/15 (93%)", "kept"),
    ("ga511", "dry/wet/unusable", "ai_review (visual, test-split cameras)", 274, 274, "n/a -- this IS the label (167 in the first pass, 107 more after the 511GA sweep finished)", "recorded to ai_review_labels.csv"),
]

# Sources excluded wholesale (never gathered at all, so they don't appear in
# the per-source table above) -- orchestrator asked these be accounted for
# explicitly rather than silently missing. See docs/phase_reports/phase2_prep.md.
EXCLUDED_SOURCES_TABLE = [
    # (source/group, n_images, reason)
    (
        "flood_area_segmentation",
        290,
        (
            "Kaggle Flood Area Segmentation: a 12-image random spot-check found the "
            "source is elevated drone/aerial flood photography (whole towns/fields "
            "from above), not the ground-level road view this classifier targets. "
            "CLIP margins were also weakly negative on average across all 290 "
            "images (mean pos-neg margin -0.017, max +0.057), so a per-image CLIP "
            "threshold would keep only obviously-aerial images or exclude nearly "
            "everything. Excluded wholesale rather than row-by-row; never gathered "
            "by build_manifest.gather_rows."
        ),
    ),
    (
        "flood_master_test (Italian video)",
        1406,
        (
            "Flood Master Database Italian test video: 24 frames spread across the "
            "full video are drone/news b-roll (on-screen NYTimes credit, panning "
            "and tilting aerial shots of vineyards and rooftops), not a fixed "
            "road-camera view. Excluded; the Greek video (567 frames, a fixed "
            "elevated camera over a flooded street) was reinstated and kept -- see "
            "the spot-check table below."
        ),
    ),
]


def append_pipeline_report_sections(
    out_md: Path,
    clip_stats: dict | None = None,
    dedup_stats: dict | None = None,
    spotcheck_table: list[tuple] | None = None,
    excluded_sources_table: list[tuple] | None = None,
) -> None:
    """Appends CLIP removal counts, dedup/thinning counts, wholesale-excluded
    sources, and the per-(source, label) spot-check table to an existing
    class-counts report (call after `write_class_counts_report`).
    """
    spotcheck_table = spotcheck_table if spotcheck_table is not None else SPOTCHECK_TABLE
    excluded_sources_table = (
        excluded_sources_table if excluded_sources_table is not None else EXCLUDED_SOURCES_TABLE
    )
    lines = ["", "## Sources excluded wholesale (not in the per-source table above)", ""]
    lines.append("| source | n images | reason |")
    lines.append("|---|---|---|")
    for source, n, reason in excluded_sources_table:
        lines.append(f"| {source} | {n} | {reason} |")
    lines.append("")

    lines.append("## CLIP road-scene filter (removed per source)")
    lines.append("")
    lines.append("| source | kept | removed | removed frac |")
    lines.append("|---|---|---|---|")
    for source, stats in sorted((clip_stats or {}).items()):
        kept, removed = stats.get("kept", 0), stats.get("removed", 0)
        total = kept + removed
        frac = f"{removed / total:.1%}" if total else "-"
        lines.append(f"| {source} | {kept} | {removed} | {frac} |")
    lines.append("")

    lines.append("## Dedup / temporal thinning")
    lines.append("")
    dedup_stats = dedup_stats or {}
    for k, v in dedup_stats.items():
        if k == "thinning_by_group":
            continue
        lines.append(f"- {k}: {v}")
    thinning_by_group = dedup_stats.get("thinning_by_group") or {}
    if thinning_by_group:
        lines.append("")
        lines.append("| video/sequence group | frames before | frames after |")
        lines.append("|---|---|---|")
        for group, counts in sorted(thinning_by_group.items()):
            lines.append(f"| {group} | {counts['before']} | {counts['after']} |")
    lines.append("")

    lines.append("## Per-(source, label) spot-check table")
    lines.append("")
    lines.append("| source | label | rule | population | n viewed | agreement | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for source, label, rule, population, n_viewed, agreement, verdict in spotcheck_table:
        lines.append(f"| {source} | {label} | {rule} | {population} | {n_viewed} | {agreement} | {verdict} |")
    lines.append("")

    with open(out_md, "a") as f:
        f.write("\n".join(lines))
