# download + characterize street/elevated flood photo datasets
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FLOOD_ML_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = FLOOD_ML_ROOT / "data" / "raw"
LOG_DIR = FLOOD_ML_ROOT / "logs" / "jobs"
REPORTS_DIR = FLOOD_ML_ROOT / "reports"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


EU_FLOOD_DEST = DATA_RAW / "eu_flood_2013"
EU_FLOOD_GITHUB_RAW = "https://raw.githubusercontent.com/cvjena/eu-flood-dataset/master"
EU_FLOOD_ANNOTATION_FILES = [
    "metadata.json",
    "relevance/flooding.txt",
    "relevance/depth.txt",
    "relevance/pollution.txt",
    "relevance/irrelevant.txt",
    "queries/flooding.txt",
    "queries/depth.txt",
    "queries/pollution.txt",
    "important_regions/flooding.json",
    "important_regions/depth.json",
    "important_regions/pollution.json",
]
EU_FLOOD_IMAGES_ZIP_URL = "https://archive.org/download/european-flood-2013/european-flood-2013_imgs_small.zip"


FLOODIMG_DEST = DATA_RAW / "floodimg"
FLOODIMG_KAGGLE_REF = "hhrclemson/flooding-image-dataset"
FLOODIMG_SAMPLE_DEFAULT = 2000
FLOODIMG_SEED = 42


ALLEYFN_DEST = DATA_RAW / "alleyfloodnet"
ALLEYFN_KAGGLE_REF = "seonyseony/alleyfloodnet"


def _log_path(name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{name}.log"


def _nonempty_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _retry(fn, *args, retries: int = 5, backoff: float = 3.0, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def download_eu_flood_2013() -> None:
    dest = EU_FLOOD_DEST
    dest.mkdir(parents=True, exist_ok=True)

    for rel in EU_FLOOD_ANNOTATION_FILES:
        local = dest / rel
        if local.exists():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f"{EU_FLOOD_GITHUB_RAW}/{rel}"
        print(f"eu_flood_2013: downloading {rel}")
        _retry(urllib.request.urlretrieve, url, str(local))

    img_dir = dest / "images"
    if _nonempty_dir(img_dir):
        print("eu_flood_2013: images already present, skipping zip download")
        return
    zip_path = dest / "european-flood-2013_imgs_small.zip"
    if not zip_path.exists():
        print("eu_flood_2013: downloading images zip (~1.1 GB, archive.org)")
        _retry(urllib.request.urlretrieve, EU_FLOOD_IMAGES_ZIP_URL, str(zip_path))
    img_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX") or "/._" in name:
                continue
            zf.extract(name, img_dir)
    zip_path.unlink()
    print("eu_flood_2013: extracted images")


def parse_eu_flood_labels(
    all_ids: list[str], flooding_ids: set[str], irrelevant_ids: set[str]
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for image_id in all_ids:
        if image_id in irrelevant_ids:
            labels[image_id] = "not_flooded"
        elif image_id in flooding_ids:
            labels[image_id] = "flooded"
        else:
            labels[image_id] = "unknown"
    return labels


def _read_id_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _kaggle_api():
    import kaggle

    kaggle.api.authenticate()
    return kaggle.api


def _list_floodimg_files() -> list[dict[str, Any]]:
    cache = FLOODIMG_DEST / "file_list.json"
    if cache.exists():
        return json.loads(cache.read_text())

    api = _kaggle_api()
    files: list[dict[str, Any]] = []
    token = None
    while True:
        res = _retry(api.dataset_list_files, FLOODIMG_KAGGLE_REF, page_token=token, page_size=200)
        files.extend({"name": f.name, "bytes": f.total_bytes} for f in res.files)
        token = res.next_page_token
        time.sleep(1.0)
        if not token:
            break
    FLOODIMG_DEST.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(files))
    return files


def deterministic_sample(names: list[str], n: int, seed: int) -> list[str]:
    ordered = sorted(names)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return sorted(ordered[: min(n, len(ordered))])


def floodimg_local_name(kaggle_path: str) -> str:
    return Path(kaggle_path).name


def _kaggle_download_one(api, dest_dir: Path, kaggle_name: str) -> None:
    target = dest_dir / Path(kaggle_name).name
    _retry(api.dataset_download_file, FLOODIMG_KAGGLE_REF, kaggle_name, path=str(dest_dir), force=True)
    zip_path = dest_dir / (target.name + ".zip")
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(target.name, dest_dir)
        zip_path.unlink()


def download_floodimg(sample_size: int = FLOODIMG_SAMPLE_DEFAULT, seed: int = FLOODIMG_SEED) -> None:
    dest = FLOODIMG_DEST
    dest.mkdir(parents=True, exist_ok=True)
    api = _kaggle_api()
    files = _list_floodimg_files()

    ann_dest = dest / "annotation_sample"
    ann_dest.mkdir(exist_ok=True)
    ann_names = [f["name"] for f in files if f["name"].startswith("Annotation/")]
    for name in ann_names:
        if (ann_dest / Path(name).name).exists():
            continue
        _kaggle_download_one(api, ann_dest, name)
        time.sleep(0.4)

    img_names = [f["name"] for f in files if f["name"].startswith("Flood Images/")]
    selected = deterministic_sample(img_names, sample_size, seed)
    img_dest = dest / "images"
    img_dest.mkdir(exist_ok=True)
    n_new = 0
    for name in selected:
        if (img_dest / floodimg_local_name(name)).exists():
            continue
        _kaggle_download_one(api, img_dest, name)
        n_new += 1
        time.sleep(0.35)

    (dest / "sample_manifest.json").write_text(
        json.dumps({"seed": seed, "sample_size": sample_size, "selected": selected}, indent=2)
    )
    print(f"floodimg: {n_new} new file(s) downloaded, {len(selected)} images in sample")


def download_alleyfloodnet() -> None:
    import subprocess

    dest = ALLEYFN_DEST
    if _nonempty_dir(dest / "AlleyFloodNet"):
        print("alleyfloodnet: already present, skipping")
        return
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", ALLEYFN_KAGGLE_REF, "-p", str(dest), "--unzip"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "metadata", "-d", ALLEYFN_KAGGLE_REF, "-p", str(dest)],
        check=True,
    )


DOWNLOADERS = {
    "eu_flood_2013": lambda **kw: download_eu_flood_2013(),
    "floodimg": lambda sample_size=FLOODIMG_SAMPLE_DEFAULT, seed=FLOODIMG_SEED, **kw: download_floodimg(
        sample_size, seed
    ),
    "alleyfloodnet": lambda **kw: download_alleyfloodnet(),
}


def download(dataset: str, sample_size: int, seed: int) -> None:
    names = list(DOWNLOADERS) if dataset == "all" else [dataset]
    for name in names:
        DOWNLOADERS[name](sample_size=sample_size, seed=seed)


def _check_images(paths: list[Path]) -> dict[str, Any]:
    from PIL import Image

    corrupt: list[str] = []
    total_bytes = 0
    for p in paths:
        try:
            size = p.stat().st_size
        except OSError:
            corrupt.append(str(p))
            continue
        total_bytes += size
        if size == 0:
            corrupt.append(str(p))
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
        except Exception:  # noqa: BLE001
            corrupt.append(str(p))
    return {"count": len(paths), "bytes": total_bytes, "corrupt": corrupt}


def verify_eu_flood_2013() -> dict[str, Any]:
    dest = EU_FLOOD_DEST
    img_dir = dest / "images"
    paths = sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    result = _check_images(paths)

    all_ids = []
    if (dest / "metadata.json").exists():
        meta = json.loads((dest / "metadata.json").read_text())
        all_ids = [str(m["pageid"]) for m in meta]
    flooding_ids = _read_id_list(dest / "relevance" / "flooding.txt")
    irrelevant_ids = _read_id_list(dest / "relevance" / "irrelevant.txt")
    labels = parse_eu_flood_labels(all_ids, flooding_ids, irrelevant_ids)
    label_counts: dict[str, int] = {}
    for v in labels.values():
        label_counts[v] = label_counts.get(v, 0) + 1

    return {
        "present": img_dir.is_dir(),
        "images": {"count": result["count"], "bytes": result["bytes"], "n_corrupt": len(result["corrupt"])},
        "corrupt": result["corrupt"],
        "wikimedia_metadata_rows": len(all_ids),
        "relevance_flooding_ids": len(flooding_ids),
        "relevance_irrelevant_ids": len(irrelevant_ids),
        "label_counts_of_metadata_ids": label_counts,
    }


def verify_floodimg() -> dict[str, Any]:
    dest = FLOODIMG_DEST
    img_dir = dest / "images"
    paths = sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    result = _check_images(paths)
    total_available = 0
    if (dest / "file_list.json").exists():
        files = json.loads((dest / "file_list.json").read_text())
        total_available = sum(1 for f in files if f["name"].startswith("Flood Images/"))
    return {
        "present": img_dir.is_dir(),
        "images": {"count": result["count"], "bytes": result["bytes"], "n_corrupt": len(result["corrupt"])},
        "corrupt": result["corrupt"],
        "total_available_upstream": total_available,
        "note": "every image is 'flooded' by dataset scope; no non-flooded images",
    }


ALLEYFN_SPLIT_DIRS = ["flood_train_", "flood_test_"]
ALLEYFN_LABEL_DIRS = {"flooded": "flooded", "non_flooded": "not_flooded"}


def verify_alleyfloodnet() -> dict[str, Any]:
    dest = ALLEYFN_DEST / "AlleyFloodNet"
    counts: dict[str, int] = {}
    all_paths: list[Path] = []
    for split in ALLEYFN_SPLIT_DIRS:
        for folder in ALLEYFN_LABEL_DIRS:
            d = dest / split / folder
            paths = sorted(p for p in d.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if d.is_dir() else []
            counts[f"{split}/{folder}"] = len(paths)
            all_paths.extend(paths)
    result = _check_images(all_paths)
    return {
        "present": dest.is_dir(),
        "images": {"count": result["count"], "bytes": result["bytes"], "n_corrupt": len(result["corrupt"])},
        "corrupt": result["corrupt"],
        "counts_by_split_label": counts,
    }


VERIFIERS = {
    "eu_flood_2013": verify_eu_flood_2013,
    "floodimg": verify_floodimg,
    "alleyfloodnet": verify_alleyfloodnet,
}


def verify(dataset: str) -> dict[str, Any]:
    names = list(VERIFIERS) if dataset == "all" else [dataset]
    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    for name in names:
        entry = VERIFIERS[name]()
        report["datasets"][name] = entry
        out_dir = DATA_RAW / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "inventory.json").write_text(json.dumps(entry, indent=2, sort_keys=True))

    log_path = _log_path("flood_photos_verify")
    with open(log_path, "w") as f:
        f.write(f"flood_photos --verify run at {report['generated_at']}\n\n")
        f.write(json.dumps(report, indent=2))
    print(f"wrote inventory.json per dataset, and {log_path}")
    return report


def _eu_flood_rows() -> list[dict[str, Any]]:
    dest = EU_FLOOD_DEST
    meta = json.loads((dest / "metadata.json").read_text())
    meta_by_id = {str(m["pageid"]): m for m in meta}
    flooding_ids = _read_id_list(dest / "relevance" / "flooding.txt")
    irrelevant_ids = _read_id_list(dest / "relevance" / "irrelevant.txt")
    pollution_ids = _read_id_list(dest / "relevance" / "pollution.txt")

    img_dir = dest / "images"
    rows = []
    for p in sorted(img_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        image_id = p.stem
        m = meta_by_id.get(image_id)
        if m is not None:
            label = parse_eu_flood_labels([image_id], flooding_ids, irrelevant_ids)[image_id]
            license_str = m.get("license", "unverified")
            group_id = f"euflood_user_{m.get('user', 'unknown')}"
        elif image_id in pollution_ids or image_id.startswith("pollution_"):
            label = "unknown"
            license_str = "unverified (manually harvested search-engine image, not Wikimedia)"
            group_id = "euflood_pollution_batch"
        else:
            label = "unknown"
            license_str = "unverified"
            group_id = "euflood_unmatched"
        rows.append(
            {
                "path": str(p.relative_to(FLOOD_ML_ROOT)),
                "label": label,
                "label_source": "dataset_label",
                "road_score": "",
                "view_guess": "unknown",
                "group_id": group_id,
                "license": license_str,
                "phash": "",
            }
        )
    return rows


def _floodimg_rows() -> list[dict[str, Any]]:
    img_dir = FLOODIMG_DEST / "images"
    rows = []
    for p in sorted(img_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rows.append(
            {
                "path": str(p.relative_to(FLOOD_ML_ROOT)),
                "label": "flooded",
                "label_source": "dataset_label",
                "road_score": "",
                "view_guess": "unknown",
                "group_id": "floodimg_unknown",
                "license": "CC0-1.0",
                "phash": "",
            }
        )
    return rows


def _alleyfloodnet_rows() -> list[dict[str, Any]]:
    dest = ALLEYFN_DEST / "AlleyFloodNet"
    rows = []
    for split in ALLEYFN_SPLIT_DIRS:
        for folder, label in ALLEYFN_LABEL_DIRS.items():
            d = dest / split / folder
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*")):
                if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                    continue
                rows.append(
                    {
                        "path": str(p.relative_to(FLOOD_ML_ROOT)),
                        "label": label,
                        "label_source": "dataset_label",
                        "road_score": "",
                        "view_guess": "unknown",
                        "group_id": "alleyfloodnet_unknown",
                        "license": "CC BY 4.0 (dataset-level grant; per-photo origin unverified -- see report)",
                        "phash": "",
                    }
                )
    return rows


ROW_BUILDERS = {
    "eu_flood_2013": _eu_flood_rows,
    "floodimg": _floodimg_rows,
    "alleyfloodnet": _alleyfloodnet_rows,
}


def _add_phash(rows: list[dict[str, Any]]) -> None:
    import imagehash
    from PIL import Image

    for row in rows:
        p = FLOOD_ML_ROOT / row["path"]
        try:
            with Image.open(p) as im:
                row["phash"] = str(imagehash.phash(im.convert("RGB")))
        except Exception:  # noqa: BLE001
            row["phash"] = ""


def _add_road_scores(rows: list[dict[str, Any]]) -> None:
    sys.path.insert(0, str(FLOOD_ML_ROOT / "src"))
    from prep.clip_filter import load_clip, score_batch

    bundle = load_clip()
    paths = [str(FLOOD_ML_ROOT / row["path"]) for row in rows]
    results = score_batch(paths, bundle=bundle)
    for row, (is_road, pos, neg) in zip(rows, results):
        row["road_score"] = round(pos - neg, 4)


def characterize(dataset: str, skip_clip: bool = False) -> None:
    names = list(ROW_BUILDERS) if dataset == "all" else [dataset]
    for name in names:
        rows = ROW_BUILDERS[name]()
        print(f"{name}: {len(rows)} images, computing phash...")
        _add_phash(rows)
        if not skip_clip:
            print(f"{name}: scoring road-scene relevance with CLIP...")
            _add_road_scores(rows)
        import csv

        out_path = DATA_RAW / name / "candidates.csv"
        fieldnames = ["path", "label", "label_source", "road_score", "view_guess", "group_id", "license", "phash"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_path} ({len(rows)} rows)")


def contact_sheet(dataset: str, n: int, seed: int, out: Path | None = None, cols: int = 8) -> Path:
    from PIL import Image

    candidates_path = DATA_RAW / dataset / "candidates.csv"
    import csv

    with open(candidates_path) as f:
        paths = [row["path"] for row in csv.DictReader(f)]
    rng = random.Random(seed)
    sample = rng.sample(paths, min(n, len(paths)))

    thumb = 160
    rows = (len(sample) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "black")
    for i, rel in enumerate(sample):
        try:
            with Image.open(FLOOD_ML_ROOT / rel) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb))
                x, y = (i % cols) * thumb, (i // cols) * thumb
                sheet.paste(im, (x, y))
        except Exception:  # noqa: BLE001, S112
            continue

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out or REPORTS_DIR / f"flood_photos_{dataset}_contact_sheet.png"
    sheet.save(out_path)
    (out_path.with_suffix(".json")).write_text(json.dumps(sample, indent=2))
    print(f"wrote {out_path} ({len(sample)} images, {cols} cols)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download")
    dl.add_argument("--dataset", choices=["all", *DOWNLOADERS.keys()], default="all")
    dl.add_argument("--sample-size", type=int, default=FLOODIMG_SAMPLE_DEFAULT)
    dl.add_argument("--seed", type=int, default=FLOODIMG_SEED)

    vf = sub.add_parser("verify")
    vf.add_argument("--dataset", choices=["all", *VERIFIERS.keys()], default="all")

    ch = sub.add_parser("characterize")
    ch.add_argument("--dataset", choices=["all", *ROW_BUILDERS.keys()], default="all")
    ch.add_argument("--skip-clip", action="store_true")

    cs = sub.add_parser("contact-sheet")
    cs.add_argument("--dataset", choices=list(ROW_BUILDERS.keys()), required=True)
    cs.add_argument("--n", type=int, default=48)
    cs.add_argument("--seed", type=int, default=0)
    cs.add_argument("--cols", type=int, default=8)

    args = parser.parse_args()
    if args.command == "download":
        download(args.dataset, args.sample_size, args.seed)
    elif args.command == "verify":
        verify(args.dataset)
    elif args.command == "characterize":
        characterize(args.dataset, args.skip_clip)
    elif args.command == "contact-sheet":
        contact_sheet(args.dataset, args.n, args.seed, cols=args.cols)


if __name__ == "__main__":
    main()

