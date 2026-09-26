# download + verify the public datasets
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FLOOD_ML_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = FLOOD_ML_ROOT / "data" / "raw"
LOG_DIR = FLOOD_ML_ROOT / "logs" / "jobs"

LARGE_DATASET_GB = 5.0

NYSDOT_ZENODO_RECORD = "8370665"
NYSDOT_ZIP_NAME = "NYSDOT_quantitative_content_analysis.zip"
NYSDOT_ZIP_URL = (
    f"https://zenodo.org/records/{NYSDOT_ZENODO_RECORD}/files/{NYSDOT_ZIP_NAME}?download=1"
)
NYSDOT_ZIP_MD5 = "fc7924e225393f732cc9b2545a034ebe"
NYSDOT_LABEL_CODES = {
    1: "snow_severe",
    2: "snow",
    3: "wet",
    4: "dry",
    5: "poor_vis",
    6: "obstructed",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _log_path(name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{name}.log"


def _nonempty_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def download_roadway_flooding() -> None:
    dest = DATA_RAW / "roadway_flooding"
    if _nonempty_dir(dest):
        print("roadway_flooding: already present, skipping")
        return
    dest.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            "saurabhshahane/roadway-flooding-image-dataset",
            "-p",
            str(dest),
            "--unzip",
        ]
    )


def download_flood_area_segmentation() -> None:
    dest = DATA_RAW / "flood_area_segmentation"
    if _nonempty_dir(dest):
        print("flood_area_segmentation: already present, skipping")
        return
    dest.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            "faizalkarim/flood-area-segmentation",
            "-p",
            str(dest),
            "--unzip",
        ]
    )


def download_water_segmentation(allow_large: bool) -> None:
    dest = DATA_RAW / "water_segmentation"
    if _nonempty_dir(dest):
        print("water_segmentation: already present, skipping")
        return
    if not allow_large:
        print(
            "water_segmentation is ~5 GB (over the 5 GB rule); rerun with "
            "--allow-large to fetch it from Kaggle."
        )
        return
    dest.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            "gvclsu/water-segmentation-dataset",
            "-p",
            str(dest),
            "--unzip",
        ]
    )


def download_fred(allow_large: bool) -> None:
    dest = DATA_RAW / "fred"
    if _nonempty_dir(dest):
        print("fred: already present, skipping")
        return
    if not allow_large:
        print(
            "fred (front camera images + labels) is ~11 GB (over the 5 GB "
            "rule); rerun with --allow-large to fetch it from Hugging Face."
        )
        return
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        "README.md",
        "dry/KITTI-style/*/front-imgs/*",
        "dry/KITTI-style/*/front-labels/*",
        "dry/KITTI-style/*/imu/*",
        "dry/KITTI-style/*/utm/*",
        "dry/KITTI-style/*/ground_plane_eqn.txt",
        "flooded/KITTI-style/*/front-imgs/*",
        "flooded/KITTI-style/*/front-labels/*",
        "flooded/KITTI-style/*/imu/*",
        "flooded/KITTI-style/*/utm/*",
        "flooded/KITTI-style/*/ground_plane_eqn.txt",
    ]
    snapshot_download(
        repo_id="CMalone-Jupiter/FRED",
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow_patterns,
    )


def download_tinycamml() -> None:
    dest = DATA_RAW / "tinycamml"
    if _nonempty_dir(dest):
        print("tinycamml: already present, skipping")
        return
    _run(["git", "clone", "https://github.com/TinyCamML/TinyCamML", str(dest)])


def _extract_nysdot_labels(extracted_root: Path) -> int:
    import openpyxl

    xlsx_path = extracted_root / "NYSDOT_ICR_Example.xlsx"
    images_root = extracted_root / "Images"
    trial_dirs = {
        f"Trial{i}": images_root / f"Trial{i}" for i in range(1, 5)
    }
    files_by_trial = {
        trial: {p.name for p in d.iterdir()} if d.is_dir() else set()
        for trial, d in trial_dirs.items()
    }

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    out_rows = []
    for trial_num in range(1, 5):
        trial = f"Trial{trial_num}"
        ws = wb[f"trial{trial_num}_labels"]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] is None or row[1] is None:
                continue
            img_name = row[1]
            votes = [v for v in row[2:8] if isinstance(v, int)]
            if not votes:
                continue
            counts = Counter(votes)
            maj_code, maj_n = counts.most_common(1)[0]
            normalized = img_name.replace(":", "_")
            if img_name in files_by_trial[trial]:
                resolved = img_name
            elif normalized in files_by_trial[trial]:
                resolved = normalized
            else:
                resolved = ""
            out_rows.append(
                {
                    "trial": trial,
                    "image_filename": img_name,
                    "resolved_filename": resolved,
                    "rel_path": (
                        f"NYSDOT_quantitative_content_analysis/Images/{trial}/{resolved}"
                        if resolved
                        else ""
                    ),
                    "has_image": "1" if resolved else "0",
                    "majority_label": NYSDOT_LABEL_CODES.get(maj_code, str(maj_code)),
                    "n_coders": len(votes),
                    "n_agree": maj_n,
                    "agreement_frac": round(maj_n / len(votes), 3),
                    "votes_raw": ";".join(str(v) for v in votes),
                }
            )

    labels_path = extracted_root.parent / "labels.csv"
    fieldnames = [
        "trial",
        "image_filename",
        "resolved_filename",
        "rel_path",
        "has_image",
        "majority_label",
        "n_coders",
        "n_agree",
        "agreement_frac",
        "votes_raw",
    ]
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return len(out_rows)


def download_nysdot_road_surface() -> None:
    import zipfile

    dest = DATA_RAW / "nysdot_road_surface"
    if (dest / "labels.csv").exists():
        print("nysdot_road_surface: already present, skipping")
        return
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = dest / NYSDOT_ZIP_NAME
    if not zip_path.exists():
        print(f"downloading {NYSDOT_ZIP_URL}")
        urllib.request.urlretrieve(NYSDOT_ZIP_URL, zip_path)

    digest = hashlib.md5(zip_path.read_bytes()).hexdigest()
    if digest != NYSDOT_ZIP_MD5:
        raise RuntimeError(
            f"nysdot_road_surface: md5 mismatch (got {digest}, expected {NYSDOT_ZIP_MD5})"
        )

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX") or "/._" in name:
                continue
            zf.extract(name, dest)

    extracted_root = dest / "NYSDOT_quantitative_content_analysis"
    n_rows = _extract_nysdot_labels(extracted_root)
    zip_path.unlink()
    print(f"nysdot_road_surface: extracted, wrote labels.csv with {n_rows} rows")


DOWNLOADERS = {
    "roadway_flooding": lambda allow_large: download_roadway_flooding(),
    "flood_area_segmentation": lambda allow_large: download_flood_area_segmentation(),
    "water_segmentation": download_water_segmentation,
    "fred": download_fred,
    "tinycamml": lambda allow_large: download_tinycamml(),
    "nysdot_road_surface": lambda allow_large: download_nysdot_road_surface(),
}


def download(dataset: str, allow_large: bool) -> None:
    names = list(DOWNLOADERS) if dataset == "all" else [dataset]
    for name in names:
        DOWNLOADERS[name](allow_large)


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


def _glob_many(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(root.glob(pat))
    return sorted({p for p in out if p.is_file()})


DATASET_GLOBS: dict[str, dict[str, list[str]]] = {
    "roadway_flooding": {
        "images": ["Dataset/images/*"],
        "masks": ["Dataset/labels/*"],
    },
    "flood_area_segmentation": {
        "images": ["Image/*"],
        "masks": ["Mask/*"],
    },
    "water_segmentation": {
        "images": [
            "water_v1/water_v1/JPEGImages/*/*",
            "water_v2/water_v2/JPEGImages/*/*",
        ],
        "masks": [
            "water_v1/water_v1/Annotations/*/*",
            "water_v2/water_v2/Annotations/*/*",
        ],
    },
    "fred": {
        "images": [
            "dry/KITTI-style/*/front-imgs/*",
            "flooded/KITTI-style/*/front-imgs/*",
        ],
        "masks": ["flooded/KITTI-style/*/front-labels/*"],
    },
    "nysdot_road_surface": {
        "images": ["NYSDOT_quantitative_content_analysis/Images/*/*"],
        "masks": [],
    },
}


def verify() -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }
    corrupt_lines: list[str] = []

    for name, globs in DATASET_GLOBS.items():
        root = DATA_RAW / name
        if not root.is_dir():
            report["datasets"][name] = {"present": False}
            continue
        entry: dict[str, Any] = {"present": True}
        for kind, patterns in globs.items():
            if not patterns:
                continue
            paths = [p for p in _glob_many(root, patterns) if p.suffix.lower() in IMAGE_EXTS]
            result = _check_images(paths)
            for c in result["corrupt"]:
                corrupt_lines.append(f"{name}/{kind}: {c}")
            entry[kind] = {
                "count": result["count"],
                "bytes": result["bytes"],
                "n_corrupt": len(result["corrupt"]),
            }
        report["datasets"][name] = entry

    tcm = DATA_RAW / "tinycamml"
    if tcm.is_dir():
        files = [p for p in tcm.rglob("*") if p.is_file() and ".git" not in p.parts]
        report["datasets"]["tinycamml"] = {
            "present": True,
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "note": "no labeled roadway image dataset in this repo",
        }
    else:
        report["datasets"]["tinycamml"] = {"present": False}

    inventory_path = DATA_RAW / "inventory.json"
    inventory_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    log_path = _log_path("public_datasets_verify")
    with open(log_path, "w") as f:
        f.write(f"public_datasets --verify run at {report['generated_at']}\n\n")
        for name, entry in report["datasets"].items():
            f.write(f"{name}: {json.dumps(entry)}\n")
        f.write(f"\ncorrupt files ({len(corrupt_lines)}):\n")
        f.writelines(line + "\n" for line in corrupt_lines)

    total_corrupt = len(corrupt_lines)
    print(f"wrote {inventory_path}")
    print(f"wrote {log_path} ({total_corrupt} corrupt file(s) found)")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download any dataset not already on disk")
    dl.add_argument(
        "--dataset",
        choices=["all", *DOWNLOADERS.keys()],
        default="all",
    )
    dl.add_argument(
        "--allow-large",
        action="store_true",
        help="allow a fresh (non-idempotent) download of a dataset over 5 GB "
        "(water_segmentation, fred)",
    )

    sub.add_parser("verify", help="check what's on disk, write data/raw/inventory.json")

    args = parser.parse_args()
    if args.command == "download":
        download(args.dataset, args.allow_large)
    elif args.command == "verify":
        verify()


if __name__ == "__main__":
    main()

