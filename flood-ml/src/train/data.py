# tf.data pipeline from the manifest (stage a/b)
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/processed/manifest.csv")
IMG_SIZE = 224
RESIZE_SHORT_SIDE = 256

STAGE_A = "a"
STAGE_B = "b"
STAGES = (STAGE_A, STAGE_B)

VARIANT_SPEC = "spec"
VARIANT_MIXED = "mixed"
VARIANTS = (VARIANT_SPEC, VARIANT_MIXED)

MODE_CROP = "crop"
MODE_SQUASH = "squash"
MODE_LETTERBOX = "letterbox"
MODES = (MODE_CROP, MODE_SQUASH, MODE_LETTERBOX)

CROP_PRECROP_RATIO = 256 / 224
LETTERBOX_PAD_VALUE = 128.0

WET_UPWEIGHT_DEFAULT = 8.0

_REQUIRED_COLUMNS = ("path", "label", "split")


def load_manifest(manifest_path: Path | str = MANIFEST_PATH) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest {manifest_path} is missing columns: {missing}")
    return df


def select_stage_rows(
    df: pd.DataFrame,
    stage: str,
    variant: str | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if split is not None:
        df = df[df["split"] == split]

    if stage == STAGE_A:
        # not_flooded could be dry or wet, so stage a can't use it
        rows = df[df["label"] != "not_flooded"].copy()
        rows["label_bin"] = (rows["label"] != "dry").astype("float32")
    else:
        if variant not in VARIANTS:
            raise ValueError(f"stage b needs variant in {VARIANTS}, got {variant!r}")
        if variant == VARIANT_SPEC:
            rows = df[df["label"].isin(["wet", "flooded"])].copy()
        else:
            rows = df.copy()
        rows["label_bin"] = (rows["label"] == "flooded").astype("float32")

    return rows.reset_index(drop=True)


def select_all_rows(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    rows = df.copy()
    if split is not None:
        rows = rows[rows["split"] == split]
    return rows.reset_index(drop=True)


def compute_class_weights(label_bin: np.ndarray) -> dict[int, float]:
    label_bin = np.asarray(label_bin).astype(int)
    classes, counts = np.unique(label_bin, return_counts=True)
    if len(classes) < 2:
        return {0: 1.0, 1: 1.0}
    n = len(label_bin)
    return {int(c): n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}


def compute_sample_weights(
    rows: pd.DataFrame,
    stage: str,
    variant: str | None = None,
    wet_upweight: float = WET_UPWEIGHT_DEFAULT,
) -> np.ndarray:
    class_w = compute_class_weights(rows["label_bin"].to_numpy())
    weights = rows["label_bin"].map(lambda v: class_w[int(v)]).astype("float32").to_numpy()
    if stage == STAGE_B and variant == VARIANT_MIXED:
        wet_mask = (rows["label"] == "wet").to_numpy()
        weights = weights.copy()
        weights[wet_mask] *= wet_upweight
    return weights


def _resize_short_side(image, short_side: int = RESIZE_SHORT_SIDE):
    import tensorflow as tf

    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    scale = tf.cast(short_side, tf.float32) / tf.minimum(h, w)
    new_h = tf.cast(tf.round(h * scale), tf.int32)
    new_w = tf.cast(tf.round(w * scale), tf.int32)
    return tf.image.resize(image, [new_h, new_w], method="bilinear")


def _squash_resize(image, size: int):
    import tensorflow as tf

    image = tf.cast(image, tf.float32)
    return tf.image.resize(image, [size, size], method="bilinear")


def _letterbox_resize(image, size: int, pad_value: float = LETTERBOX_PAD_VALUE):
    import tensorflow as tf

    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    scale = tf.cast(size, tf.float32) / tf.maximum(h, w)
    new_h = tf.maximum(1, tf.cast(tf.round(h * scale), tf.int32))
    new_w = tf.maximum(1, tf.cast(tf.round(w * scale), tf.int32))
    resized = tf.image.resize(image, [new_h, new_w], method="bilinear")
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    padded = tf.image.pad_to_bounding_box(resized - pad_value, top, left, size, size) + pad_value
    return tf.clip_by_value(padded, 0.0, 255.0)


def _load_and_prep(
    path, label, weight, raw_label, *, augment: bool, img_size: int, mode: str, repo_root: str,
    night_aug: bool = True,
):
    import tensorflow as tf

    from prep.augment import tf_camera_style

    full_path = tf.strings.join([repo_root, path], separator="/") if repo_root else path
    raw = tf.io.read_file(full_path)
    image = tf.io.decode_jpeg(raw, channels=3)

    if mode == MODE_CROP:
        precrop = max(img_size, round(img_size * CROP_PRECROP_RATIO))
        image = _resize_short_side(image, short_side=precrop)
        image = tf.cast(tf.clip_by_value(image, 0.0, 255.0), tf.uint8)
        if augment:
            image = tf_camera_style(image, raw_label, night_aug=night_aug)
            image = tf.image.random_crop(image, [img_size, img_size, 3])
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)
    else:
        image = _squash_resize(image, img_size) if mode == MODE_SQUASH else _letterbox_resize(image, img_size)
        image = tf.cast(tf.clip_by_value(image, 0.0, 255.0), tf.uint8)
        if augment:
            image = tf_camera_style(image, raw_label, night_aug=night_aug)
            image = tf.image.random_flip_left_right(image)

    image = tf.cast(image, tf.float32)
    image.set_shape([img_size, img_size, 3])
    return image, label, weight


def make_dataset(
    rows: pd.DataFrame,
    *,
    stage: str,
    variant: str | None = None,
    training: bool,
    batch_size: int = 32,
    img_size: int = IMG_SIZE,
    mode: str = MODE_CROP,
    shuffle_buffer: int = 4096,
    seed: int = 0,
    repo_root: str | None = None,
    wet_upweight: float = WET_UPWEIGHT_DEFAULT,
    num_parallel_calls=None,
    night_aug: bool = True,
):
    import tensorflow as tf

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if num_parallel_calls is None:
        num_parallel_calls = tf.data.AUTOTUNE

    if "sample_weight" in rows.columns:
        weights = rows["sample_weight"].astype("float32").to_numpy()
    else:
        weights = compute_sample_weights(rows, stage, variant, wet_upweight=wet_upweight)

    paths = rows["path"].astype(str).to_numpy()
    labels = rows["label_bin"].astype("float32").to_numpy()
    raw_labels = rows["label"].astype(str).to_numpy() if "label" in rows.columns else np.array([""] * len(rows))

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights.astype("float32"), raw_labels))
    if training:
        ds = ds.shuffle(min(shuffle_buffer, max(len(rows), 1)), seed=seed, reshuffle_each_iteration=True)

    root = repo_root or ""

    def _map(path, label, weight, raw_label):
        return _load_and_prep(
            path, label, weight, raw_label, augment=training, img_size=img_size, mode=mode, repo_root=root,
            night_aug=night_aug,
        )

    ds = ds.map(_map, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


@dataclass
class ThroughputResult:
    images_per_sec: float
    n_images: int
    seconds: float
    batch_size: int


def benchmark_throughput(ds, n_batches: int = 20, warmup_batches: int = 2) -> ThroughputResult:
    it = iter(ds)
    batch_size = None
    for _ in range(warmup_batches):
        batch = next(it, None)
        if batch is None:
            break
        batch_size = int(batch[0].shape[0])

    n_images = 0
    start = time.perf_counter()
    for _ in range(n_batches):
        batch = next(it, None)
        if batch is None:
            break
        n_images += int(batch[0].shape[0])
        batch_size = batch_size or int(batch[0].shape[0])
    elapsed = time.perf_counter() - start
    rate = n_images / elapsed if elapsed > 0 else float("nan")
    return ThroughputResult(images_per_sec=rate, n_images=n_images, seconds=elapsed, batch_size=batch_size or 0)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Benchmark the training tf.data pipeline's throughput.")
    parser.add_argument("--stage", choices=STAGES, default=STAGE_A)
    parser.add_argument("--variant", choices=VARIANTS, default=VARIANT_SPEC)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--mode", choices=MODES, default=MODE_CROP)
    parser.add_argument("--n-batches", type=int, default=30)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--log-file", default="logs/jobs/data_benchmark.log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )

    df = load_manifest(args.manifest)
    rows = select_stage_rows(df, args.stage, args.variant, split=args.split)
    ds = make_dataset(
        rows,
        stage=args.stage,
        variant=args.variant,
        training=(args.split == "train"),
        batch_size=args.batch_size,
        img_size=args.img_size,
        mode=args.mode,
    )
    result = benchmark_throughput(ds, n_batches=args.n_batches)
    payload = {
        "stage": args.stage,
        "variant": args.variant,
        "split": args.split,
        "mode": args.mode,
        "img_size": args.img_size,
        "n_rows": len(rows),
        **result.__dict__,
    }
    logger.info("throughput: %s", json.dumps(payload))
    print(json.dumps(payload, indent=2))

