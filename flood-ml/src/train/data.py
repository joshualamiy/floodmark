"""tf.data input pipeline for Stage A/B training, built from
`data/processed/manifest.csv`.

Stage / variant row selection (PLAN.md section 3, overridden for Stage B by
the Phase 3 brief -- wet-but-not-flooded has only 15 train / 6 val rows, so
we train and compare two Stage B variants rather than one):

- Stage A: every row. label_bin = 0 if label == "dry" else 1.
- Stage B "spec" (PLAN.md as written): wet + flooded rows only.
  label_bin = 1 if label == "flooded" else 0 (i.e. wet == 0).
- Stage B "mixed": every row. label_bin = 1 if label == "flooded" else 0, so
  "not flooded" = wet + dry. Wet rows get an extra sample-weight multiplier
  (`wet_upweight`) so the rare wet class still pulls its weight against the
  much larger dry negative pool.

Images on disk already have short side ~256 (a few sources are a little
under that, e.g. cropped NYSDOT frames), so the pipeline re-resizes the
short side to `RESIZE_SHORT_SIDE` before cropping, rather than assuming it.

Training augmentation: `prep.augment.tf_camera_style` (domain randomization)
then a random 224 crop and a horizontal flip (no vertical flip, since roads
never appear upside down). Validation: resize + center crop only.
"""
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
    """Filters `df` (already loaded from the manifest) to the rows a given
    stage/variant trains or evaluates on, restricted to `split` if given, and
    attaches a float32 `label_bin` column. Never call with split="test": the
    caller is responsible for that rule (see PLAN.md rule 4 / the Phase 3
    brief); this function itself has no special-casing that would stop it.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if split is not None:
        df = df[df["split"] == split]

    if stage == STAGE_A:
        rows = df.copy()
        rows["label_bin"] = (rows["label"] != "dry").astype("float32")
    else:
        if variant not in VARIANTS:
            raise ValueError(f"stage b needs variant in {VARIANTS}, got {variant!r}")
        if variant == VARIANT_SPEC:
            rows = df[df["label"].isin(["wet", "flooded"])].copy()
        else:  # mixed
            rows = df.copy()
        rows["label_bin"] = (rows["label"] == "flooded").astype("float32")

    return rows.reset_index(drop=True)


def compute_class_weights(label_bin: np.ndarray) -> dict[int, float]:
    """Balanced class weights: n_samples / (n_classes * n_class_i). Falls
    back to {0: 1.0, 1: 1.0} if only one class is present (degenerate, but
    keeps callers from crashing on a tiny synthetic test set).
    """
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
    """Per-row sample weight = balanced class weight, times `wet_upweight`
    for wet rows when training Stage B's "mixed" variant (where wet rows are
    a small slice of the negative class and would otherwise be drowned out
    by dry negatives).
    """
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


def _load_and_prep(path, label, weight, *, augment: bool, img_size: int, repo_root: str):
    import tensorflow as tf

    from prep.augment import tf_camera_style

    full_path = tf.strings.join([repo_root, path], separator="/") if repo_root else path
    raw = tf.io.read_file(full_path)
    image = tf.io.decode_jpeg(raw, channels=3)
    image = _resize_short_side(image)
    image = tf.cast(tf.clip_by_value(image, 0.0, 255.0), tf.uint8)
    if augment:
        image = tf_camera_style(image)
        image = tf.image.random_crop(image, [img_size, img_size, 3])
        image = tf.image.random_flip_left_right(image)
    else:
        image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)
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
    shuffle_buffer: int = 4096,
    seed: int = 0,
    repo_root: str | None = None,
    wet_upweight: float = WET_UPWEIGHT_DEFAULT,
    num_parallel_calls=None,
):
    """Builds a batched `tf.data.Dataset` of (image[0..255] float32, label,
    sample_weight) from an already-selected rows DataFrame (see
    `select_stage_rows`). `repo_root` lets callers run from any cwd; paths in
    the manifest are relative to flood-ml/, so pass that directory's
    absolute path when cwd isn't already flood-ml/.
    """
    import tensorflow as tf

    if num_parallel_calls is None:
        num_parallel_calls = tf.data.AUTOTUNE

    if "sample_weight" in rows.columns:
        weights = rows["sample_weight"].astype("float32").to_numpy()
    else:
        weights = compute_sample_weights(rows, stage, variant, wet_upweight=wet_upweight)

    paths = rows["path"].astype(str).to_numpy()
    labels = rows["label_bin"].astype("float32").to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights.astype("float32")))
    if training:
        ds = ds.shuffle(min(shuffle_buffer, max(len(rows), 1)), seed=seed, reshuffle_each_iteration=True)

    root = repo_root or ""

    def _map(path, label, weight):
        return _load_and_prep(path, label, weight, augment=training, img_size=img_size, repo_root=root)

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
    """Iterates `n_batches` batches (after `warmup_batches` to prime the
    tf.data pipeline / thread pool) and reports images/sec, so we can confirm
    the input pipeline isn't the bottleneck for the reported CPU fine-tune
    speeds (~280 img/s MobileNetV3Small, ~73 img/s EfficientNetB0).
    """
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
    )
    result = benchmark_throughput(ds, n_batches=args.n_batches)
    payload = {
        "stage": args.stage,
        "variant": args.variant,
        "split": args.split,
        "n_rows": len(rows),
        **result.__dict__,
    }
    logger.info("throughput: %s", json.dumps(payload))
    print(json.dumps(payload, indent=2))
