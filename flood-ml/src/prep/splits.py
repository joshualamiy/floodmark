"""Group-aware split assignment.

Two independent mechanisms, per PLAN.md section 3 and the Phase 2 brief:

- `ga511_camera_splits`: 511GA cameras get a *persisted, stable* split from
  a salted hash of `camera_id`, so re-running the manifest build after new
  frames (or even new cameras) arrive never moves a camera that was already
  assigned. About 20% of cameras land in test; the rest split ~80/20
  train/val. A camera already known (at the time it is first assigned) to
  have a manually- or ai_review-confirmed wet/flooded frame is preferentially
  pulled into test, up to a cap, so the held-out set isn't all-dry by luck.

- `greedy_group_stratified_split`: external data groups (FRED locations,
  NYSDOT camera sites, still-image dedup clusters) are packed into
  train/val/test at ~70/15/15 with a greedy per-label-deficit heuristic,
  after any `forced` groups (Flood Master test videos, the held-out FRED
  test location) are pinned to their split.

Both are deterministic given the same inputs, and `assert_disjoint` checks
the final invariant: no `group_id`, `camera_id`, or `dup_cluster` spans two
splits.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from prep.common import stable_hash_fraction

GA511_SPLIT_SALT = "ga511-camera-split-v1"


def ga511_camera_splits(
    camera_ids: list[str],
    existing: dict[str, str] | None = None,
    reviewed_positive_cameras: set[str] | None = None,
    test_frac: float = 0.20,
    val_frac_of_nontest: float = 0.20,
    max_test_frac: float = 0.30,
) -> dict[str, str]:
    """Returns camera_id -> split for every camera in `camera_ids`. Cameras
    already present in `existing` keep their split unchanged. New cameras
    get a deterministic hash-based split, except that a new camera with a
    known wet/flooded review is pulled into test if test is still under
    `max_test_frac` of all cameras seen so far.
    """
    existing = dict(existing or {})
    reviewed_positive_cameras = reviewed_positive_cameras or set()

    all_ids = sorted(set(camera_ids) | set(existing.keys()))
    n_total = len(all_ids)
    n_test_existing = sum(1 for c in existing.values() if c == "test")

    result = dict(existing)
    for cam in sorted(camera_ids):
        if cam in result:
            continue
        test_draw = stable_hash_fraction(cam, GA511_SPLIT_SALT + ":test")
        base_split = "test" if test_draw < test_frac else None
        if base_split is None:
            val_draw = stable_hash_fraction(cam, GA511_SPLIT_SALT + ":val")
            base_split = "val" if val_draw < val_frac_of_nontest else "train"

        if base_split != "test" and cam in reviewed_positive_cameras:
            current_test_frac = n_test_existing / max(1, n_total)
            if current_test_frac < max_test_frac:
                base_split = "test"

        result[cam] = base_split
        if base_split == "test":
            n_test_existing += 1

    return result


def greedy_group_stratified_split(
    group_labels: dict[str, Counter],
    forced: dict[str, str] | None = None,
    target_fracs: dict[str, float] | None = None,
) -> dict[str, str]:
    """`group_labels`: group_id -> Counter(label -> count). `forced`:
    group_id -> split, pinned before the greedy pass (e.g. Flood Master test
    videos, the held-out FRED test location). Remaining groups are visited
    largest-first (deterministic tie-break by group_id) and each is assigned
    to whichever split has the largest current deficit against its target
    fraction, summed over the group's own label mix so common labels don't
    get starved in one split.
    """
    target_fracs = target_fracs or {"train": 0.70, "val": 0.15, "test": 0.15}
    forced = forced or {}
    splits = list(target_fracs.keys())

    assigned: dict[str, str] = {}
    running: dict[str, Counter] = {s: Counter() for s in splits}
    label_totals: Counter = Counter()

    for gid, split in forced.items():
        counts = group_labels.get(gid, Counter())
        assigned[gid] = split
        running[split].update(counts)
        label_totals.update(counts)

    remaining = [gid for gid in group_labels if gid not in assigned]
    remaining.sort(key=lambda g: (-sum(group_labels[g].values()), g))

    for gid in remaining:
        counts = group_labels[gid]
        label_totals.update(counts)
        best_split, best_score = None, None
        for s in splits:
            score = 0.0
            for label in counts:
                target = target_fracs[s] * label_totals[label]
                deficit = target - running[s][label]
                score += deficit
            if best_score is None or score > best_score:
                best_split, best_score = s, score
        assigned[gid] = best_split
        running[best_split].update(counts)

    return assigned


def assert_disjoint(rows: list[dict], keys: tuple[str, ...] = ("group_id", "camera_id", "dup_cluster")) -> list[str]:
    """Returns a list of human-readable violation messages (empty if none).
    Does not raise, so callers can log-and-fix or hard-fail as appropriate.
    """
    violations = []
    for key in keys:
        by_value: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            v = r.get(key)
            if v is None or v == "":
                continue
            split = r.get("split")
            if split:
                by_value[str(v)].add(split)
        for v, splitset in by_value.items():
            if len(splitset) > 1:
                violations.append(f"{key}={v!r} spans splits {sorted(splitset)}")
    return violations
