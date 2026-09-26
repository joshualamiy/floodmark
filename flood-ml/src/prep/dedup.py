"""Perceptual-hash dedup across all sources, and temporal thinning of long
video-like runs (FRED sequences, Flood Master test videos).

Uses `imagehash.phash` (via a precomputed hex string per row) and a simple
LSH-style bucketed search: the 64-bit hash is split into 4x16-bit bands, and
only rows sharing at least one band are compared exactly (Hamming <= 6),
which is much cheaper than all-pairs for thousands of images while still
finding every true near-duplicate pair (any two hashes within Hamming
distance 6 of a 64-bit hash must match exactly on at least one 16-bit band,
by pigeonhole: 4 bands * 6 max differing bits could in principle all land in
different bands only if bits are adversarially spread, so this is a
heuristic search, not an exact guarantee -- documented in the phase report).
"""
from __future__ import annotations

from collections import defaultdict

from prep.common import hamming, phash_hex_to_int

HAMMING_THRESHOLD = 6
N_BANDS = 4
BAND_BITS = 16


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _bands(value: int) -> list[int]:
    return [(value >> (BAND_BITS * i)) & 0xFFFF for i in range(N_BANDS)]


def cluster_phashes(phashes: list[str | None], threshold: int = HAMMING_THRESHOLD) -> list[int]:
    """Returns a cluster id (int, 0-based, stable given input order) for
    every row. Rows with no phash (None) each get their own singleton
    cluster. Uses bucketed union-find so it scales to thousands of rows.
    """
    n = len(phashes)
    uf = UnionFind(n)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    ints: list[int | None] = [phash_hex_to_int(p) if p else None for p in phashes]

    for i, val in enumerate(ints):
        if val is None:
            continue
        for band_idx, band_val in enumerate(_bands(val)):
            buckets[(band_idx, band_val)].append(i)

    for members in buckets.values():
        for a in range(len(members)):
            ia = members[a]
            for b in range(a + 1, len(members)):
                ib = members[b]
                if uf.find(ia) == uf.find(ib):
                    continue
                if hamming(ints[ia], ints[ib]) <= threshold:
                    uf.union(ia, ib)

    # Renumber roots to small dense ids in a stable order.
    root_to_id: dict[int, int] = {}
    ids: list[int] = []
    next_id = 0
    for i in range(n):
        if ints[i] is None:
            ids.append(next_id)
            next_id += 1
            continue
        root = uf.find(i)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        ids.append(root_to_id[root])
    return ids


def thin_sequence(items: list[tuple[int, str | None]], min_distance: int = 6,
                   max_gap: int = 15) -> list[int]:
    """Thin a single ordered video-like sequence of (row_index, phash_hex).
    Keeps a frame if it differs from the last *kept* frame by more than
    `min_distance`, OR if `max_gap` frames have been dropped in a row (so a
    long static stretch is still sampled every `max_gap` frames instead of
    collapsing to a single frame). Always keeps the first frame. Returns the
    list of kept row indices, in the given order.
    """
    kept: list[int] = []
    last_kept_hash: int | None = None
    dropped_since_last = 0
    for idx, phash_hex in items:
        val = phash_hex_to_int(phash_hex) if phash_hex else None
        if last_kept_hash is None or val is None:
            kept.append(idx)
            last_kept_hash = val
            dropped_since_last = 0
            continue
        if hamming(val, last_kept_hash) > min_distance or dropped_since_last >= max_gap:
            kept.append(idx)
            last_kept_hash = val
            dropped_since_last = 0
        else:
            dropped_since_last += 1
    return kept
