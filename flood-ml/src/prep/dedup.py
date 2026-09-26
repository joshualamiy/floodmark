# phash dedup + thinning long video runs
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

