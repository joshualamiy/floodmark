"""Unit tests for prep.dedup: phash clustering (union-find over banded
buckets) and sequence thinning. No network, data, or models -- all synthetic
hex phashes.
"""
from itertools import pairwise

from prep.dedup import cluster_phashes, thin_sequence


def test_identical_hashes_cluster_together():
    phashes = ["aaaaaaaaaaaaaaaa"] * 3 + ["ffffffffffffffff"]
    ids = cluster_phashes(phashes)
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] != ids[0]


def test_hashes_within_hamming_threshold_cluster_together():
    # Differ by exactly 1 bit -> well within Hamming <= 6.
    a = "0000000000000000"
    b = "0000000000000001"
    ids = cluster_phashes([a, b])
    assert ids[0] == ids[1]


def test_hashes_far_apart_do_not_cluster():
    a = "0000000000000000"
    b = "ffffffffffffffff"  # all 64 bits differ
    ids = cluster_phashes([a, b])
    assert ids[0] != ids[1]


def test_none_phashes_get_singleton_clusters():
    ids = cluster_phashes([None, None, "abcdabcdabcdabcd"])
    assert ids[0] != ids[1]
    assert ids[0] != ids[2]
    assert ids[1] != ids[2]


def test_cluster_ids_are_stable_given_input_order():
    phashes = ["1111111111111111", "1111111111111111", "2222222222222222"]
    ids1 = cluster_phashes(phashes)
    ids2 = cluster_phashes(phashes)
    assert ids1 == ids2


def test_transitive_clustering_via_bucket_chain():
    # a-b differ by 1 bit, b-c differ by 1 bit (different bit), a-c differ by
    # 2 bits -- all still within threshold, and union-find should chain them.
    a = 0x0000000000000000
    b = 0x0000000000000001
    c = 0x0000000000000003
    ids = cluster_phashes([f"{a:016x}", f"{b:016x}", f"{c:016x}"])
    assert ids[0] == ids[1] == ids[2]


def test_thin_sequence_keeps_first_frame():
    items = [(0, "0000000000000000"), (1, "0000000000000000")]
    kept = thin_sequence(items)
    assert kept[0] == 0


def test_thin_sequence_drops_near_identical_consecutive_frames():
    # All identical: only the first frame (and periodic max_gap keepalives)
    # should be kept, not every single one.
    items = [(i, "0000000000000000") for i in range(10)]
    kept = thin_sequence(items, min_distance=6, max_gap=15)
    assert len(kept) < len(items)
    assert kept[0] == 0


def test_thin_sequence_keeps_frames_that_differ_enough():
    # Alternate between two very different hashes -> both should be kept
    # every time since each differs from the last *kept* frame.
    a, b = "0000000000000000", "ffffffffffffffff"
    items = [(i, a if i % 2 == 0 else b) for i in range(6)]
    kept = thin_sequence(items, min_distance=6)
    assert kept == list(range(6))


def test_thin_sequence_resamples_after_max_gap_even_if_static():
    items = [(i, "0000000000000000") for i in range(40)]
    kept = thin_sequence(items, min_distance=6, max_gap=10)
    # Frame 0 always kept; then no more than max_gap frames may be skipped
    # between two kept frames.
    gaps = [b - a for a, b in pairwise(kept)]
    # dropped_since_last must reach max_gap before the *next* frame is kept,
    # so consecutive kept frames are at most max_gap + 1 apart.
    assert all(g <= 11 for g in gaps)
    assert len(kept) > 1


def test_thin_sequence_handles_missing_phashes_by_keeping_them():
    items = [(0, "0000000000000000"), (1, None), (2, "0000000000000000")]
    kept = thin_sequence(items)
    assert 1 in kept  # a row with no phash is always kept (can't compare)
