from itertools import pairwise

from prep.dedup import cluster_phashes, thin_sequence


def test_identical_hashes_cluster_together():
    phashes = ["aaaaaaaaaaaaaaaa"] * 3 + ["ffffffffffffffff"]
    ids = cluster_phashes(phashes)
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] != ids[0]


def test_hashes_within_hamming_threshold_cluster_together():
    a = "0000000000000000"
    b = "0000000000000001"
    ids = cluster_phashes([a, b])
    assert ids[0] == ids[1]


def test_hashes_far_apart_do_not_cluster():
    a = "0000000000000000"
    b = "ffffffffffffffff"
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
    items = [(i, "0000000000000000") for i in range(10)]
    kept = thin_sequence(items, min_distance=6, max_gap=15)
    assert len(kept) < len(items)
    assert kept[0] == 0


def test_thin_sequence_keeps_frames_that_differ_enough():
    a, b = "0000000000000000", "ffffffffffffffff"
    items = [(i, a if i % 2 == 0 else b) for i in range(6)]
    kept = thin_sequence(items, min_distance=6)
    assert kept == list(range(6))


def test_thin_sequence_resamples_after_max_gap_even_if_static():
    items = [(i, "0000000000000000") for i in range(40)]
    kept = thin_sequence(items, min_distance=6, max_gap=10)
    gaps = [b - a for a, b in pairwise(kept)]
    assert all(g <= 11 for g in gaps)
    assert len(kept) > 1


def test_thin_sequence_handles_missing_phashes_by_keeping_them():
    items = [(0, "0000000000000000"), (1, None), (2, "0000000000000000")]
    kept = thin_sequence(items)
    assert 1 in kept

