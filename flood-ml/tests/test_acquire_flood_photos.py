"""Pure-logic tests for acquire.flood_photos -- no network, no data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acquire.flood_photos import (
    deterministic_sample,
    floodimg_local_name,
    parse_eu_flood_labels,
)


def test_deterministic_sample_reproducible():
    names = [f"img_{i}" for i in range(100)]
    a = deterministic_sample(names, 10, seed=1)
    b = deterministic_sample(names, 10, seed=1)
    assert a == b
    assert len(a) == 10
    assert a == sorted(a)


def test_deterministic_sample_different_seed_differs():
    names = [f"img_{i}" for i in range(100)]
    a = deterministic_sample(names, 10, seed=1)
    c = deterministic_sample(names, 10, seed=2)
    assert a != c


def test_deterministic_sample_caps_at_available():
    names = ["a", "b", "c"]
    assert deterministic_sample(names, 10, seed=0) == ["a", "b", "c"]


def test_floodimg_local_name_strips_folder():
    assert floodimg_local_name("Flood Images/Flood_1000.jpg") == "Flood_1000.jpg"
    assert floodimg_local_name("Annotation/flood_10.json") == "flood_10.json"


def test_parse_eu_flood_labels_flooded_and_not():
    labels = parse_eu_flood_labels(
        all_ids=["1", "2", "3"], flooding_ids={"1"}, irrelevant_ids={"2"}
    )
    assert labels == {"1": "flooded", "2": "not_flooded", "3": "unknown"}


def test_parse_eu_flood_labels_irrelevant_wins_over_flooding():
    # shouldn't happen upstream, but irrelevant.txt should still take priority
    labels = parse_eu_flood_labels(all_ids=["1"], flooding_ids={"1"}, irrelevant_ids={"1"})
    assert labels["1"] == "not_flooded"
