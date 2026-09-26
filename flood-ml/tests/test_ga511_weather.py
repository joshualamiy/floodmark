from ga511.weather import classify_weak_label


def test_likely_wet_from_1h_threshold():
    assert classify_weak_label(0.2, None, None) == "likely_wet"
    assert classify_weak_label(0.19, None, 5.0) != "likely_wet"


def test_likely_wet_from_3h_threshold():
    assert classify_weak_label(0.0, 1.0, None) == "likely_wet"
    assert classify_weak_label(0.0, 0.99, 0.0) == "likely_dry"


def test_likely_dry_when_six_hours_clear():
    assert classify_weak_label(0.0, 0.0, 0.0) == "likely_dry"


def test_uncertain_when_no_data():
    assert classify_weak_label(None, None, None) == "uncertain"


def test_uncertain_when_6h_unknown_and_below_wet_thresholds():
    assert classify_weak_label(0.05, 0.3, None) == "uncertain"
