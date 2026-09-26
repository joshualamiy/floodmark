from prep.common import ga511_is_daytime, ga511_local_time

# 2026-09-26 12:00:00 America/New_York
NOON_PATH = "data/ga511/frames/1/1790438400.jpg"
# 2026-09-26 02:00:00 America/New_York
NIGHT_PATH = "data/ga511/frames/1/1790402400.jpg"


def test_ga511_local_time_parses_epoch_from_orig_path():
    dt = ga511_local_time(NOON_PATH)
    assert (dt.year, dt.month, dt.day, dt.hour) == (2026, 9, 26, 12)


def test_ga511_local_time_returns_none_for_bad_path():
    assert ga511_local_time("data/ga511/frames/1/not_a_timestamp.jpg") is None


def test_ga511_is_daytime_true_at_noon():
    assert ga511_is_daytime(NOON_PATH) is True


def test_ga511_is_daytime_false_at_night():
    assert ga511_is_daytime(NIGHT_PATH) is False


def test_ga511_is_daytime_none_for_bad_path():
    assert ga511_is_daytime("garbage") is None
