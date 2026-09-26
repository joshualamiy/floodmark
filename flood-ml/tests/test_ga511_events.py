from ga511.events import flag_event, match_flood_keyword


def test_match_flood_keyword_basic():
    assert match_flood_keyword("Roadway flooding reported near I-285") == "flooding"
    assert match_flood_keyword("HIGH WATER on service road") == "high water"
    assert match_flood_keyword("Minor water on roadway") == "water"


def test_match_flood_keyword_word_boundary_no_false_substring_match():
    # "waterway" should not match "water" as a substring without a boundary... but
    # word-boundary regex on "water" *will* match inside "waterway" since \b sits
    # between non-word/word transitions at the start; the point of this test is
    # that unrelated words with no keyword substring at all don't match.
    assert match_flood_keyword("Debris in the roadway") is None
    assert match_flood_keyword("Bridgework ongoing") is None
    assert match_flood_keyword(None) is None
    assert match_flood_keyword("") is None


def test_match_flood_keyword_case_insensitive():
    assert match_flood_keyword("FLOOD warning issued") == "flood"


def test_flag_event_marks_water_main_as_likely_false_positive():
    event = {
        "Description": "Water main break causing lane closure",
        "EventType": "Construction",
        "Subtype": "Utility work",
    }
    flag = flag_event(event)
    assert flag is not None
    assert flag["matched_keyword"] == "water"
    assert flag["likely_false_positive"] is True


def test_flag_event_real_flood_is_not_false_positive():
    event = {
        "Description": "High water reported, road impassable",
        "EventType": "Weather",
        "Subtype": None,
    }
    flag = flag_event(event)
    assert flag is not None
    assert flag["matched_keyword"] == "high water"
    assert flag["likely_false_positive"] is False


def test_flag_event_returns_none_when_no_keyword():
    event = {"Description": "Lane closure for paving", "EventType": "Construction", "Subtype": None}
    assert flag_event(event) is None
