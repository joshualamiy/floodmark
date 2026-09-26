from ga511.cameras import build_atlanta_rows, parse_county
from ga511.geo import in_atlanta_bbox


def test_parse_county_from_trailing_parens():
    assert parse_county("I-75 NB at Spring St (Fulton)") == "Fulton"
    assert parse_county("SR 400 NB at Glenridge Dr (Fulton)") == "Fulton"
    assert parse_county("US 78 EB at Stone Mountain Fwy (DeKalb)") == "DeKalb"


def test_parse_county_missing_parens_returns_none():
    assert parse_county("I-75 NB at Spring St") is None
    assert parse_county("") is None
    assert parse_county(None) is None


def test_in_atlanta_bbox():
    assert in_atlanta_bbox(33.8, -84.4)
    assert not in_atlanta_bbox(31.0, -84.4)
    assert not in_atlanta_bbox(33.8, -83.0)
    assert in_atlanta_bbox(33.5, -84.7)
    assert in_atlanta_bbox(34.1, -84.1)


def _camera(cam_id, lat, lon, location, views):
    return {
        "Id": cam_id,
        "Roadway": "I-75",
        "Direction": "NB",
        "Latitude": lat,
        "Longitude": lon,
        "Location": location,
        "Name": f"cam-{cam_id}",
        "Views": views,
    }


def test_build_atlanta_rows_filters_bbox_and_disabled_views():
    cameras = [
        _camera(
            "1",
            33.8,
            -84.4,
            "I-75 NB at Spring St (Fulton)",
            [
                {"Id": "v1", "Url": "https://511ga.org/map/Cctv/v1", "Status": "Enabled"},
                {"Id": "v2", "Url": "https://511ga.org/map/Cctv/v2", "Status": "Disabled"},
            ],
        ),
        _camera(
            "2",
            30.0,
            -84.4,
            "I-95 NB at Somewhere (Chatham)",
            [{"Id": "v3", "Url": "https://511ga.org/map/Cctv/v3", "Status": "Enabled"}],
        ),
    ]
    rows, summary = build_atlanta_rows(cameras)

    assert summary.total_cameras == 2
    assert summary.total_views == 3
    assert summary.bbox_cameras == 1
    assert summary.bbox_views == 2
    assert summary.bbox_enabled_views == 1

    assert len(rows) == 1
    row = rows[0]
    assert row["view_id"] == "v1"
    assert row["camera_id"] == "1"
    assert row["county"] == "Fulton"
    assert row["view_status"] == "Enabled"

