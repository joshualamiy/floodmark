from ga511.geo import haversine_km


def test_haversine_zero_distance_same_point():
    assert haversine_km(33.8, -84.4, 33.8, -84.4) == 0.0


def test_haversine_known_distance_atlanta_to_marietta():
    # Downtown Atlanta (~33.7490, -84.3880) to Marietta (~33.9526, -84.5499)
    # is roughly 27 km as the crow flies.
    d = haversine_km(33.7490, -84.3880, 33.9526, -84.5499)
    assert 20 < d < 35


def test_haversine_one_degree_latitude_is_about_111_km():
    d = haversine_km(33.0, -84.4, 34.0, -84.4)
    assert 108 < d < 113


def test_haversine_symmetric():
    a = haversine_km(33.8, -84.4, 34.0, -84.3)
    b = haversine_km(34.0, -84.3, 33.8, -84.4)
    assert abs(a - b) < 1e-9
