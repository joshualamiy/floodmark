# bbox + haversine
from __future__ import annotations

import math

ATLANTA_LAT_MIN, ATLANTA_LAT_MAX = 33.5, 34.1
ATLANTA_LON_MIN, ATLANTA_LON_MAX = -84.7, -84.1

EARTH_RADIUS_KM = 6371.0088


def in_atlanta_bbox(lat: float, lon: float) -> bool:
    return ATLANTA_LAT_MIN <= lat <= ATLANTA_LAT_MAX and ATLANTA_LON_MIN <= lon <= ATLANTA_LON_MAX


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c

