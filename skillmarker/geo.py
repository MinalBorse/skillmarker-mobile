"""Approximate coordinates of German cities, for sources that give a city name but no coordinates."""
import re

CITY_COORDS = {
    "berlin": (52.520, 13.405), "hamburg": (53.551, 9.993), "münchen": (48.137, 11.575), "munich": (48.137, 11.575),
    "köln": (50.938, 6.960), "cologne": (50.938, 6.960), "frankfurt": (50.110, 8.682), "stuttgart": (48.776, 9.177),
    "düsseldorf": (51.227, 6.773), "dusseldorf": (51.227, 6.773), "leipzig": (51.340, 12.375), "dortmund": (51.514, 7.468),
    "essen": (51.456, 7.012), "bremen": (53.079, 8.802), "dresden": (51.050, 13.738), "hannover": (52.375, 9.732),
    "hanover": (52.375, 9.732), "nürnberg": (49.452, 11.077), "nuremberg": (49.452, 11.077), "duisburg": (51.434, 6.762),
    "bochum": (51.482, 7.216), "wuppertal": (51.256, 7.150), "bielefeld": (52.030, 8.532), "bonn": (50.737, 7.098),
    "münster": (51.961, 7.626), "karlsruhe": (49.007, 8.404), "mannheim": (49.488, 8.466), "augsburg": (48.371, 10.898),
    "wiesbaden": (50.078, 8.240), "mainz": (49.993, 8.247), "heidelberg": (49.399, 8.672), "darmstadt": (49.873, 8.651),
    "freiburg": (47.999, 7.842), "potsdam": (52.391, 13.064), "regensburg": (49.013, 12.102), "ulm": (48.401, 9.987),
    "aachen": (50.776, 6.084), "kiel": (54.323, 10.123), "lübeck": (53.866, 10.686), "rostock": (54.092, 12.099),
    "erlangen": (49.590, 11.004), "ingolstadt": (48.766, 11.425), "wolfsburg": (52.423, 10.787), "jena": (50.927, 11.589),
    "walldorf": (49.306, 8.642), "neckarsulm": (49.199, 9.238), "braunschweig": (52.269, 10.521), "kassel": (51.312, 9.480),
}
_PAT = re.compile("|".join(sorted(map(re.escape, CITY_COORDS), key=len, reverse=True)), re.IGNORECASE)


def locate(text: str) -> tuple[str | None, float | None, float | None]:
    """(city, lat, lon) of the first known German city mentioned in a location string."""
    m = _PAT.search(text or "")
    if not m:
        return None, None, None
    key = m.group(0).lower()
    lat, lon = CITY_COORDS[key]
    return m.group(0).title() if key.isascii() else m.group(0), lat, lon
