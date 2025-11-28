from turfpy.measurement import distance
from geojson import Point

def compute_distance(lat1, lon1, lat2, lon2):
    return distance(Point((lon1, lat1)), Point((lon2, lat2)))
