import geojson
from shapely.geometry import Point, LineString, mapping
import pyproj
from shapely.ops import transform
import math
from functools import partial

# Function to create a circle boundary as a closed LineString in meters (UTM: EPSG:32632)
def create_circle_linestring_utm(lat, lon, LAC_name, h, radius=600, num_points=35):
    # Define the projection from geographic (EPSG:4326) to UTM Zone 32N (EPSG:32632)
    project_to_utm = partial(pyproj.transform,
                             pyproj.Proj(init='epsg:4326'),  # Source CRS: WGS84
                             pyproj.Proj(init='epsg:32632'))  # Target CRS: UTM Zone 32N

    project_to_wgs84 = partial(pyproj.transform,
                               pyproj.Proj(init='epsg:32632'),  # Source CRS: UTM Zone 32N
                               pyproj.Proj(init='epsg:4326'))   # Target CRS: WGS84

    # Project the center point to UTM Zone 32N
    point = Point(lon, lat)
    point_utm = transform(project_to_utm, point)

    # Generate points around the circle in UTM coordinates (meters)
    angles = [2 * math.pi * i / num_points for i in range(num_points)]
    circle_points_utm = [
        (
            point_utm.x + radius * math.cos(angle),  # X coordinate in meters
            point_utm.y + radius * math.sin(angle)   # Y coordinate in meters
        )
        for angle in angles
    ]

    # Ensure the first and last point are the same to close the circle
    circle_points_utm.append(circle_points_utm[0])

    # Create a LineString from the points in UTM Zone 32N
    line_utm = LineString(circle_points_utm)

    # Project the LineString back to geographic coordinates (EPSG:4326)
    line_wgs84 = transform(project_to_wgs84, line_utm)

    # Convert height (h) from meters to feet
    h_in_feet = h * 3.28084

    # Create the name by concatenating eTOD_name with h_in_feet
    name = f"{LAC_name} - Height: {h_in_feet:.1f} ft"

    return geojson.Feature(geometry=mapping(line_wgs84), properties={"name": name, "lat": lat, "lon": lon})

# Load the CSV file
import pandas as pd
obstacles_csv = pd.read_csv('./resources/LAC_obstacles_v1.csv')

# Create features from the CSV data
features = [
    create_circle_linestring_utm(row['lat'], row['lon'], row['LAC_Name'], row['h'], radius=600)
    for idx, row in obstacles_csv.iterrows()
]

# Create the final GeoJSON feature collection
feature_collection = geojson.FeatureCollection(features)

# Write the output GeoJSON to a file
with open('./resources/obstacles_v2.geojson', 'w') as f:
    geojson.dump(feature_collection, f)
