import sys
import os
import time
import utm
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

ALERT_DELTA_HEIGHT_M = 300   # delta height (should be 300)

# Define conversion functions
def utm_to_latlon(x, y):
    # Convert lat/lon to UTM coordinates
    lat, lon = utm.to_latlon(x, y, 32, 'U')

    return lat, lon

# Define conversion functions
def latlon_to_utm(lat, lon):
    # Convert lat/lon to UTM coordinates
    utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)

    return utm_x, utm_y

# functions to look up min height from arrays

def find_closest_index(x_array, x):
    # Find the index of the closest value to x
    closest_index = np.argmin(np.abs(x_array - x))
    
    return closest_index

def find_min_alt(lat, lon, x_array, y_array, z_array):

  x, y = latlon_to_utm(lat, lon)

  idx_x = find_closest_index(x_array, x)
  idx_y = find_closest_index(y_array, y)

  min_alt = z_array[idx_y][idx_x]

  return min_alt

def build_surface():

    with open('./xyz_pickles/x_results.pkl','rb') as f:
        x_results = pickle.load(f)

    with open('./xyz_pickles/y_results.pkl','rb') as f:
        y_results = pickle.load(f)

    with open('./xyz_pickles/z_results.pkl','rb') as f:
        z_results = pickle.load(f)

    x_array = np.array([])

    for i in range(len(x_results)):
        x_array = np.concatenate((x_array, x_results[i][0]))

    y_array = np.array([])

    for j in range(len(y_results)):
        y_array = np.concatenate((y_array, y_results[0][j]))

    lst = z_results

    row=len(lst)
    col=len(lst[0])

    for j in range(0, row):
        for i in range(0, col):
            if i==0:
                z_array_row = z_results[i][j]
            else:
                z_array_row = np.hstack((z_array_row, z_results[i][j]))
    
        if j==0:
            z_array = z_array_row
        else:
            z_array = np.vstack((z_array, z_array_row))

    z_array = z_array + ALERT_DELTA_HEIGHT_M

    return x_array, y_array, z_array


if __name__ == "__main__":

    # Check argument
    try:
      arg1 = sys.argv[1]
    except IndexError:
      print('Usage: ' + os.path.basename(__file__) + ' <.json gdf file to process>')
      sys.exit(1)

    # Start timer
    start_time = time.time()

    # Open the gdf file
    gdf_file = arg1

    if 'gdf_' not in gdf_file:
        print('Skipping %s (not  gdf file) and exiting.'%(gdf_file))
        sys.exit(0)
    else:
        print('Opening %s...'%(gdf_file))

    gdf = pd.read_json(gdf_file, lines=False)
    gdf = gdf.reset_index(drop=True)

    # Open pickles and build surface arrays
    x_array, y_array, z_array = build_surface()

    # Initialise min alt column with zeroes
    gdf['lidar_min_alt'] = 0

    # Iterate over gdf rows
    for row in tqdm(gdf.itertuples()):

        # Find min alt
        lidar_min_alt = find_min_alt(float(row.lat), float(row.lon), x_array, y_array, z_array)

        # Add min alt parameter to row
        gdf.loc[row[0], 'lidar_min_alt'] = lidar_min_alt

    # Save gdf and exit
    gdf.to_json(gdf_file)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Processed in {execution_time:.2f} seconds.")
    print('Saved %s. Exiting.'%(gdf_file))