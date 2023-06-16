import pandas as pd
import numpy as np
import json
import pickle
import sys
import rasterio
import os
import utm
from osgeo import gdal # When GDAL is installed with Conda
from pyproj import Transformer, transform
from shapely.geometry import Point
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict
from glob import glob
#from geopy.distance import geodesic # Only necessary if geodesic is used

import dem_func
from polygons import rhein_polygon, cologne_polygon

dem_file = './resources/Cologne_EUDEM_v11.tif'
path_to_obstacles_json = './resources/obstacles.json'
dataframes_path = './dataframes/'

TIME_BETWEEN_TRAJS = 30  # if two points have consecutive times of more than that, we consider two trajectories for the same icao24
USEDEM = True

# SERA.5005(f)(1) criteria
ALERT_DISTANCE_M = 600      # alert distance wrt obstacles (should be 600)
ALERT_DELTA_HEIGHT_M = 300   # delta height (should be 300)

CPA_MARGIN_M = 20 # allowance for lateral distance to obstacle
DIP_MARGIN_M = 20 # allowance for dip below minimum height (45m corresponds to GVA = 2)
N_MIN = 5

GEOID_HEIGHT_M = 47  # geoid height for Cologne

DEFAULT_GND_ELEV_M = 50 # default ground elevation used to calculate dip below minimum height above ground (away from obstacles)

# Conversion WGS84 to UTM
def latlon_to_utm(lat, lon):
    # Convert lat/lon to UTM coordinates
    utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)
    return utm_x, utm_y

# Functions to look up min height from arrays
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

# Load xyz pickles, build LiDAR min alt surface and return arrays
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

def get_line_lst(line_txt):
    '''
    Return a list of non-empty strings that were delimited by the "|" character in the input string.
    Leading and trailing whitespace are removed
    '''
    return [elt.strip() for elt in line_txt.split("|") if elt.strip()]

def haversine(pt1, pt2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Returned units are in metres. Differs slightly from PostGIS geography
    distance, which uses a spheroid, rather than a sphere.
    """

    lat1, lon1 = pt1[0], pt1[1]
    lat2, lon2 = pt2[0], pt2[1]

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of earth in m
    return c * r

def process_data(*args):

  osn_data = []
  with open(args[0], 'rb') as f:
    osn_data += pickle.load(f)

  gdal_data = gdal.Open(dem_file)
  gdal_band = gdal_data.GetRasterBand(1)
  nodataval = gdal_band.GetNoDataValue()
  elev_array = gdal_data.ReadAsArray().astype(float) # convert to a numpy array

  # Open pickles and build surface arrays
  x_array, y_array, z_array = build_surface()

  # replace missing values if necessary
  if np.any(elev_array == nodataval):
      elev_array[elev_array == nodataval] = np.nan

  crs_transformer = Transformer.from_crs(4326, 3035, always_xy = True) # Transformer from WGS-84 to ETRS89-LAEA

  columns = [
      "time",
      "icao24",
      "lat",
      "lon",
      "velocity",
      "heading",
      "vertrate",
      "callsign",
      "onground",
      "alert",
      "spi",
      "squawk",
      "baroaltitude",
      "geoaltitude",
      "lastposupdate",
      "lastcontact",
      "hour",
  ]

  # Load OSN data in a dataframe
  df = pd.DataFrame([get_line_lst(osn_data[i]) for i in range(3, len(osn_data) - 1) if not osn_data[i].startswith("+-")], columns=columns)

  df.dropna(inplace=True)

  df = df[df["callsign"] != "callsign"]
  df = df[df["callsign"] != ""]

  df = df.reset_index(drop=True)

  ## Remove callsign for government, military and ambulance flights
  #
  callsign_exceptions = ["CHX", "HUMMEL", "BPO", "SAR", "JOKER", "FCK", "IBIS", "HELI", "AIRESC", "GAM", "RESQ"]
  pattern = '|'.join(callsign_exceptions)
  mask = df['callsign'].str.contains(pattern)
  excluded_rows = df[mask]
  unique_excluded_callsigns = excluded_rows['callsign'].nunique()
  print('%s occurences of callsign exceptions to drop from dataframe'%(str(unique_excluded_callsigns)))
  df = df[~mask]
  #
  ##

  gdf = df.copy()

  # Add column to have an unique identifier for each trajectory (same icao24 but different trajectories)
  gdf["prev_time"] = gdf.time.shift()

  map_time_traj = defaultdict(dict)

  for icao, sgdf in gdf.groupby("icao24"):
      map_time_traj[icao][sgdf.iloc[0]["time"]] = icao + "_1"
      n_traj = 1
      for i in range(1, sgdf.shape[0]):
          time = sgdf.iloc[i]["time"]
          diff = abs(int(time) - int(sgdf.iloc[i]["prev_time"]))
          if diff > TIME_BETWEEN_TRAJS:
              n_traj += 1
          map_time_traj[icao][time] = icao + "_" + str(int(n_traj))

  gdf['ref'] = gdf.apply(lambda x: map_time_traj[x.icao24][x.time], axis=1) + '_' + gdf.time.apply(lambda x: pd.to_datetime(x, unit='s').strftime("%d%m%y"))

  gdf.drop(['prev_time', 'spi', 'alert', 'vertrate'], axis=1, inplace=True)
  gdf['inf_flt'] = False
  gdf['inf_pt'] = False

  # Add a distance column and compute cumulative along-track distance for each flight
  gdf['dist'] = 0.0
  for flight in gdf.ref.unique():
    first = True
    current = gdf[gdf['ref'].isin([flight])] # gets the trajectory of the current flight
    for row in current.itertuples():
      if not(first):
        current_pt = (float(row.lat), float(row.lon))
        delta_dist = haversine(previous_pt, current_pt)
        gdf.loc[row[0],'dist'] = previous_dist + delta_dist
      previous_pt = (float(row.lat), float(row.lon))
      previous_dist = gdf.loc[row[0],'dist']
      first = False

  # Get ground elevation for each trajectory point
  gdf['gnd_elev'] = np.nan
  if USEDEM:
    print('Getting ground elevation for all trajectory points')
    dem_src = rasterio.open(dem_file)
    for index, row in tqdm(gdf.iterrows()):
      gdf.at[index, 'gnd_elev'] = dem_func.get_elev(elev_array, (row['lat'], row['lon']), crs_transformer, dem_src)

  # Open obstacle database
  with open(path_to_obstacles_json) as obstacles_database:
      obstacles_data = json.load(obstacles_database)
  obs_df = pd.json_normalize(obstacles_data, record_path =['obstacles'])

  # Get ground elevation for each obstacle
  obs_df['dem_gnd_elev'] = np.nan
  if USEDEM:
    dem_src = rasterio.open(dem_file)
    for index, row in obs_df.iterrows():
      obs_df.at[index, 'dem_gnd_elev'] = dem_func.get_elev(elev_array, (row['lat'], row['lon']), crs_transformer, dem_src)

  obs_df = obs_df.sort_values(by=['height_m']) # sort obstacles by incresing height, to avoid that the min_hgt profil is wrong if a shorter obstacle comes after a taller one, in case the aircraft is within two obstacles clearance areas

  gdf['inf_flt'] = False
  gdf['inf_pt'] = False
  gdf['gnd_inf_flt'] = False
  gdf['gnd_inf_pt'] = False
  gdf['min_hgt'] = gdf['gnd_elev'] + 300 # Minimum height away from obstacles is 300 m above ground (over congested areas)
  
  # Add the LiDAR minimum altitude information
  gdf['lidar_min_alt'] = 0
  for row in tqdm(gdf.itertuples()):
      # Find min alt
      lidar_min_alt = find_min_alt(float(row.lat), float(row.lon), x_array, y_array, z_array)
      # Add min alt parameter to row
      gdf.loc[row[0], 'lidar_min_alt'] = lidar_min_alt

  # prepare the infraction dataframe to store infraction (1 line per infraction)
  inf_df = pd.DataFrame(columns=
    [
      'timestamp_cpa', 
      'icao24', 
      'callsign', 
      'obstacle_name', 
      'cpa', 
      'dip_max', 
      'n', 
      'ref', 
      'url', 
      'hour'
    ]) 

  # prepare the ground infraction dataframe to store infraction (1 line per infraction)
  gnd_inf_df = pd.DataFrame(columns=
    [
      'timestamp_gprox', 
      'icao24', 
      'callsign', 
      'obstacle_name', 
      'dip_max', 
      'n', 
      'ref', 
      'url', 
      'hour'
    ]) 

  i = 0 # infraction counter
  gi = 0 # ground infraction counter

  print('Checking each flight for minimum height compliance')

  for flight in tqdm(gdf.ref.unique()):  # loop on individual flights
    current = gdf[gdf['ref'].isin([flight])] # gets the trajectory of the current flight
    for row_obs in obs_df.itertuples():  # for each flight, loop on obstacles
      infraction = False
      cpa = ALERT_DISTANCE_M + 999
      dip_max = 0
      inf_obs = ""
      ac = ""
      icao = ""
      n = 0
      cpa_time = 0
      cpa_timestamp = ""

      obs_pt = (float(row_obs.lat), float(row_obs.lon))
      obs_h_m = float(row_obs.height_m)

      if USEDEM:
        obs_elev_m = row_obs.dem_gnd_elev # Use the ground elevation calculated from the DEM
      else:
        obs_elev_m = float(row_obs.terrain_elevation_m) # Or, use the ground elevation from the obstacle JSON

      for row in current.itertuples():  # for each flight and a given obstacle, loop on trajectory points
        if row.onground=='false':
          ac_pt = (float(row.lat),float(row.lon))
          
          #dist = geodesic(obs_pt,ac_pt).m  # geodesic is accurate but slow
          dist = haversine(obs_pt,ac_pt)    # haversine function is less accurate but fast (typically more than twice as fast than haversine, error about 1 meter)

          dip = GEOID_HEIGHT_M + obs_elev_m + obs_h_m + ALERT_DELTA_HEIGHT_M - float(row.geoaltitude)

          in_rhein = rhein_polygon.contains(Point(float(row.lon), float(row.lat)))

          if dist < ALERT_DISTANCE_M:

            gdf.loc[row[0], 'min_hgt'] = obs_elev_m + obs_h_m + ALERT_DELTA_HEIGHT_M

            if dip > 0 and not(in_rhein):

              infraction = True
              n += 1
              
              gdf.loc[row[0],'inf_pt'] = True

              if dip > dip_max:
                dip_max = dip

              if dist < cpa:
                cpa = dist
                inf_obs = row_obs.name
                cpa_timestamp = row.time
                cpa_time = int(cpa_timestamp)

      if infraction:
        i +=1
        
        gdf['inf_flt'] = np.where(gdf.ref == row.ref, True, gdf.inf_flt)
        
        url = "https://globe.adsbexchange.com/?icao=%s&lat=50.928&lon=6.947&zoom=13.2&showTrace=%s&timestamp=%s" % (row.icao24, str(pd.to_datetime(cpa_time, utc=True, unit='s'))[:-15], cpa_timestamp)
        
        inf_df.loc[i] = [
          str(pd.to_datetime(cpa_time, utc=True, unit='s'))[:-6], 
          row.icao24, 
          row.callsign, 
          inf_obs, 
          round(cpa), 
          round(dip_max), 
          n, 
          flight, 
          url, 
          row.hour
        ]

    g = 0
    infraction_gnd = False
    dip_max_gnd = 0
    gprox_time = 0
    gprox_timestamp = ""

    for row in current.itertuples():  # for each flight, loop on trajectory point to perform the ground check
      if row.onground=='false':
          ac_pt = (float(row.lat),float(row.lon))
          
          dip_gnd = GEOID_HEIGHT_M + DEFAULT_GND_ELEV_M + ALERT_DELTA_HEIGHT_M - float(row.geoaltitude)
          
          in_rhein = rhein_polygon.contains(Point(float(row.lon), float(row.lat)))
          in_cologne = cologne_polygon.contains(Point(float(row.lon), float(row.lat)))
          
          if dip_gnd > 0 and not(in_rhein) and in_cologne:
            infraction_gnd = True
            g += 1
            gdf.loc[row[0],'gnd_inf_pt'] = True
            if dip_gnd > dip_max_gnd:
              dip_max_gnd = dip_gnd
              gprox_timestamp = row.time
              gprox_time = int(gprox_timestamp)
    
    if infraction_gnd:
      gi +=1
      
      gdf['gnd_inf_flt'] = np.where(gdf.ref == row.ref, True, gdf.gnd_inf_flt) # bugfix : was using gdf.inf_flt as default instead of gdf.gnd_inf_flt
      
      url = "https://globe.adsbexchange.com/?icao=%s&lat=50.928&lon=6.947&zoom=13.2&showTrace=%s&timestamp=%s" % (row.icao24, str(pd.to_datetime(gprox_time, utc=True, unit='s'))[:-15], gprox_timestamp)
      
      gnd_inf_df.loc[gi] = [
        str(pd.to_datetime(gprox_time, utc=True, unit='s'))[:-6], 
        row.icao24, 
        row.callsign, 
        "Below 300m AGL", 
        round(dip_max_gnd), 
        g, 
        flight, 
        url, 
        row.hour
      ]

  inf_df = inf_df.sort_values(by='timestamp_cpa') # Sorts the dataframe chronologically
  inf_df = inf_df.reset_index(drop = True) # reindex the dataframe (remove the old index)

  gnd_inf_df = gnd_inf_df.sort_values(by='timestamp_gprox') # Sorts the dataframe chronologically
  gnd_inf_df = gnd_inf_df.reset_index(drop = True) # reindex the dataframe (remove the old index)

  # Cleans the dataframes by applying the defined margins
  clean_inf_df = inf_df[(inf_df["cpa"] < (ALERT_DISTANCE_M - CPA_MARGIN_M)) & (inf_df["dip_max"] > DIP_MARGIN_M) & (inf_df["n"] >= N_MIN)].reset_index(drop = True) 
  clean_gnd_inf_df = gnd_inf_df[(gnd_inf_df["dip_max"] > 1) & (gnd_inf_df["n"] >= 3)].reset_index(drop = True)

  return gdf, clean_inf_df, clean_gnd_inf_df


if __name__ == "__main__":
  try:
      arg1 = sys.argv[1]
  except IndexError:
      print('Usage: ' + os.path.basename(__file__) + ' <.pkl file to process>')
      sys.exit(1)
  
  # TODO: check that pickle file name contains proper date range
  # Format example: './OSN_pickles/svdata4_2022-12-22_2023-01-22.pkl'
  date_range = arg1[-25:-4]

  # start the program

  print('Processing the data...')
  gdf, clean_inf_df, clean_gnd_inf_df = process_data(arg1)
  print('Data processed, saving...')

  # Save the dataframes

  df_path = dataframes_path + date_range
  if not os.path.exists(df_path):
      os.makedirs(df_path)

  gdf_json = df_path + '/gdf_%s.json'%(date_range)
  gdf.to_json(gdf_json)
  print('Saved %s'%(gdf_json))

  clean_inf_df_json = df_path + '/clean_inf_df_%s.json'%(date_range)
  clean_inf_df.to_json(clean_inf_df_json)
  print('Saved %s'%(clean_inf_df_json))

  clean_gnd_inf_df_json = df_path + '/clean_gnd_inf_df_%s.json'%(date_range)
  clean_gnd_inf_df.to_json(clean_gnd_inf_df_json)
  print('Saved %s'%(clean_gnd_inf_df_json))

  print('Done, exiting.')