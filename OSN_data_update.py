import json
from collections import defaultdict
from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import paramiko
import rasterio
import requests
import sshtunnel
from pyopensky.trino import Trino
from pyproj import Transformer
from sqlalchemy import create_engine, text, MetaData, insert
from sshtunnel import SSHTunnelForwarder

update_start_time = datetime.now()

LAT_MIN, LAT_MAX = 50.88385859501204322, 50.98427935836787128
LON_MIN, LON_MAX = 6.85029965503896943, 7.005  # 7.03641128126701965
ALT_MIN, ALT_MAX = (
    0,
    750,
)  # update from 700 m to 750 m, in line with CTR limit at 2500 ft plus margin (and accounting for Geoid Height)

TIME_BETWEEN_TRAJS = 30

# SERA.5005(f)(1) criteria
CONGESTED_ALERT_DISTANCE_M = 600  # alert distance wrt obstacles (should be 600)
CONGESTED_ALERT_DELTA_HEIGHT_M = 300  # delta height (should be 300)

# SERA.5005(f)(2) criteria
NONCONGESTED_ALERT_DISTANCE_M = 150
NONCONGESTED_ALERT_DELTA_HEIGHT_M = 150

CPA_MARGIN_M = 20  # allowance for lateral distance to obstacle
DIP_MARGIN_M = 20  # allowance for dip below minimum height (45m corresponds to GVA = 2)
N_MIN = 5

GEOID_HEIGHT_M = 47  # geoid height for Cologne

sshtunnel.SSH_TIMEOUT = 15.0
sshtunnel.TUNNEL_TIMEOUT = 15.0

MYSQL_secrets_json = "./mysql_secrets.json"
PYA_secrets_json = "./PYA_secrets.json"

with open(MYSQL_secrets_json) as MYSQL_secrets:
    MYSQL_creds = json.load(MYSQL_secrets)

with open(PYA_secrets_json) as PYA_secrets:
    PYA_creds = json.load(PYA_secrets)

username = PYA_creds["PYA_username"]
token = PYA_creds["PYA_token"]
host = PYA_creds["PYA_host"]
domain_name = PYA_creds["PYA_domain"]

ed25519_key = paramiko.Ed25519Key(filename="./.ssh/id_ed25519")

def manifest_update(engine, table_name, processed_date, record_count, start_time, end_time, status, error_message=None):

    try:

        if isinstance(start_time, int):
            start_time = datetime.fromtimestamp(start_time)
        if isinstance(end_time, int):
            end_time = datetime.fromtimestamp(end_time)

        duration_sec = int((end_time - start_time).total_seconds())

        metadata = MetaData()
        metadata.reflect(bind=engine)
        manifest_table = metadata.tables['manifest']

        insert_stmt = insert(manifest_table).values(
            table_name=table_name,
            processed_date=processed_date,
            record_count=record_count,
            start_time=start_time,
            end_time=end_time,
            duration_sec=duration_sec,
            status=status,
            error_message=error_message
        )

        with engine.begin() as connection:
            connection.execute(insert_stmt)
            print(f"Manifest entry added for table: {table_name}")

    except Exception as e:
        print(f"Failed to log update to manifest table: {e}")

# Obtain and format the date to retrieve data for (2 days ago)
two_days_ago = datetime.now() - timedelta(days=2)
date_string = two_days_ago.strftime("%Y-%m-%d")
start = two_days_ago.replace(hour=0, minute=0, second=0, microsecond=0)
end = two_days_ago.replace(hour=23, minute=59, second=59, microsecond=999999)
start_time = int(start.timestamp())
start_hour = start_time - (start_time % 3600)
end_time = int(end.timestamp())
end_hour = end_time - (end_time % 3600)

# First query for State Vectors
svdata4_query = (
    f"SELECT * FROM state_vectors_data4"
    f" WHERE icao24 LIKE '%'"
    f" AND time >= {start_time} AND time <= {end_time}"
    f" AND hour >= {start_hour} AND hour <= {end_hour}"
    f" AND lat >= {LAT_MIN} AND lat <= {LAT_MAX}"
    f" AND lon>= {LON_MIN} AND lon <= {LON_MAX}"
    f" AND geoaltitude >= {ALT_MIN} AND geoaltitude <= {ALT_MAX}"
    f" ORDER BY time"
)

print("Connecting to OSN database...")
trino = Trino()
svdata4_df = trino.query(
    svdata4_query,
    cached=False,
    compress=True,
)

# Save svdata4 pickle
svdata4_df.to_pickle(f"./OSN_pickles/svdata4df_new_{date_string}.pkl")

# Second Query for Ops Status
icao_list = svdata4_df.icao24.unique()
icao24_str = ", ".join(f"'{item}'" for item in icao_list)

ops_sts_query = (
    f"SELECT icao24, mintime, maxtime, nacv, systemdesignassurance, version, positionnac, geometricverticalaccuracy, sourceintegritylevel, barometricaltitudeintegritycode  FROM operational_status_data4"
    f" WHERE icao24 IN ({icao24_str})"
    f" AND mintime >= {start_time} AND maxtime <= {end_time}"
    f" AND hour >= {start_hour} AND hour <= {end_hour}"
    f" ORDER by mintime"
)

print("Connecting to OSN database...")
trino = Trino()
ops_sts_df = trino.query(
    ops_sts_query,
    cached=False,
)

ops_sts_df["time"] = ops_sts_df["mintime"].astype("int64")

# Save ops_sts pickle
ops_sts_df.to_pickle(f"./OSN_pickles/opsstsdf_new_{date_string}.pkl")

# Initialize an empty DataFrame to hold the results
merged_df = pd.DataFrame()

# Loop over each unique 'icao24' in both dataframes
unique_icao24s = pd.concat([svdata4_df["icao24"], ops_sts_df["icao24"]]).unique()

for icao24 in unique_icao24s:
    # Filter each dataframe by 'icao24'
    sub_df1 = svdata4_df[svdata4_df["icao24"] == icao24]
    sub_df2 = ops_sts_df[ops_sts_df["icao24"] == icao24]

    # Ensure both sub-dataframes are sorted by 'time'
    sub_df1 = sub_df1.sort_values("time")
    sub_df2 = sub_df2.sort_values("time")

    sub_df1["icao24"] = sub_df1["icao24"].astype("object")
    sub_df2["icao24"] = sub_df2["icao24"].astype("object")

    sub_df1["time"] = sub_df1["time"].astype("int64")
    sub_df2["time"] = sub_df2["time"].astype("int64")

    # Perform merge_asof on the filtered and sorted dataframes
    merged_sub_df = pd.merge_asof(
        sub_df1, sub_df2, on="time", by="icao24", direction="backward"
    )

    # Append the result to the main dataframe
    merged_df = pd.concat([merged_df, merged_sub_df], ignore_index=True)

# Third Query for Position data (to get the NIC)
posdata4_query = (
    f"SELECT mintime, icao24, nic  FROM position_data4"
    f" WHERE icao24 IN ({icao24_str})"
    f" AND lat >= {LAT_MIN} AND lat <= {LAT_MAX}"
    f" AND lon>= {LON_MIN} AND lon <= {LON_MAX}"
    f" AND mintime >= {start_time} AND maxtime <= {end_time}"
    f" AND hour >= {start_hour} AND hour <= {end_hour}"
    f" ORDER by mintime"
)

print("Connecting to OSN database...")
trino = Trino()
posdata4_df = trino.query(
    posdata4_query,
    cached=False,
)

posdata4_df["time"] = posdata4_df["mintime"].astype("int64")

# Save ops_sts pickle
posdata4_df.to_pickle(f"./OSN_pickles/posdata4df_new_{date_string}.pkl")

# Initialize an empty DataFrame to hold the results
final_df = pd.DataFrame()

for icao24 in unique_icao24s:
    # Filter each dataframe by 'icao24'
    sub_df1 = merged_df[merged_df["icao24"] == icao24]
    sub_df2 = posdata4_df[posdata4_df["icao24"] == icao24]

    # Ensure both sub-dataframes are sorted by 'time'
    sub_df1 = sub_df1.sort_values("time")
    sub_df2 = sub_df2.sort_values("time")

    sub_df1["icao24"] = sub_df1["icao24"].astype("object")
    sub_df2["icao24"] = sub_df2["icao24"].astype("object")

    # Perform merge_asof on the filtered and sorted dataframes
    merged_sub_df = pd.merge_asof(
        sub_df1, sub_df2, on="time", by="icao24", direction="backward"
    )

    # Append the result to the main dataframe
    final_df = pd.concat([final_df, merged_sub_df], ignore_index=True)

final_df = final_df.drop(columns=["hour", "mintime_x", "maxtime", "mintime_y"])

## Add DEM ground elevation information

crs_transformer = Transformer.from_crs(
    4326, 25832, always_xy=True
)  # Transformer from WGS-84 to ETRS89-LAEA (3035 for EU-DEM v1.1, 25832 for LAS DEM)

def transform_coords(lon, lat):
    return crs_transformer.transform(lon, lat)

final_df["gnd_elev"] = np.nan

dem_src = rasterio.open("./resources/Cologne_DEM_merged_from_LAS_25x25.tif")

final_df["etrs89_x"], final_df["etrs89_y"] = zip(
    *final_df.apply(lambda row: transform_coords(row["lon"], row["lat"]), axis=1)
)

def get_elevation(x, y, dem, default_value=None):
    try:
        row, col = dem.index(x, y)
        return dem.read(1)[row, col]
    except IndexError:
        return default_value

final_df["gnd_elev"] = final_df.apply(
    lambda row: get_elevation(
        row["etrs89_x"], row["etrs89_y"], dem_src, row.get("gnd_elev", None)
    ),
    axis=1,
)

missing_elev_count = final_df["gnd_elev"].isnull().sum()
print(f"Number of rows with None or NaN in 'gnd_elev': {missing_elev_count}")
final_df = final_df.dropna(subset=["gnd_elev"])

## Add population density

final_df["pop_density"] = np.nan

pop_src = rasterio.open(
    "./resources/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_Cologne.tif"
)

def get_population(lon, lat, pop_src):
    row, col = pop_src.index(lon, lat)
    return pop_src.read(1)[row, col]

final_df["pop_density"] = final_df.apply(
    lambda row: get_population(row["lon"], row["lat"], pop_src), axis=1
)

## Process df with distance information

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

final_df["prev_time"] = final_df.time.shift()
final_df["closest_obst_name"] = ""
final_df["inf_flt"] = False
final_df["inf_pt"] = False
final_df["gnd_inf_flt"] = False
final_df["gnd_inf_pt"] = False
final_df["min_hgt"] = np.nan
final_df["congested"] = final_df["pop_density"] > 2

map_time_traj = defaultdict(dict)

for icao, sfinal_df in final_df.groupby("icao24"):
    map_time_traj[icao][sfinal_df.iloc[0]["time"]] = icao + "_1"
    n_traj = 1
    for i in range(1, sfinal_df.shape[0]):
        time = sfinal_df.iloc[i]["time"]
        diff = abs(int(time) - int(sfinal_df.iloc[i]["prev_time"]))
        if diff > TIME_BETWEEN_TRAJS:
            n_traj += 1
        map_time_traj[icao][time] = icao + "_" + str(int(n_traj))

final_df["ref"] = (
    final_df.apply(lambda x: map_time_traj[x.icao24][x.time], axis=1)
    + "_"
    + final_df.time.apply(lambda x: pd.to_datetime(x, unit="s").strftime("%d%m%y"))
)

# Add a distance column and compute cumulative along-track distance for each flight
final_df["dist"] = 0.0
for flight in final_df.ref.unique():
    current = final_df[
        final_df["ref"].isin([flight])
    ]  # gets the trajectory of the current flight
    previous_pt = None
    previous_dist = 0
    for row in current.itertuples():
        current_pt = (float(row.lat), float(row.lon))
        if previous_pt is not None:  # Skip the distance calculation for the first point
            delta_dist = haversine(previous_pt, current_pt)
            final_df.loc[row[0], "dist"] = previous_dist + delta_dist
        previous_pt = current_pt
        previous_dist = final_df.loc[row[0], "dist"]

## Load obstacle information and check min height

path_to_obstacles_json = "./resources/LAC_obstacles_v1.csv"

obs_df = pd.read_csv(path_to_obstacles_json)

obs_df.rename(columns={"h": "height_m"}, inplace=True)

obs_df = obs_df.sort_values(
    by=["height_m"]
)  # sort obstacles by incresing height, to avoid that the min_hgt profil is wrong if a shorter obstacle comes after a taller one, in case the aircraft is within two obstacles clearance areas

# new. Test if this should not be done before adding the ref and distance info
final_df = final_df.reset_index(drop=True)

def update_closest_obstacle_xy(final_df, obstacles_df):
    # Extract the coordinates and obstacle heights
    final_coords = final_df[["etrs89_x", "etrs89_y"]].to_numpy()
    obstacle_coords = obstacles_df[["x", "y"]].to_numpy()
    obstacle_heights = obstacles_df["height_m"].to_numpy()
    obstacle_names = obstacles_df["LAC_Name"].to_numpy()
    obstacle_ground_elevs = obstacles_df["gnd_elev"].to_numpy()

    # Iterate over each point in final_df and calculate distances
    for i, (x_f, y_f) in enumerate(final_coords):
        radius = (
            CONGESTED_ALERT_DISTANCE_M
            if final_df.at[i, "congested"]
            else NONCONGESTED_ALERT_DISTANCE_M
        )

        # Convert radius to a squared radius for comparison with squared distances (faster than computing square roots)
        squared_radius = radius**2

        # Calculate squared Euclidean distances to all obstacles
        distances_sq = (obstacle_coords[:, 0] - x_f) ** 2 + (
            obstacle_coords[:, 1] - y_f
        ) ** 2

        # Filter obstacles within the radius
        within_radius = distances_sq <= squared_radius

        # If there are obstacles within the radius, find the tallest one
        if np.any(within_radius):
            # Get the indices of obstacles within the radius
            obstacles_in_radius_idx = np.where(within_radius)[0]

            # Find the index of the tallest obstacle within the radius
            tallest_idx = obstacles_in_radius_idx[
                np.argmax(obstacle_heights[obstacles_in_radius_idx])
            ]

            # Update final_df with the tallest obstacle details
            tallest_obstacle_name = obstacle_names[tallest_idx]
            tallest_obstacle_height = obstacle_heights[tallest_idx]
            tallest_obstacle_elev = obstacle_ground_elevs[tallest_idx]

            final_df.at[i, "closest_obst_name"] = tallest_obstacle_name

            # Important: Calculate the minimum height, considerint the tallest obstacle in the alert radius
            # min_hgt is referenced to the Geoid !
            final_df.at[i, "min_hgt"] = (
                GEOID_HEIGHT_M
                + np.float32(tallest_obstacle_elev)
                + np.float32(tallest_obstacle_height)
                + (
                    CONGESTED_ALERT_DELTA_HEIGHT_M
                    if final_df.at[i, "congested"]
                    else NONCONGESTED_ALERT_DELTA_HEIGHT_M
                )
            )
        else:
            final_df.at[i, "closest_obst_name"] = "ground"
            final_df.at[i, "min_hgt"] = (
                GEOID_HEIGHT_M +
                CONGESTED_ALERT_DELTA_HEIGHT_M
                if final_df.at[i, "congested"]
                else NONCONGESTED_ALERT_DELTA_HEIGHT_M
            )

    return final_df

final_df = update_closest_obstacle_xy(final_df, obs_df)

final_df["dip"] = final_df["min_hgt"] - final_df["geoaltitude"]

## Add infraction information

# Rule for 'inf_pt'
final_df["inf_pt"] = final_df.apply(
    lambda row: True
    if row["dip"] > 0 and row["closest_obst_name"] != "ground"
    else False,
    axis=1,
)

# Rule for 'gnd_inf_pt'
final_df["gnd_inf_pt"] = final_df.apply(
    lambda row: True
    if (row["dip"] > 0 and row["closest_obst_name"] == "ground")
    else False,
    axis=1,
)

# Group by 'ref' and update 'inf_flt' based on 'inf_pt'
final_df["inf_flt"] = final_df.groupby("ref")["inf_pt"].transform("any")

# Group by 'ref' and update 'gnd_inf_flt' based on 'gnd_inf_pt'
final_df["gnd_inf_flt"] = final_df.groupby("ref")["gnd_inf_pt"].transform("any")

final_df = final_df.drop(columns=["serials", "nacv"])

final_df[final_df["gnd_inf_pt"]].callsign.unique()

missing_callsign = final_df["callsign"].isnull().sum()
print(f"Number of rows with None or NaN in 'callsign': {missing_callsign}")
final_df = final_df.dropna(subset=["callsign"])

## Create infraction tables

inf_result = pd.DataFrame
gnd_inf_result = pd.DataFrame()

inf_pt_df = final_df[final_df.inf_pt].copy()

inf_pt_df["dist_to_obs"] = np.nan

if not inf_pt_df.empty:
    inf_pt_df["dist_to_obs"] = inf_pt_df.apply(
        lambda x: haversine(
            (
                obs_df.loc[obs_df["LAC_Name"] == x["closest_obst_name"], "lat"].iloc[0],
                obs_df.loc[obs_df["LAC_Name"] == x["closest_obst_name"], "lon"].iloc[0],
            ),
            (x["lat"], x["lon"]),
        )
        if not (x["closest_obst_name"] == "ground")
        else np.nan,
        axis=1,
    )

inf_pt_df["time"] = pd.to_datetime(inf_pt_df["time"], unit="s")
inf_pt_df = inf_pt_df.sort_values(by=["ref", "closest_obst_name", "time"])

inf_pt_df["time_diff"] = inf_pt_df.groupby(["ref", "closest_obst_name"])["time"].diff()
inf_pt_df["group"] = (inf_pt_df["time_diff"] >= pd.Timedelta(seconds=30)).cumsum()

inf_grouped = inf_pt_df.groupby(["ref", "closest_obst_name", "group"])

inf_min_dist = inf_grouped.apply(
    lambda x: x.loc[x["dist_to_obs"].idxmin()]
).reset_index(drop=True)
inf_max_dip = inf_grouped["dip"].max().reset_index()

group_size = inf_grouped.size().reset_index(name="n")

inf_result = inf_min_dist[
    [
        "icao24",
        "callsign",
        "group",
        "ref",
        "closest_obst_name",
        "time",
        "lat",
        "lon",
        "dist_to_obs",
        "congested",
        "pop_density",
        "systemdesignassurance",
        "version",
        "positionnac",
        "sourceintegritylevel",
        "nic",
    ]
].copy()
inf_result = inf_result.merge(inf_max_dip, on=["ref", "closest_obst_name", "group"])
inf_result = inf_result.merge(group_size, on=["ref", "closest_obst_name", "group"])

inf_result.rename(columns={"dist_to_obs": "cpa", "dip": "dip_max"}, inplace=True)

inf_result["entry_count"] = inf_result.groupby("ref").cumcount()
inf_result["inf_ref"] = (
    inf_result["ref"].astype(str) + "_" + inf_result["entry_count"].astype(str)
)

if not inf_result.empty:
    inf_result["url"] = inf_result.apply(
        lambda row: "https://globe.adsbexchange.com/?icao=%s&lat=50.928&lon=6.947&zoom=13.2&showTrace=%s&timestamp=%s"
        % (
            row["icao24"],
            row["time"].strftime("%Y-%m-%d"),
            str(int(row["time"].timestamp())),
        ),
        axis=1,
    )
else:
    inf_result["url"] = []

inf_result = inf_result.reset_index(drop=True)

inf_result = inf_result.drop(columns=["entry_count", "group"])

# Ground infractions

gnd_inf_pt_df = final_df[
    final_df.gnd_inf_pt & (final_df.dip >= 0) & (final_df.closest_obst_name == "ground")
].copy()

gnd_inf_pt_df["time"] = pd.to_datetime(gnd_inf_pt_df["time"], unit="s")
gnd_inf_pt_df = gnd_inf_pt_df.sort_values(by=["ref", "time"])

gnd_inf_pt_df["time_diff"] = gnd_inf_pt_df.groupby(["ref"])["time"].diff()
gnd_inf_pt_df["group"] = (
    gnd_inf_pt_df["time_diff"] >= pd.Timedelta(seconds=30)
).cumsum()

gnd_inf_grouped = gnd_inf_pt_df.groupby(["ref", "group"])

gnd_inf_max_dip = gnd_inf_grouped.apply(lambda x: x.loc[x["dip"].idxmax()]).reset_index(
    drop=True
)

group_size = gnd_inf_grouped.size().reset_index(name="n")

gnd_inf_result = gnd_inf_max_dip[
    [
        "icao24",
        "callsign",
        "group",
        "ref",
        "closest_obst_name",
        "time",
        "lat",
        "lon",
        "dip",
        "congested",
        "pop_density",
        "systemdesignassurance",
        "version",
        "positionnac",
        "sourceintegritylevel",
        "nic",
    ]
].copy()

gnd_inf_result = gnd_inf_result.merge(group_size, on=["ref", "group"])

gnd_inf_result.rename(columns={"dip": "dip_max"}, inplace=True)

gnd_inf_result["entry_count"] = gnd_inf_result.groupby("ref").cumcount()
gnd_inf_result["inf_ref"] = (
    gnd_inf_result["ref"].astype(str)
    + "_"
    + "gnd_"
    + gnd_inf_result["entry_count"].astype(str)
)

if not gnd_inf_result.empty:
    gnd_inf_result["url"] = gnd_inf_result.apply(
        lambda row: "https://globe.adsbexchange.com/?icao=%s&lat=50.928&lon=6.947&zoom=13.2&showTrace=%s&timestamp=%s"
        % (
            row["icao24"],
            row["time"].strftime("%Y-%m-%d"),
            str(int(row["time"].timestamp())),
        ),
        axis=1,
    )
else:
     gnd_inf_result["url"] = []

gnd_inf_result = gnd_inf_result.reset_index(drop=True)

gnd_inf_result = gnd_inf_result.drop(columns=["entry_count", "group"])

###

gdf_record_count = len(final_df)
inf_record_count = len(inf_result)
gndinf_record_count = len(gnd_inf_result)
processed_date = two_days_ago.date()

### Upload data to the MySQL server

with SSHTunnelForwarder(
    (MYSQL_creds["SSH_ADDRESS"], 22),
    ssh_username=MYSQL_creds["SSH_USERNAME"],
    ssh_pkey=ed25519_key,  # Use the loaded RSA key
    remote_bind_address=(
        MYSQL_creds["REMOTE_BIND_ADDRESS"],
        MYSQL_creds["REMOTE_BIND_PORT"],
    ),
    allow_agent=False,
) as tunnel:
    engstr = (
        "mysql+pymysql://"
        + MYSQL_creds["SSH_USERNAME"]
        + ":"
        + MYSQL_creds["PYANYWHERE_PASSWORD"]
        + "@127.0.0.1:"
        + str(tunnel.local_bind_port)
        + "/dme3$"
        + MYSQL_creds["PROD_DATABASE_NAME"]
    )

    engine = create_engine(engstr)

    query = text(
        "SELECT max(processed_date) as max_date FROM manifest"
    )
    result = pd.read_sql_query(query, con=engine)
    max_date = result["max_date"].iloc[0]

if max_date < two_days_ago.date():
    # Set up the SSH tunnel with the RSA key
    with SSHTunnelForwarder(
        (MYSQL_creds["SSH_ADDRESS"], 22),
        ssh_username=MYSQL_creds["SSH_USERNAME"],
        ssh_pkey=ed25519_key,
        remote_bind_address=(
            MYSQL_creds["REMOTE_BIND_ADDRESS"],
            MYSQL_creds["REMOTE_BIND_PORT"],
        ),
        allow_agent=False,
    ) as tunnel:
        print("connected")

        engstr = (
            "mysql+pymysql://"
            + MYSQL_creds["SSH_USERNAME"]
            + ":"
            + MYSQL_creds["PYANYWHERE_PASSWORD"]
            + "@127.0.0.1:"
            + str(tunnel.local_bind_port)
            + "/dme3$"
            + MYSQL_creds["PROD_DATABASE_NAME"]
        )

        engine = create_engine(engstr)

        print("step 1")
        final_df.to_sql(
            con=engine, name=MYSQL_creds["MAIN_PROD_TABLE_NAME"], if_exists="append"
        )
        manifest_update(engine, MYSQL_creds["MAIN_PROD_TABLE_NAME"], processed_date, gdf_record_count, update_start_time, datetime.now(), 'SUCCESS', None)

        print("step 2")
        inf_result.to_sql(
            con=engine, name=MYSQL_creds["INF_PROD_TABLE_NAME"], if_exists="append"
        )
        manifest_update(engine, MYSQL_creds["INF_PROD_TABLE_NAME"], processed_date, inf_record_count, update_start_time, datetime.now(), 'SUCCESS', None)

        print("step 3")
        gnd_inf_result.to_sql(
            con=engine, name=MYSQL_creds["GNDINF_PROD_TABLE_NAME"], if_exists="append"
        )
        manifest_update(engine, MYSQL_creds["GNDINF_PROD_TABLE_NAME"], processed_date, gndinf_record_count, update_start_time, datetime.now(), 'SUCCESS', None)

        print("Insertion in database done")
else:
    print('Date already in database, exiting...')
    exit()

### Reload Web app

response = requests.post(
    "https://{host}/api/v0/user/{username}/webapps/{domain_name}/reload/".format(
        host=host, username=username, domain_name=domain_name
    ),
    headers={"Authorization": "Token {token}".format(token=token)},
)

if response.status_code == 200:
    print("Web app reloaded successfully.")
    print(response.content)
else:
    print(f"Error: Received status code {response.status_code}")
    print("Response content:", response.content)
