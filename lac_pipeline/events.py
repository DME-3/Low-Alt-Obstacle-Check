from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

DistanceFn = Callable[[tuple[float, float], tuple[float, float]], float]

OBSTACLE_EVENT_COLUMNS = [
    "icao24",
    "callsign",
    "ref",
    "closest_obst_name",
    "time",
    "lat",
    "lon",
    "cpa",
    "congested",
    "pop_density",
    "systemdesignassurance",
    "version",
    "positionnac",
    "sourceintegritylevel",
    "nic",
    "dip_max",
    "n",
    "inf_ref",
    "url",
]

GROUND_EVENT_COLUMNS = [
    "icao24",
    "callsign",
    "ref",
    "closest_obst_name",
    "time",
    "lat",
    "lon",
    "dip_max",
    "congested",
    "pop_density",
    "systemdesignassurance",
    "version",
    "positionnac",
    "sourceintegritylevel",
    "nic",
    "n",
    "inf_ref",
    "url",
]


def build_event_tables(
    final_df: pd.DataFrame,
    obstacles_df: pd.DataFrame,
    distance_fn: DistanceFn,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        build_obstacle_events(final_df, obstacles_df, distance_fn),
        build_ground_events(final_df),
    )


def build_obstacle_events(
    final_df: pd.DataFrame,
    obstacles_df: pd.DataFrame,
    distance_fn: DistanceFn,
) -> pd.DataFrame:
    inf_pt_df = final_df[final_df.inf_pt].copy()
    if inf_pt_df.empty:
        return empty_obstacle_events()

    inf_pt_df["dist_to_obs"] = inf_pt_df.apply(
        lambda row: _distance_to_obstacle(row, obstacles_df, distance_fn),
        axis=1,
    )
    inf_pt_df["time"] = pd.to_datetime(inf_pt_df["time"], unit="s")
    inf_pt_df = inf_pt_df.sort_values(by=["ref", "closest_obst_name", "time"])
    inf_pt_df["time_diff"] = inf_pt_df.groupby(["ref", "closest_obst_name"])[
        "time"
    ].diff()
    inf_pt_df["group"] = (inf_pt_df["time_diff"] >= pd.Timedelta(seconds=30)).cumsum()

    group_keys = ["ref", "closest_obst_name", "group"]
    inf_pt_df["_dist_for_idx"] = inf_pt_df["dist_to_obs"].fillna(np.inf)
    min_distance_idx = inf_pt_df.groupby(group_keys)["_dist_for_idx"].idxmin()
    inf_min_dist = inf_pt_df.loc[min_distance_idx].reset_index(drop=True)
    inf_max_dip = inf_pt_df.groupby(group_keys)["dip"].max().reset_index()
    group_size = inf_pt_df.groupby(group_keys).size().reset_index(name="n")

    result = inf_min_dist[
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
    result = result.merge(inf_max_dip, on=group_keys)
    result = result.merge(group_size, on=group_keys)
    result.rename(columns={"dist_to_obs": "cpa", "dip": "dip_max"}, inplace=True)
    result["entry_count"] = result.groupby("ref").cumcount()
    result["inf_ref"] = result["ref"].astype(str) + "_" + result["entry_count"].astype(str)
    result["url"] = result.apply(build_adsbexchange_url, axis=1)
    result = result.drop(columns=["entry_count", "group"]).reset_index(drop=True)
    return result[OBSTACLE_EVENT_COLUMNS]


def build_ground_events(final_df: pd.DataFrame) -> pd.DataFrame:
    gnd_inf_pt_df = final_df[
        final_df.gnd_inf_pt & (final_df.dip >= 0) & (final_df.closest_obst_name == "ground")
    ].copy()
    if gnd_inf_pt_df.empty:
        return empty_ground_events()

    gnd_inf_pt_df["time"] = pd.to_datetime(gnd_inf_pt_df["time"], unit="s")
    gnd_inf_pt_df = gnd_inf_pt_df.sort_values(by=["ref", "time"])
    gnd_inf_pt_df["time_diff"] = gnd_inf_pt_df.groupby(["ref"])["time"].diff()
    gnd_inf_pt_df["group"] = (
        gnd_inf_pt_df["time_diff"] >= pd.Timedelta(seconds=30)
    ).cumsum()

    group_keys = ["ref", "group"]
    max_dip_idx = gnd_inf_pt_df.groupby(group_keys)["dip"].idxmax()
    gnd_inf_max_dip = gnd_inf_pt_df.loc[max_dip_idx].reset_index(drop=True)
    group_size = gnd_inf_pt_df.groupby(group_keys).size().reset_index(name="n")

    result = gnd_inf_max_dip[
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
    result = result.merge(group_size, on=group_keys)
    result.rename(columns={"dip": "dip_max"}, inplace=True)
    result["entry_count"] = result.groupby("ref").cumcount()
    result["inf_ref"] = (
        result["ref"].astype(str) + "_gnd_" + result["entry_count"].astype(str)
    )
    result["url"] = result.apply(build_adsbexchange_url, axis=1)
    result = result.drop(columns=["entry_count", "group"]).reset_index(drop=True)
    return result[GROUND_EVENT_COLUMNS]


def build_adsbexchange_url(row: pd.Series) -> str:
    return (
        "https://globe.adsbexchange.com/"
        f"?icao={row['icao24']}&lat=50.928&lon=6.947&zoom=13.2"
        f"&showTrace={row['time'].strftime('%Y-%m-%d')}"
        f"&timestamp={int(row['time'].timestamp())}"
    )


def empty_obstacle_events() -> pd.DataFrame:
    return pd.DataFrame(columns=OBSTACLE_EVENT_COLUMNS)


def empty_ground_events() -> pd.DataFrame:
    return pd.DataFrame(columns=GROUND_EVENT_COLUMNS)


def _distance_to_obstacle(
    row: pd.Series,
    obstacles_df: pd.DataFrame,
    distance_fn: DistanceFn,
) -> float:
    if row["closest_obst_name"] == "ground":
        return np.nan

    obstacle = obstacles_df.loc[obstacles_df["LAC_Name"] == row["closest_obst_name"]]
    if obstacle.empty:
        return np.nan

    return distance_fn(
        (float(obstacle["lat"].iloc[0]), float(obstacle["lon"].iloc[0])),
        (float(row["lat"]), float(row["lon"])),
    )

