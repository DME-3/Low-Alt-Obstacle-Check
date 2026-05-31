from __future__ import annotations

from collections.abc import Callable
from math import asin, cos, radians, sin, sqrt

import pandas as pd

DistanceFn = Callable[[tuple[float, float], tuple[float, float]], float]


def haversine(pt1: tuple[float, float], pt2: tuple[float, float]) -> float:
    """
    Calculate the great-circle distance between decimal-degree points in metres.

    This intentionally preserves the historical spherical calculation. It differs
    slightly from PostGIS geography distance, which uses a spheroid.
    """

    lat1, lon1 = pt1[0], pt1[1]
    lat2, lon2 = pt2[0], pt2[1]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    c = 2 * asin(sqrt(a))
    return c * 6371000


def add_trajectory_columns(
    frame: pd.DataFrame,
    time_between_trajs: int,
    distance_fn: DistanceFn = haversine,
) -> pd.DataFrame:
    result = add_trajectory_refs(frame, time_between_trajs)
    return add_along_track_distance(result, distance_fn=distance_fn)


def add_trajectory_refs(frame: pd.DataFrame, time_between_trajs: int) -> pd.DataFrame:
    result = frame.copy()
    result["prev_time"] = result.groupby("icao24")["time"].shift()
    result["_traj_gap"] = (
        (result["time"] - result["prev_time"]).abs() > time_between_trajs
    ).fillna(False)
    result["_traj_num"] = result.groupby("icao24")["_traj_gap"].cumsum() + 1
    result["ref"] = (
        result["icao24"].astype(str)
        + "_"
        + result["_traj_num"].astype(int).astype(str)
        + "_"
        + pd.to_datetime(result["time"], unit="s").dt.strftime("%d%m%y")
    )
    return result.drop(columns=["_traj_gap", "_traj_num"])


def add_along_track_distance(
    frame: pd.DataFrame,
    distance_fn: DistanceFn = haversine,
) -> pd.DataFrame:
    result = frame.copy()
    result["dist"] = 0.0

    for _, trajectory in result.groupby("ref", sort=False):
        previous_pt = None
        previous_dist = 0.0
        for row in trajectory.itertuples():
            current_pt = (float(row.lat), float(row.lon))
            if previous_pt is not None:
                previous_dist += distance_fn(previous_pt, current_pt)
                result.loc[row.Index, "dist"] = previous_dist
            previous_pt = current_pt

    return result

