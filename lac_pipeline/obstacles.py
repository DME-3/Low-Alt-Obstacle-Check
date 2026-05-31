from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClearanceConfig:
    congested_alert_distance_m: float
    congested_alert_delta_height_m: float
    noncongested_alert_distance_m: float
    noncongested_alert_delta_height_m: float
    geoid_height_m: float
    congested_population_threshold: float = 2.0


def load_obstacles(path: str | Path) -> pd.DataFrame:
    obstacles = pd.read_csv(path)
    obstacles = obstacles.rename(columns={"h": "height_m"})
    return obstacles.sort_values(by=["height_m"])


def add_obstacle_clearance(
    frame: pd.DataFrame,
    obstacles_df: pd.DataFrame,
    config: ClearanceConfig,
) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    result["closest_obst_name"] = ""
    result["inf_flt"] = False
    result["inf_pt"] = False
    result["gnd_inf_flt"] = False
    result["gnd_inf_pt"] = False
    result["min_hgt"] = np.nan
    result["congested"] = result["pop_density"] > config.congested_population_threshold

    final_coords = result[["etrs89_x", "etrs89_y"]].to_numpy()
    obstacle_coords = obstacles_df[["x", "y"]].to_numpy()
    obstacle_heights = obstacles_df["height_m"].to_numpy()
    obstacle_names = obstacles_df["LAC_Name"].to_numpy()
    obstacle_ground_elevs = obstacles_df["gnd_elev"].to_numpy()

    for i, (x_f, y_f) in enumerate(final_coords):
        radius = (
            config.congested_alert_distance_m
            if result.at[i, "congested"]
            else config.noncongested_alert_distance_m
        )
        distances_sq = (obstacle_coords[:, 0] - x_f) ** 2 + (
            obstacle_coords[:, 1] - y_f
        ) ** 2
        within_radius = distances_sq <= radius**2

        if np.any(within_radius):
            obstacle_idx = np.where(within_radius)[0]
            tallest_idx = obstacle_idx[np.argmax(obstacle_heights[obstacle_idx])]
            result.at[i, "closest_obst_name"] = obstacle_names[tallest_idx]
            result.at[i, "min_hgt"] = (
                config.geoid_height_m
                + np.float32(obstacle_ground_elevs[tallest_idx])
                + np.float32(obstacle_heights[tallest_idx])
                + _delta_height(result.at[i, "congested"], config)
            )
        else:
            result.at[i, "closest_obst_name"] = "ground"
            result.at[i, "min_hgt"] = (
                config.geoid_height_m
                + result.at[i, "gnd_elev"]
                + _delta_height(result.at[i, "congested"], config)
            )

    result["dip"] = result["min_hgt"] - result["geoaltitude"]
    result["inf_pt"] = (result["dip"] > 0) & (result["closest_obst_name"] != "ground")
    result["gnd_inf_pt"] = (result["dip"] > 0) & (
        result["closest_obst_name"] == "ground"
    )
    result["inf_flt"] = result.groupby("ref")["inf_pt"].transform("any")
    result["gnd_inf_flt"] = result.groupby("ref")["gnd_inf_pt"].transform("any")
    return result


def _delta_height(congested: bool, config: ClearanceConfig) -> float:
    if congested:
        return config.congested_alert_delta_height_m
    return config.noncongested_alert_delta_height_m

