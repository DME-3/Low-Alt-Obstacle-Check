import pandas as pd

from lac_pipeline.obstacles import ClearanceConfig, add_obstacle_clearance


def test_add_obstacle_clearance_selects_tallest_obstacle_in_radius():
    frame = _frame(x=0.0, y=0.0, pop_density=3.0, gnd_elev=40.0, geoaltitude=200.0)
    obstacles = pd.DataFrame(
        {
            "x": [10.0, 20.0],
            "y": [0.0, 0.0],
            "height_m": [20.0, 80.0],
            "LAC_Name": ["Short", "Tall"],
            "gnd_elev": [45.0, 50.0],
        }
    )

    result = add_obstacle_clearance(frame, obstacles, _config())

    assert result["closest_obst_name"].iloc[0] == "Tall"
    assert result["min_hgt"].iloc[0] == 47 + 50 + 80 + 300
    assert result["dip"].iloc[0] == 277
    assert result["inf_pt"].iloc[0]
    assert not result["gnd_inf_pt"].iloc[0]


def test_add_obstacle_clearance_falls_back_to_ground():
    frame = _frame(x=999.0, y=999.0, pop_density=1.0, gnd_elev=40.0, geoaltitude=200.0)
    obstacles = pd.DataFrame(
        {
            "x": [0.0],
            "y": [0.0],
            "height_m": [80.0],
            "LAC_Name": ["Tower"],
            "gnd_elev": [50.0],
        }
    )

    result = add_obstacle_clearance(frame, obstacles, _config())

    assert result["closest_obst_name"].iloc[0] == "ground"
    assert result["min_hgt"].iloc[0] == 47 + 40 + 150
    assert result["gnd_inf_pt"].iloc[0]
    assert not result["inf_pt"].iloc[0]


def _frame(x, y, pop_density, gnd_elev, geoaltitude):
    return pd.DataFrame(
        {
            "etrs89_x": [x],
            "etrs89_y": [y],
            "pop_density": [pop_density],
            "gnd_elev": [gnd_elev],
            "geoaltitude": [geoaltitude],
            "ref": ["abc_1_290526"],
        }
    )


def _config():
    return ClearanceConfig(
        congested_alert_distance_m=600,
        congested_alert_delta_height_m=300,
        noncongested_alert_distance_m=150,
        noncongested_alert_delta_height_m=150,
        geoid_height_m=47,
    )

