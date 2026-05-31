from datetime import datetime

import pandas as pd

from lac_pipeline.trajectory import add_trajectory_columns, haversine


def test_add_trajectory_columns_splits_per_icao_and_accumulates_distance():
    base = int(datetime(2026, 5, 29, 12, 0, 0).timestamp())
    frame = pd.DataFrame(
        {
            "icao24": ["aaa", "bbb", "aaa", "aaa", "bbb"],
            "time": [base, base, base + 10, base + 50, base + 10],
            "lat": [50.0, 51.0, 50.0, 50.0, 51.0],
            "lon": [6.0, 7.0, 6.1, 6.2, 7.1],
        }
    ).sort_values(["icao24", "time"])

    result = add_trajectory_columns(frame, time_between_trajs=30, distance_fn=_flat_distance)

    aaa = result[result["icao24"] == "aaa"]
    bbb = result[result["icao24"] == "bbb"]
    assert aaa["ref"].tolist() == [
        "aaa_1_290526",
        "aaa_1_290526",
        "aaa_2_290526",
    ]
    assert bbb["ref"].tolist() == ["bbb_1_290526", "bbb_1_290526"]
    assert aaa["dist"].tolist() == [0.0, 10.0, 0.0]
    assert bbb["dist"].tolist() == [0.0, 10.0]


def test_haversine_returns_zero_for_same_point():
    assert haversine((50.0, 6.0), (50.0, 6.0)) == 0


def _flat_distance(left, right):
    return round(abs(right[1] - left[1]) * 100)

