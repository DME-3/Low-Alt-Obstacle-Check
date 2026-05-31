from datetime import datetime

import pandas as pd

from lac_pipeline.events import (
    GROUND_EVENT_COLUMNS,
    OBSTACLE_EVENT_COLUMNS,
    build_event_tables,
)


def test_build_event_tables_returns_schema_for_empty_events():
    final_df = _base_final_df(inf_pt=False, gnd_inf_pt=False)
    obstacles_df = _obstacles_df()

    obstacle_events, ground_events = build_event_tables(
        final_df, obstacles_df, lambda left, right: 0.0
    )

    assert list(obstacle_events.columns) == OBSTACLE_EVENT_COLUMNS
    assert list(ground_events.columns) == GROUND_EVENT_COLUMNS
    assert obstacle_events.empty
    assert ground_events.empty


def test_build_event_tables_builds_obstacle_event():
    final_df = _base_final_df(
        inf_pt=True,
        gnd_inf_pt=False,
        closest_obst_name="Tower",
        dip=42.0,
    )
    obstacles_df = _obstacles_df()

    obstacle_events, ground_events = build_event_tables(
        final_df, obstacles_df, lambda left, right: 123.4
    )

    assert len(obstacle_events) == 1
    assert ground_events.empty
    event = obstacle_events.iloc[0]
    assert event["closest_obst_name"] == "Tower"
    assert event["cpa"] == 123.4
    assert event["dip_max"] == 42.0
    assert event["n"] == 1
    assert event["inf_ref"] == "abc123_1_290526_0"
    assert "icao=abc123" in event["url"]


def test_build_event_tables_builds_ground_event_with_max_dip():
    base = _base_final_df(inf_pt=False, gnd_inf_pt=True, dip=5.0)
    later = _base_final_df(inf_pt=False, gnd_inf_pt=True, dip=20.0, offset_seconds=10)
    final_df = pd.concat([base, later], ignore_index=True)

    obstacle_events, ground_events = build_event_tables(
        final_df, _obstacles_df(), lambda left, right: 0.0
    )

    assert obstacle_events.empty
    assert len(ground_events) == 1
    event = ground_events.iloc[0]
    assert event["dip_max"] == 20.0
    assert event["n"] == 2
    assert event["inf_ref"] == "abc123_1_290526_gnd_0"


def _base_final_df(
    inf_pt,
    gnd_inf_pt,
    closest_obst_name="ground",
    dip=10.0,
    offset_seconds=0,
):
    timestamp = int(datetime(2026, 5, 29, 12, 0, offset_seconds).timestamp())
    return pd.DataFrame(
        {
            "icao24": ["abc123"],
            "callsign": ["TEST1"],
            "ref": ["abc123_1_290526"],
            "closest_obst_name": [closest_obst_name],
            "time": [timestamp],
            "lat": [50.9],
            "lon": [6.9],
            "dip": [dip],
            "congested": [True],
            "pop_density": [3.0],
            "systemdesignassurance": [2],
            "version": [2],
            "positionnac": [9],
            "sourceintegritylevel": [3],
            "nic": [8],
            "inf_pt": [inf_pt],
            "gnd_inf_pt": [gnd_inf_pt],
        }
    )


def _obstacles_df():
    return pd.DataFrame(
        {
            "LAC_Name": ["Tower"],
            "lat": [50.91],
            "lon": [6.91],
        }
    )

