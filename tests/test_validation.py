from datetime import date, datetime

import pandas as pd
import pytest

from lac_pipeline.validation import ValidationError, validate_pipeline_outputs


def test_validate_pipeline_outputs_accepts_required_columns():
    processed_date = date(2026, 5, 29)
    main_df = _main_df(int(datetime(2026, 5, 29, 12, 0, 0).timestamp()))
    inf_df = _event_df(datetime(2026, 5, 29, 12, 0, 0), "abc_0")
    gndinf_df = _event_df(datetime(2026, 5, 29, 13, 0, 0), "abc_gnd_0")

    results = validate_pipeline_outputs(main_df, inf_df, gndinf_df, processed_date)

    assert [result.row_count for result in results] == [1, 1, 1]


def test_validate_pipeline_outputs_rejects_wrong_main_date():
    processed_date = date(2026, 5, 29)
    main_df = _main_df(int(datetime(2026, 5, 30, 12, 0, 0).timestamp()))
    inf_df = _event_df(datetime(2026, 5, 29, 12, 0, 0), "abc_0")
    gndinf_df = _event_df(datetime(2026, 5, 29, 13, 0, 0), "abc_gnd_0")

    with pytest.raises(ValidationError):
        validate_pipeline_outputs(main_df, inf_df, gndinf_df, processed_date)


def test_validate_pipeline_outputs_rejects_duplicate_event_refs():
    processed_date = date(2026, 5, 29)
    main_df = _main_df(int(datetime(2026, 5, 29, 12, 0, 0).timestamp()))
    inf_df = pd.concat(
        [
            _event_df(datetime(2026, 5, 29, 12, 0, 0), "abc_0"),
            _event_df(datetime(2026, 5, 29, 12, 5, 0), "abc_0"),
        ],
        ignore_index=True,
    )
    gndinf_df = _event_df(datetime(2026, 5, 29, 13, 0, 0), "abc_gnd_0")

    with pytest.raises(ValidationError):
        validate_pipeline_outputs(main_df, inf_df, gndinf_df, processed_date)


def _main_df(unix_time):
    return pd.DataFrame(
        {
            "icao24": ["abc123"],
            "callsign": ["TEST1"],
            "time": [unix_time],
            "lat": [50.9],
            "lon": [6.9],
            "geoaltitude": [200.0],
            "gnd_elev": [50.0],
            "pop_density": [3.0],
            "ref": ["abc123_1_290526"],
            "dip": [10.0],
            "inf_pt": [False],
            "gnd_inf_pt": [True],
        }
    )


def _event_df(timestamp, inf_ref):
    return pd.DataFrame(
        {
            "icao24": ["abc123"],
            "callsign": ["TEST1"],
            "ref": ["abc123_1_290526"],
            "closest_obst_name": ["ground"],
            "time": [timestamp],
            "lat": [50.9],
            "lon": [6.9],
            "dip_max": [10.0],
            "congested": [True],
            "pop_density": [3.0],
            "n": [2],
            "inf_ref": [inf_ref],
            "url": ["https://example.invalid"],
        }
    )

