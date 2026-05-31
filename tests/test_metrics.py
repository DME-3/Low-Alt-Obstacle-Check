from datetime import date

import pandas as pd

from lac_pipeline.metrics import (
    collect_pipeline_metrics,
    read_metrics_json,
    write_metrics_json,
)


def test_collect_pipeline_metrics_counts_rows_and_unique_aircraft():
    main_df = pd.DataFrame({"icao24": ["abc123", "abc123", "def456"]})
    inf_df = pd.DataFrame({"icao24": ["abc123", "ghi789"]})
    gndinf_df = pd.DataFrame({"icao24": ["abc123", None]})

    metrics = collect_pipeline_metrics(main_df, inf_df, gndinf_df)

    assert metrics["main_data"].rows == 3
    assert metrics["main_data"].unique_aircraft == 2
    assert metrics["inf_data"].rows == 2
    assert metrics["inf_data"].unique_aircraft == 2
    assert metrics["gndinf_data"].rows == 2
    assert metrics["gndinf_data"].unique_aircraft == 1


def test_metrics_json_round_trip(tmp_path):
    path = tmp_path / "metrics.json"
    metrics = collect_pipeline_metrics(
        pd.DataFrame({"icao24": ["abc123"]}),
        pd.DataFrame({"icao24": []}),
        pd.DataFrame({"icao24": ["def456"]}),
    )

    write_metrics_json(path, date(2026, 5, 29), metrics)
    metrics_date, loaded = read_metrics_json(path)

    assert metrics_date == date(2026, 5, 29)
    assert loaded == metrics
