from datetime import date, datetime

import pytest
from sqlalchemy import create_engine, text

from lac_pipeline.metrics import TableMetrics
from lac_pipeline.production_compare import (
    compare_metrics,
    fetch_production_metrics,
    format_comparison_report,
    table_metrics_query,
)
from lac_pipeline.publishing import PublishTarget


def test_fetch_production_metrics_counts_rows_and_unique_aircraft():
    engine = create_engine("sqlite:///:memory:")
    target = PublishTarget(
        name="prod",
        database_name="ignored",
        main_table="main_data",
        inf_table="inf_data",
        gndinf_table="gndinf_data",
    )
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE main_data (time INTEGER, icao24 TEXT)"))
        connection.execute(text("CREATE TABLE inf_data (time DATETIME, icao24 TEXT)"))
        connection.execute(text("CREATE TABLE gndinf_data (time DATETIME, icao24 TEXT)"))
        connection.execute(
            text(
                "INSERT INTO main_data (time, icao24) VALUES "
                "(1780005600, 'abc123'), (1780005601, 'abc123'), "
                "(1780091999, 'def456'), (1780092000, 'outside')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO inf_data (time, icao24) VALUES "
                "(:inside, 'abc123'), (:outside, 'outside')"
            ),
            {
                "inside": datetime(2026, 5, 28, 22, 0, 0),
                "outside": datetime(2026, 5, 29, 22, 0, 0),
            },
        )
        connection.execute(
            text("INSERT INTO gndinf_data (time, icao24) VALUES (:inside, 'def456')"),
            {"inside": datetime(2026, 5, 29, 21, 59, 59)},
        )
        metrics = fetch_production_metrics(connection, target, date(2026, 5, 29))

    assert metrics["main_data"] == TableMetrics(rows=3, unique_aircraft=2)
    assert metrics["inf_data"] == TableMetrics(rows=1, unique_aircraft=1)
    assert metrics["gndinf_data"] == TableMetrics(rows=1, unique_aircraft=1)


def test_table_metrics_query_avoids_reserved_rows_alias():
    query = table_metrics_query("main_data")

    assert "AS row_count" in query
    assert "AS rows" not in query


def test_fetch_table_metrics_rejects_unsafe_table_name():
    from lac_pipeline.production_compare import fetch_table_metrics

    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as connection:
        with pytest.raises(ValueError):
            fetch_table_metrics(connection, "main_data;drop", 0, 1)


def test_format_comparison_report_shows_differences():
    dry = {
        "main_data": TableMetrics(rows=4, unique_aircraft=2),
        "inf_data": TableMetrics(rows=3, unique_aircraft=2),
        "gndinf_data": TableMetrics(rows=1, unique_aircraft=1),
    }
    prod = {
        "main_data": TableMetrics(rows=2, unique_aircraft=2),
        "inf_data": TableMetrics(rows=4, unique_aircraft=3),
        "gndinf_data": TableMetrics(rows=1, unique_aircraft=1),
    }

    report = format_comparison_report(date(2026, 5, 29), compare_metrics(dry, prod))

    assert "main_data" in report
    assert "unique_aircraft" in report
    assert "+2" in report
    assert "-1" in report
