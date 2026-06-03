import logging
from datetime import datetime

import pandas as pd
import pytest
from trino import exceptions as trino_exceptions

from lac_pipeline.opensky import (
    GeographicBounds,
    OpenSkyUnavailableError,
    build_operational_status_query,
    build_position_query,
    build_query_window,
    build_state_vectors_query,
    fetch_opensky_dataframe,
    format_icao_filter_values,
)


def test_build_state_vectors_query_preserves_bounds_and_window():
    window = build_query_window(datetime(2026, 5, 29, 12, 30))
    bounds = GeographicBounds(
        lat_min=50.88,
        lat_max=50.98,
        lon_min=6.85,
        lon_max=7.005,
    )

    query = build_state_vectors_query(window, bounds, 0, 750)

    assert "FROM state_vectors_data4" in query
    assert f"time >= {window.start_time}" in query
    assert f"hour <= {window.end_hour}" in query
    assert "lat >= 50.88 AND lat <= 50.98" in query
    assert "lon>= 6.85 AND lon <= 7.005" in query
    assert "geoaltitude >= 0 AND geoaltitude <= 750" in query


def test_build_related_queries_use_validated_icao_filter():
    window = build_query_window(datetime(2026, 5, 29))
    bounds = GeographicBounds(
        lat_min=50.88,
        lat_max=50.98,
        lon_min=6.85,
        lon_max=7.005,
    )

    ops_query = build_operational_status_query(["3c6444", "ABCDEF"], window)
    pos_query = build_position_query(["3c6444"], window, bounds)

    assert "icao24 IN ('3c6444', 'ABCDEF')" in ops_query
    assert "FROM operational_status_data4" in ops_query
    assert "icao24 IN ('3c6444')" in pos_query
    assert "FROM position_data4" in pos_query


def test_format_icao_filter_values_rejects_unsafe_values():
    with pytest.raises(ValueError, match="invalid ICAO24"):
        format_icao_filter_values(["3c6444'; DROP TABLE manifest; --"])


def test_fetch_opensky_dataframe_uses_bounded_retry():
    calls = {"count": 0}
    expected = pd.DataFrame({"icao24": ["3c6444"]})

    class FakeTrino:
        def query(self, query, cached, compress):
            calls["count"] += 1
            assert query == "SELECT 1"
            assert cached is False
            assert compress is True
            if calls["count"] == 1:
                raise RuntimeError("transient")
            return expected

    result = fetch_opensky_dataframe(
        "state_vectors_data4",
        "SELECT 1",
        attempts=2,
        retry_delay_seconds=0,
        logger=logging.getLogger("test"),
        cached=False,
        compress=True,
        trino_factory=FakeTrino,
    )

    assert result.equals(expected)
    assert calls["count"] == 2


def test_fetch_opensky_dataframe_turns_repeated_trino_404_into_unavailable(caplog):
    calls = {"count": 0}

    class FakeTrino:
        def query(self, query, cached, compress):
            calls["count"] += 1
            raise trino_exceptions.HttpError(
                "error 404: b'<html><head><title>404 Not Found</title></head></html>'"
            )

    caplog.set_level(logging.WARNING)

    with pytest.raises(OpenSkyUnavailableError) as error:
        fetch_opensky_dataframe(
            "state_vectors_data4",
            "SELECT 1",
            attempts=2,
            retry_delay_seconds=0,
            logger=logging.getLogger("test"),
            trino_factory=FakeTrino,
        )

    assert calls["count"] == 2
    assert "table=state_vectors_data4" in str(error.value)
    assert 'last_error="HTTP 404 Not Found"' in str(error.value)
    assert "HTTP 404 Not Found" in caplog.text
    assert "<html>" not in caplog.text


def test_fetch_opensky_dataframe_leaves_non_trino_404_errors_alone():
    class FakeTrino:
        def query(self, query, cached, compress):
            raise RuntimeError("error 404: application lookup")

    with pytest.raises(RuntimeError, match="application lookup"):
        fetch_opensky_dataframe(
            "state_vectors_data4",
            "SELECT 1",
            attempts=1,
            retry_delay_seconds=0,
            logger=logging.getLogger("test"),
            trino_factory=FakeTrino,
        )
