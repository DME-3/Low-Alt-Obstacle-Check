from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from pyopensky.trino import Trino

from lac_pipeline.runtime import retry


@dataclass(frozen=True)
class GeographicBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class QueryWindow:
    start_time: int
    end_time: int
    start_hour: int
    end_hour: int


_ICAO24_RE = re.compile(r"^[0-9A-Fa-f]{6}$")


def build_query_window(target_day: datetime) -> QueryWindow:
    start = target_day.replace(hour=0, minute=0, second=0, microsecond=0)
    end = target_day.replace(hour=23, minute=59, second=59, microsecond=999999)
    start_time = int(start.timestamp())
    end_time = int(end.timestamp())
    return QueryWindow(
        start_time=start_time,
        end_time=end_time,
        start_hour=start_time - (start_time % 3600),
        end_hour=end_time - (end_time % 3600),
    )


def build_state_vectors_query(
    window: QueryWindow,
    bounds: GeographicBounds,
    altitude_min_m: int,
    altitude_max_m: int,
) -> str:
    return (
        f"SELECT * FROM state_vectors_data4"
        f" WHERE icao24 LIKE '%'"
        f" AND time >= {window.start_time} AND time <= {window.end_time}"
        f" AND hour >= {window.start_hour} AND hour <= {window.end_hour}"
        f" AND lat >= {bounds.lat_min} AND lat <= {bounds.lat_max}"
        f" AND lon>= {bounds.lon_min} AND lon <= {bounds.lon_max}"
        f" AND geoaltitude >= {altitude_min_m} AND geoaltitude <= {altitude_max_m}"
        f" ORDER BY time"
    )


def build_operational_status_query(
    icao_values: Iterable[object],
    window: QueryWindow,
) -> str:
    return (
        "SELECT icao24, mintime, maxtime, nacv, systemdesignassurance, version, "
        "positionnac, geometricverticalaccuracy, sourceintegritylevel, "
        "barometricaltitudeintegritycode FROM operational_status_data4"
        f" WHERE icao24 IN ({format_icao_filter_values(icao_values)})"
        f" AND mintime >= {window.start_time} AND maxtime <= {window.end_time}"
        f" AND hour >= {window.start_hour} AND hour <= {window.end_hour}"
        f" ORDER by mintime"
    )


def build_position_query(
    icao_values: Iterable[object],
    window: QueryWindow,
    bounds: GeographicBounds,
) -> str:
    return (
        f"SELECT mintime, icao24, nic  FROM position_data4"
        f" WHERE icao24 IN ({format_icao_filter_values(icao_values)})"
        f" AND lat >= {bounds.lat_min} AND lat <= {bounds.lat_max}"
        f" AND lon>= {bounds.lon_min} AND lon <= {bounds.lon_max}"
        f" AND mintime >= {window.start_time} AND maxtime <= {window.end_time}"
        f" AND hour >= {window.start_hour} AND hour <= {window.end_hour}"
        f" ORDER by mintime"
    )


def format_icao_filter_values(icao_values: Iterable[object]) -> str:
    cleaned = [str(value).strip() for value in icao_values]
    if not cleaned:
        raise ValueError("at least one ICAO24 value is required")
    invalid = [value for value in cleaned if not _ICAO24_RE.fullmatch(value)]
    if invalid:
        raise ValueError(f"invalid ICAO24 value(s): {invalid}")
    return ", ".join(f"'{value}'" for value in cleaned)


def fetch_opensky_dataframe(
    label: str,
    query: str,
    attempts: int,
    retry_delay_seconds: int,
    logger: logging.Logger,
    *,
    cached: bool = False,
    compress: bool = False,
    trino_factory: Callable[[], Trino] = Trino,
) -> pd.DataFrame:
    return retry(
        label,
        attempts,
        retry_delay_seconds,
        logger,
        lambda: trino_factory().query(query, cached=cached, compress=compress),
    )
