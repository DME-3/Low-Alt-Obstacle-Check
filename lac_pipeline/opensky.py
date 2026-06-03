from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus

import pandas as pd
from pyopensky.trino import Trino
from trino import exceptions as trino_exceptions

from lac_pipeline.runtime import GracefulPipelineError, retry


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


_HTTP_ERROR_RE = re.compile(r"\berror\s+(?P<status>\d{3})(?::|\b)", re.IGNORECASE)
_HTML_TITLE_RE = re.compile(r"<title>\s*(?P<title>[^<]+?)\s*</title>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_OPEN_SKY_UNAVAILABLE_HTTP_STATUSES = {404, 429, 500, 502, 503, 504}


class OpenSkyUnavailableError(GracefulPipelineError):
    """Raised when the OpenSky/Trino source is temporarily unavailable."""

    status = "opensky_unavailable"
    exit_code = 75

    def __init__(self, label: str, attempts: int, last_error: BaseException) -> None:
        self.label = label
        self.attempts = attempts
        self.last_error_summary = summarize_opensky_error(last_error)
        super().__init__(
            "OpenSky/Trino source unavailable "
            f"table={label} attempts={attempts} "
            f'last_error="{self.last_error_summary}"'
        )


def is_opensky_unavailable_error(exc: BaseException) -> bool:
    for candidate in _walk_exception_chain(exc):
        if isinstance(candidate, trino_exceptions.TrinoConnectionError):
            return True
        if isinstance(candidate, trino_exceptions.HttpError):
            status = _http_status_from_error(candidate)
            return status in _OPEN_SKY_UNAVAILABLE_HTTP_STATUSES
    return False


def summarize_opensky_error(exc: BaseException) -> str:
    for candidate in _walk_exception_chain(exc):
        if not isinstance(candidate, trino_exceptions.HttpError):
            continue
        status = _http_status_from_error(candidate)
        if status is not None:
            return _format_http_status(status, str(candidate))

    message = _clean_error_text(str(exc))
    return message or exc.__class__.__name__


def _walk_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        next_exc = getattr(current, "orig", None) or current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, BaseException) else None


def _http_status_from_error(exc: BaseException) -> int | None:
    match = _HTTP_ERROR_RE.search(str(exc))
    if not match:
        return None
    return int(match.group("status"))


def _format_http_status(status: int, message: str) -> str:
    reason = _http_reason_from_message(status, message)
    if reason:
        return f"HTTP {status} {reason}"
    return f"HTTP {status}"


def _http_reason_from_message(status: int, message: str) -> str:
    title_match = _HTML_TITLE_RE.search(message)
    if title_match:
        reason = _clean_error_text(title_match.group("title"))
        reason = re.sub(rf"^{status}\s*", "", reason).strip(" -:")
        if reason:
            return reason

    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return ""


def _clean_error_text(message: str) -> str:
    text = _HTML_TAG_RE.sub(" ", message)
    text = text.replace("\\r", " ").replace("\\n", " ")
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    if len(text) > 200:
        return f"{text[:197]}..."
    return text


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
    try:
        return retry(
            label,
            attempts,
            retry_delay_seconds,
            logger,
            lambda: trino_factory().query(query, cached=cached, compress=compress),
            error_formatter=summarize_opensky_error,
        )
    except Exception as exc:  # noqa: BLE001
        if is_opensky_unavailable_error(exc):
            raise OpenSkyUnavailableError(label, attempts, exc) from exc
        raise
