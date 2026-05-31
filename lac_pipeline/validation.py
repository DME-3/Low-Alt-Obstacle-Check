from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd


class ValidationError(ValueError):
    """Raised when pipeline output is not safe to publish."""


@dataclass(frozen=True)
class FrameValidationResult:
    name: str
    row_count: int
    null_counts: dict[str, int]


MAIN_REQUIRED_COLUMNS = {
    "icao24",
    "callsign",
    "time",
    "lat",
    "lon",
    "geoaltitude",
    "gnd_elev",
    "pop_density",
    "ref",
    "dip",
    "inf_pt",
    "gnd_inf_pt",
}

EVENT_REQUIRED_COLUMNS = {
    "icao24",
    "callsign",
    "ref",
    "closest_obst_name",
    "time",
    "lat",
    "lon",
    "dip_max",
    "congested",
    "pop_density",
    "n",
    "inf_ref",
    "url",
}


def validate_required_columns(
    frame: pd.DataFrame, required_columns: Iterable[str], name: str
) -> None:
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValidationError(f"{name} is missing required columns: {missing}")


def validate_no_duplicate_keys(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame.empty:
        return
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValidationError(f"{name} duplicate-key check missing columns: {missing}")
    duplicated = frame.duplicated(subset=keys).sum()
    if duplicated:
        raise ValidationError(f"{name} has {duplicated} duplicate rows for keys {keys}")


def validate_pipeline_outputs(
    main_df: pd.DataFrame,
    inf_df: pd.DataFrame,
    gndinf_df: pd.DataFrame,
    processed_date: date,
) -> list[FrameValidationResult]:
    validate_required_columns(main_df, MAIN_REQUIRED_COLUMNS, "main_df")
    validate_required_columns(inf_df, EVENT_REQUIRED_COLUMNS, "inf_df")
    validate_required_columns(gndinf_df, EVENT_REQUIRED_COLUMNS, "gndinf_df")

    _validate_main_date(main_df, processed_date)
    _validate_event_date(inf_df, processed_date, "inf_df")
    _validate_event_date(gndinf_df, processed_date, "gndinf_df")

    validate_no_duplicate_keys(inf_df, ["inf_ref"], "inf_df")
    validate_no_duplicate_keys(gndinf_df, ["inf_ref"], "gndinf_df")

    return [
        _result("main_df", main_df, ["icao24", "callsign", "time", "lat", "lon"]),
        _result("inf_df", inf_df, ["icao24", "callsign", "time", "inf_ref"]),
        _result("gndinf_df", gndinf_df, ["icao24", "callsign", "time", "inf_ref"]),
    ]


def _validate_main_date(frame: pd.DataFrame, processed_date: date) -> None:
    if frame.empty:
        return
    times = pd.to_datetime(frame["time"], unit="s", errors="coerce")
    if times.isna().any():
        raise ValidationError("main_df contains invalid unix timestamps")
    _validate_dates_match(times.dt.date, processed_date, "main_df")


def _validate_event_date(frame: pd.DataFrame, processed_date: date, name: str) -> None:
    if frame.empty:
        return
    times = pd.to_datetime(frame["time"], errors="coerce")
    if times.isna().any():
        raise ValidationError(f"{name} contains invalid timestamps")
    _validate_dates_match(times.dt.date, processed_date, name)


def _validate_dates_match(actual_dates: pd.Series, processed_date: date, name: str) -> None:
    unexpected_dates = sorted({value for value in actual_dates.unique() if value != processed_date})
    if unexpected_dates:
        raise ValidationError(
            f"{name} contains rows outside processed_date={processed_date}: {unexpected_dates}"
        )


def _result(
    name: str, frame: pd.DataFrame, important_columns: list[str]
) -> FrameValidationResult:
    present = [column for column in important_columns if column in frame.columns]
    return FrameValidationResult(
        name=name,
        row_count=len(frame),
        null_counts={column: int(frame[column].isna().sum()) for column in present},
    )

