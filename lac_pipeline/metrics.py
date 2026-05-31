from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TableMetrics:
    rows: int
    unique_aircraft: int


METRIC_TABLES = ("main_data", "inf_data", "gndinf_data")


def collect_pipeline_metrics(
    main_df: pd.DataFrame,
    inf_df: pd.DataFrame,
    gndinf_df: pd.DataFrame,
) -> dict[str, TableMetrics]:
    return {
        "main_data": table_metrics(main_df),
        "inf_data": table_metrics(inf_df),
        "gndinf_data": table_metrics(gndinf_df),
    }


def empty_pipeline_metrics() -> dict[str, TableMetrics]:
    return {name: TableMetrics(rows=0, unique_aircraft=0) for name in METRIC_TABLES}


def table_metrics(frame: pd.DataFrame) -> TableMetrics:
    return TableMetrics(
        rows=len(frame),
        unique_aircraft=_unique_aircraft_count(frame),
    )


def write_metrics_json(
    path: Path,
    processed_date: date,
    metrics: dict[str, TableMetrics],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_date": processed_date.isoformat(),
        "metrics": metrics_to_jsonable(metrics),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_metrics_json(path: Path) -> tuple[date, dict[str, TableMetrics]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        date.fromisoformat(payload["processed_date"]),
        metrics_from_jsonable(payload["metrics"]),
    )


def metrics_to_jsonable(metrics: dict[str, TableMetrics]) -> dict[str, dict[str, int]]:
    return {
        table_name: {
            "rows": table_metrics.rows,
            "unique_aircraft": table_metrics.unique_aircraft,
        }
        for table_name, table_metrics in metrics.items()
    }


def metrics_from_jsonable(payload: dict[str, dict[str, int]]) -> dict[str, TableMetrics]:
    return {
        table_name: TableMetrics(
            rows=int(values["rows"]),
            unique_aircraft=int(values["unique_aircraft"]),
        )
        for table_name, values in payload.items()
    }


def _unique_aircraft_count(frame: pd.DataFrame) -> int:
    if frame.empty or "icao24" not in frame.columns:
        return 0
    return int(frame["icao24"].nunique(dropna=True))
