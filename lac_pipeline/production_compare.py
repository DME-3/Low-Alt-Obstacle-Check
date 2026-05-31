from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone

from sqlalchemy import text
from sqlalchemy.engine import Connection

from lac_pipeline.metrics import METRIC_TABLES, TableMetrics
from lac_pipeline.opensky import build_query_window
from lac_pipeline.publishing import PublishTarget

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class MetricComparison:
    table_name: str
    metric_name: str
    dry_run_value: int
    production_value: int

    @property
    def difference(self) -> int:
        return self.dry_run_value - self.production_value


def fetch_production_metrics(
    connection: Connection,
    target: PublishTarget,
    processed_date: date,
) -> dict[str, TableMetrics]:
    window = build_query_window(datetime.combine(processed_date, time.min))
    start_dt = datetime.fromtimestamp(window.start_time, tz=timezone.utc).replace(
        tzinfo=None
    )
    end_dt = datetime.fromtimestamp(window.end_time, tz=timezone.utc).replace(
        tzinfo=None
    )
    return {
        "main_data": fetch_table_metrics(
            connection,
            target.main_table,
            start_value=window.start_time,
            end_value=window.end_time,
        ),
        "inf_data": fetch_table_metrics(
            connection,
            target.inf_table,
            start_value=start_dt,
            end_value=end_dt,
        ),
        "gndinf_data": fetch_table_metrics(
            connection,
            target.gndinf_table,
            start_value=start_dt,
            end_value=end_dt,
        ),
    }


def fetch_table_metrics(
    connection: Connection,
    table_name: str,
    start_value: int | datetime,
    end_value: int | datetime,
) -> TableMetrics:
    result = connection.execute(
        text(
            f"SELECT COUNT(*) AS rows, COUNT(DISTINCT icao24) AS unique_aircraft "
            f"FROM `{_checked_identifier(table_name)}` "
            "WHERE `time` >= :start_value AND `time` <= :end_value"
        ),
        {"start_value": start_value, "end_value": end_value},
    ).mappings().one()
    return TableMetrics(
        rows=int(result["rows"]),
        unique_aircraft=int(result["unique_aircraft"]),
    )


def _checked_identifier(name: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"unsafe SQL identifier: {name}")
    return name


def compare_metrics(
    dry_run_metrics: dict[str, TableMetrics],
    production_metrics: dict[str, TableMetrics],
) -> list[MetricComparison]:
    comparisons: list[MetricComparison] = []
    for table_name in METRIC_TABLES:
        dry = dry_run_metrics[table_name]
        prod = production_metrics[table_name]
        comparisons.extend(
            [
                MetricComparison(table_name, "rows", dry.rows, prod.rows),
                MetricComparison(
                    table_name,
                    "unique_aircraft",
                    dry.unique_aircraft,
                    prod.unique_aircraft,
                ),
            ]
        )
    return comparisons


def format_comparison_report(
    processed_date: date,
    comparisons: list[MetricComparison],
) -> str:
    lines = [
        f"Dry-run vs production comparison for {processed_date.isoformat()}",
        "",
        f"{'table':<12} {'metric':<16} {'dry_run':>10} {'production':>12} {'diff':>10}",
        "-" * 64,
    ]
    for comparison in comparisons:
        lines.append(
            f"{comparison.table_name:<12} "
            f"{comparison.metric_name:<16} "
            f"{comparison.dry_run_value:>10} "
            f"{comparison.production_value:>12} "
            f"{comparison.difference:>+10}"
        )
    return "\n".join(lines)
