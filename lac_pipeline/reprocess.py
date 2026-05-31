from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone

from sqlalchemy import text
from sqlalchemy.engine import Connection

from lac_pipeline.metrics import TableMetrics
from lac_pipeline.opensky import build_query_window
from lac_pipeline.production_compare import fetch_table_metrics
from lac_pipeline.publishing import PublishTarget

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class DayWindow:
    main_start: int
    main_end: int
    event_start: datetime
    event_end: datetime


@dataclass(frozen=True)
class ProductionDayPlan:
    logical_name: str
    table_name: str
    metrics: TableMetrics
    manifest_rows: int


@dataclass(frozen=True)
class ProductionDeleteResult:
    logical_name: str
    table_name: str
    deleted_rows: int
    deleted_manifest_rows: int


def production_day_window(processed_date: date) -> DayWindow:
    window = build_query_window(datetime.combine(processed_date, time.min))
    return DayWindow(
        main_start=window.start_time,
        main_end=window.end_time,
        event_start=datetime.fromtimestamp(window.start_time, tz=timezone.utc).replace(
            tzinfo=None
        ),
        event_end=datetime.fromtimestamp(window.end_time, tz=timezone.utc).replace(
            tzinfo=None
        ),
    )


def plan_production_day_replacement(
    connection: Connection,
    target: PublishTarget,
    processed_date: date,
) -> list[ProductionDayPlan]:
    day_window = production_day_window(processed_date)
    return [
        ProductionDayPlan(
            logical_name="main_data",
            table_name=target.main_table,
            metrics=fetch_table_metrics(
                connection,
                target.main_table,
                day_window.main_start,
                day_window.main_end,
            ),
            manifest_rows=_manifest_row_count(
                connection, target.main_table, processed_date
            ),
        ),
        ProductionDayPlan(
            logical_name="inf_data",
            table_name=target.inf_table,
            metrics=fetch_table_metrics(
                connection,
                target.inf_table,
                day_window.event_start,
                day_window.event_end,
            ),
            manifest_rows=_manifest_row_count(connection, target.inf_table, processed_date),
        ),
        ProductionDayPlan(
            logical_name="gndinf_data",
            table_name=target.gndinf_table,
            metrics=fetch_table_metrics(
                connection,
                target.gndinf_table,
                day_window.event_start,
                day_window.event_end,
            ),
            manifest_rows=_manifest_row_count(
                connection, target.gndinf_table, processed_date
            ),
        ),
    ]


def delete_production_day(
    connection: Connection,
    target: PublishTarget,
    processed_date: date,
) -> list[ProductionDeleteResult]:
    day_window = production_day_window(processed_date)
    specs = [
        ("main_data", target.main_table, day_window.main_start, day_window.main_end),
        ("inf_data", target.inf_table, day_window.event_start, day_window.event_end),
        ("gndinf_data", target.gndinf_table, day_window.event_start, day_window.event_end),
    ]
    results: list[ProductionDeleteResult] = []
    for logical_name, table_name, start_value, end_value in specs:
        deleted_rows = connection.execute(
            text(
                f"DELETE FROM `{_checked_identifier(table_name)}` "
                "WHERE `time` >= :start_value AND `time` <= :end_value"
            ),
            {"start_value": start_value, "end_value": end_value},
        ).rowcount
        deleted_manifest_rows = connection.execute(
            text(
                "DELETE FROM manifest "
                "WHERE processed_date = :processed_date AND table_name = :table_name"
            ),
            {"processed_date": processed_date, "table_name": table_name},
        ).rowcount
        results.append(
            ProductionDeleteResult(
                logical_name=logical_name,
                table_name=table_name,
                deleted_rows=int(deleted_rows or 0),
                deleted_manifest_rows=int(deleted_manifest_rows or 0),
            )
        )
    return results


def format_replacement_plan(processed_date: date, plan: list[ProductionDayPlan]) -> str:
    lines = [
        f"Production replacement plan for {processed_date.isoformat()}",
        "",
        f"{'table':<12} {'prod_table':<18} {'rows':>10} "
        f"{'aircraft':>10} {'manifest':>10}",
        "-" * 66,
    ]
    for item in plan:
        lines.append(
            f"{item.logical_name:<12} {item.table_name:<18} "
            f"{item.metrics.rows:>10} {item.metrics.unique_aircraft:>10} "
            f"{item.manifest_rows:>10}"
        )
    return "\n".join(lines)


def format_delete_results(
    processed_date: date, results: list[ProductionDeleteResult]
) -> str:
    lines = [
        f"Deleted production rows for {processed_date.isoformat()}",
        "",
        f"{'table':<12} {'prod_table':<18} {'rows':>10} {'manifest':>10}",
        "-" * 54,
    ]
    for item in results:
        lines.append(
            f"{item.logical_name:<12} {item.table_name:<18} "
            f"{item.deleted_rows:>10} {item.deleted_manifest_rows:>10}"
        )
    return "\n".join(lines)


def _manifest_row_count(
    connection: Connection,
    table_name: str,
    processed_date: date,
) -> int:
    result = connection.execute(
        text(
            "SELECT COUNT(*) AS row_count FROM manifest "
            "WHERE processed_date = :processed_date AND table_name = :table_name"
        ),
        {"processed_date": processed_date, "table_name": table_name},
    )
    return int(result.scalar_one())


def _checked_identifier(name: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"unsafe SQL identifier: {name}")
    return name
