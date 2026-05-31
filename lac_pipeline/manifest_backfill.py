from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy.engine import Connection

from lac_pipeline.publishing import PublishTarget, insert_manifest, manifest_success_counts


@dataclass(frozen=True)
class ManifestBackfillAction:
    processed_date: date
    table_name: str
    action: str
    existing_success_count: int


def plan_manifest_backfill(
    connection: Connection,
    target: PublishTarget,
    start_date: date,
    end_date: date,
    force: bool = False,
) -> list[ManifestBackfillAction]:
    _validate_date_range(start_date, end_date)
    actions: list[ManifestBackfillAction] = []

    for processed_date in date_range(start_date, end_date):
        counts = manifest_success_counts(connection, processed_date, target.table_names)
        for table_name in target.table_names:
            existing_count = counts[table_name]
            action = "insert" if force or existing_count == 0 else "skip_existing"
            actions.append(
                ManifestBackfillAction(
                    processed_date=processed_date,
                    table_name=table_name,
                    action=action,
                    existing_success_count=existing_count,
                )
            )

    return actions


def execute_manifest_backfill(
    connection: Connection,
    actions: list[ManifestBackfillAction],
    started_at: datetime,
    reason: str,
) -> int:
    inserted = 0
    for action in actions:
        if action.action != "insert":
            continue
        now = datetime.now()
        insert_manifest(
            connection,
            table_name=action.table_name,
            processed_date=action.processed_date,
            record_count=0,
            start_time=started_at,
            end_time=now,
            status="SUCCESS",
            error_message=reason,
        )
        inserted += 1
    return inserted


def date_range(start_date: date, end_date: date):
    _validate_date_range(start_date, end_date)
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _validate_date_range(start_date: date, end_date: date) -> None:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
