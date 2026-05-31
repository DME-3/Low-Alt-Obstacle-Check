from datetime import date, datetime

from sqlalchemy import create_engine, text

from lac_pipeline.manifest_backfill import (
    execute_manifest_backfill,
    plan_manifest_backfill,
)
from lac_pipeline.publishing import PublishTarget


def test_plan_manifest_backfill_inserts_missing_and_skips_existing():
    engine = create_engine("sqlite:///:memory:")
    target = _target()
    _create_manifest(engine)

    with engine.begin() as connection:
        _insert_existing(connection, target.main_table, date(2026, 5, 29))
        actions = plan_manifest_backfill(
            connection,
            target,
            date(2026, 5, 29),
            date(2026, 5, 29),
        )

    assert [action.action for action in actions] == [
        "skip_existing",
        "insert",
        "insert",
    ]


def test_execute_manifest_backfill_writes_only_insert_actions():
    engine = create_engine("sqlite:///:memory:")
    target = _target()
    _create_manifest(engine)

    with engine.begin() as connection:
        _insert_existing(connection, target.main_table, date(2026, 5, 29))
        actions = plan_manifest_backfill(
            connection,
            target,
            date(2026, 5, 29),
            date(2026, 5, 29),
        )
        inserted = execute_manifest_backfill(
            connection,
            actions,
            started_at=datetime(2026, 5, 31, 2, 0, 0),
            reason="manual manifest backfill",
        )

    with engine.connect() as connection:
        total = connection.execute(text("SELECT COUNT(*) FROM manifest")).scalar_one()

    assert inserted == 2
    assert total == 3


def test_plan_manifest_backfill_force_inserts_all_entries():
    engine = create_engine("sqlite:///:memory:")
    target = _target()
    _create_manifest(engine)

    with engine.begin() as connection:
        _insert_existing(connection, target.main_table, date(2026, 5, 29))
        actions = plan_manifest_backfill(
            connection,
            target,
            date(2026, 5, 29),
            date(2026, 5, 29),
            force=True,
        )

    assert [action.action for action in actions] == ["insert", "insert", "insert"]


def _target():
    return PublishTarget(
        name="test",
        database_name="ignored",
        main_table="main_data_test",
        inf_table="inf_data_test",
        gndinf_table="gndinf_data_test",
    )


def _create_manifest(engine):
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE manifest ("
                "table_name TEXT, processed_date DATE, record_count INTEGER, "
                "start_time DATETIME, end_time DATETIME, duration_sec INTEGER, "
                "status TEXT, error_message TEXT)"
            )
        )


def _insert_existing(connection, table_name, processed_date):
    connection.execute(
        text(
            "INSERT INTO manifest "
            "(table_name, processed_date, record_count, start_time, end_time, "
            "duration_sec, status, error_message) "
            "VALUES (:table_name, :processed_date, 0, :start_time, :end_time, "
            "0, 'SUCCESS', NULL)"
        ),
        {
            "table_name": table_name,
            "processed_date": processed_date,
            "start_time": datetime(2026, 5, 31, 2, 0, 0),
            "end_time": datetime(2026, 5, 31, 2, 0, 1),
        },
    )
