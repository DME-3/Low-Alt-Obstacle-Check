from datetime import date, datetime

from sqlalchemy import create_engine, text

from lac_pipeline.publishing import PublishTarget
from lac_pipeline.reprocess import (
    delete_production_day,
    format_delete_results,
    format_replacement_plan,
    plan_production_day_replacement,
)


def test_plan_and_delete_production_day_only_target_window():
    engine = create_engine("sqlite:///:memory:")
    target = PublishTarget(
        name="prod",
        database_name="ignored",
        main_table="main_data",
        inf_table="inf_data",
        gndinf_table="gndinf_data",
    )
    _create_tables(engine)
    processed_date = date(2026, 5, 29)

    with engine.begin() as connection:
        plan = plan_production_day_replacement(connection, target, processed_date)
        results = delete_production_day(connection, target, processed_date)

    assert [item.metrics.rows for item in plan] == [2, 1, 1]
    assert [item.manifest_rows for item in plan] == [1, 1, 1]
    assert [item.deleted_rows for item in results] == [2, 1, 1]
    assert [item.deleted_manifest_rows for item in results] == [1, 1, 1]

    with engine.connect() as connection:
        remaining_main = connection.execute(text("SELECT COUNT(*) FROM main_data")).scalar_one()
        remaining_manifest = connection.execute(text("SELECT COUNT(*) FROM manifest")).scalar_one()

    assert remaining_main == 1
    assert remaining_manifest == 1


def test_reprocess_report_formatters():
    engine = create_engine("sqlite:///:memory:")
    target = PublishTarget(
        name="prod",
        database_name="ignored",
        main_table="main_data",
        inf_table="inf_data",
        gndinf_table="gndinf_data",
    )
    _create_tables(engine)

    with engine.begin() as connection:
        plan = plan_production_day_replacement(connection, target, date(2026, 5, 29))
        results = delete_production_day(connection, target, date(2026, 5, 29))

    assert "Production replacement plan for 2026-05-29" in format_replacement_plan(
        date(2026, 5, 29), plan
    )
    assert "Deleted production rows for 2026-05-29" in format_delete_results(
        date(2026, 5, 29), results
    )


def _create_tables(engine):
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE main_data (time INTEGER, icao24 TEXT)"))
        connection.execute(text("CREATE TABLE inf_data (time DATETIME, icao24 TEXT)"))
        connection.execute(text("CREATE TABLE gndinf_data (time DATETIME, icao24 TEXT)"))
        connection.execute(
            text("CREATE TABLE manifest (table_name TEXT, processed_date DATE)")
        )
        connection.execute(
            text(
                "INSERT INTO main_data (time, icao24) VALUES "
                "(1780005600, 'abc123'), (1780005601, 'def456'), "
                "(1780092000, 'outside')"
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
        connection.execute(
            text(
                "INSERT INTO manifest (table_name, processed_date) VALUES "
                "('main_data', :target), ('inf_data', :target), "
                "('gndinf_data', :target), ('main_data', :other)"
            ),
            {"target": date(2026, 5, 29), "other": date(2026, 5, 30)},
        )
