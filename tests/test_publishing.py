from datetime import date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from lac_pipeline.publishing import (
    PROD_DEFAULTS,
    TEST_DEFAULTS,
    PublishProtectionError,
    PublishTarget,
    build_publish_target,
    date_fully_published,
    publish_dataframes,
    require_publish_allowed,
)


def test_require_publish_allowed_blocks_unconfirmed_prod():
    with pytest.raises(PublishProtectionError):
        require_publish_allowed(True, "prod", False)


def test_require_publish_allowed_accepts_dry_run():
    require_publish_allowed(False, "prod", False)


def test_build_publish_target_uses_safe_defaults():
    test_target = build_publish_target({}, "test")
    prod_target = build_publish_target({}, "prod")

    assert test_target.database_name == TEST_DEFAULTS["database"]
    assert test_target.main_table == TEST_DEFAULTS["main"]
    assert prod_target.database_name == PROD_DEFAULTS["database"]
    assert prod_target.main_table == PROD_DEFAULTS["main"]


def test_build_publish_target_rejects_unsafe_identifier():
    with pytest.raises(PublishProtectionError):
        build_publish_target({"MAIN_TEST_TABLE_NAME": "main_data_test;drop"}, "test")


def test_publish_dataframes_is_transactional_for_manifest_success():
    engine = create_engine("sqlite:///:memory:")
    target = PublishTarget(
        name="test",
        database_name="ignored",
        main_table="main_data_test",
        inf_table="inf_data_test",
        gndinf_table="gndinf_data_test",
    )
    processed_date = date(2026, 5, 29)
    main_df = pd.DataFrame({"time": [1779984000], "icao24": ["abc123"]})
    inf_df = pd.DataFrame({"time": [datetime(2026, 5, 29)], "inf_ref": ["abc_0"]})
    gndinf_df = pd.DataFrame({"time": [datetime(2026, 5, 29)], "inf_ref": ["abc_gnd_0"]})

    _create_publish_tables(engine, target, main_df, inf_df, gndinf_df)

    counts = publish_dataframes(
        engine,
        target,
        main_df,
        inf_df,
        gndinf_df,
        processed_date,
        datetime(2026, 5, 31, 2, 0, 0),
        logger=_NullLogger(),
    )

    assert counts.main_rows == 1
    assert counts.inf_rows == 1
    assert counts.gndinf_rows == 1
    with engine.connect() as connection:
        assert date_fully_published(connection, processed_date, target.table_names)
        manifest_count = connection.execute(text("SELECT COUNT(*) FROM manifest")).scalar_one()
        assert manifest_count == 3


def _create_publish_tables(engine, target, main_df, inf_df, gndinf_df):
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE manifest ("
                "table_name TEXT, processed_date DATE, record_count INTEGER, "
                "start_time DATETIME, end_time DATETIME, duration_sec INTEGER, "
                "status TEXT, error_message TEXT)"
            )
        )
    main_df.head(0).to_sql(target.main_table, engine, if_exists="replace")
    inf_df.head(0).to_sql(target.inf_table, engine, if_exists="replace")
    gndinf_df.head(0).to_sql(target.gndinf_table, engine, if_exists="replace")


class _NullLogger:
    def info(self, *args, **kwargs):
        return None

