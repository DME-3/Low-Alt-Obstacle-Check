from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterator, Mapping, Optional
from urllib.parse import quote_plus

import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sshtunnel import SSHTunnelForwarder


class PublishProtectionError(RuntimeError):
    """Raised when a requested publish target is not explicitly allowed."""


class PublishVerificationError(RuntimeError):
    """Raised when post-upload verification does not match expectations."""


@dataclass(frozen=True)
class PublishTarget:
    name: str
    database_name: str
    main_table: str
    inf_table: str
    gndinf_table: str

    @property
    def table_names(self) -> tuple[str, str, str]:
        return (self.main_table, self.inf_table, self.gndinf_table)


@dataclass(frozen=True)
class PublishCounts:
    main_rows: int
    inf_rows: int
    gndinf_rows: int

    @property
    def by_table_position(self) -> tuple[int, int, int]:
        return (self.main_rows, self.inf_rows, self.gndinf_rows)


TEST_DEFAULTS = {
    "database": "testdatabase",
    "main": "main_data_test",
    "inf": "inf_data_test",
    "gndinf": "gndinf_data_test",
}

PROD_DEFAULTS = {
    "database": "LAC_db",
    "main": "main_data",
    "inf": "inf_data",
    "gndinf": "gndinf_data",
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")


def build_publish_target(creds: Mapping[str, str], target_name: str) -> PublishTarget:
    if target_name == "prod":
        defaults = PROD_DEFAULTS
        prefix = "PROD"
    else:
        defaults = TEST_DEFAULTS
        prefix = "TEST"

    target = PublishTarget(
        name=target_name,
        database_name=creds.get(f"{prefix}_DATABASE_NAME", defaults["database"]),
        main_table=creds.get(f"MAIN_{prefix}_TABLE_NAME", defaults["main"]),
        inf_table=creds.get(f"INF_{prefix}_TABLE_NAME", defaults["inf"]),
        gndinf_table=creds.get(f"GNDINF_{prefix}_TABLE_NAME", defaults["gndinf"]),
    )
    validate_target_identifiers(target)
    return target


def validate_target_identifiers(target: PublishTarget) -> None:
    names = [target.database_name, *target.table_names, "manifest"]
    unsafe = [name for name in names if not _IDENTIFIER_RE.match(name)]
    if unsafe:
        raise PublishProtectionError(f"unsafe SQL identifier(s): {unsafe}")


def require_publish_allowed(
    publish: bool,
    target: str,
    confirm_production: bool,
) -> None:
    if not publish:
        return
    if target == "prod" and not confirm_production:
        raise PublishProtectionError(
            "production publish requires --publish --target prod --confirm-production"
        )


@contextmanager
def mysql_engine_via_tunnel(
    creds: Mapping[str, str],
    ssh_pkey: object,
    target: PublishTarget,
) -> Iterator[Engine]:
    with SSHTunnelForwarder(
        (creds["SSH_ADDRESS"], 22),
        ssh_username=creds["SSH_USERNAME"],
        ssh_pkey=ssh_pkey,
        remote_bind_address=(
            creds["REMOTE_BIND_ADDRESS"],
            int(creds["REMOTE_BIND_PORT"]),
        ),
        allow_agent=False,
    ) as tunnel:
        engine = create_engine(_engine_url(creds, target, tunnel.local_bind_port))
        try:
            yield engine
        finally:
            engine.dispose()


def _engine_url(creds: Mapping[str, str], target: PublishTarget, port: int) -> str:
    username = quote_plus(creds["SSH_USERNAME"])
    password = quote_plus(creds["PYANYWHERE_PASSWORD"])
    database = target.database_name
    return f"mysql+pymysql://{username}:{password}@127.0.0.1:{port}/dme3${database}"


def date_fully_published(
    connection: Connection,
    processed_date: date,
    table_names: tuple[str, str, str],
) -> bool:
    statuses = [
        _manifest_success_count(connection, table_name, processed_date)
        for table_name in table_names
    ]
    return all(count > 0 for count in statuses)


def publish_empty_day(
    engine: Engine,
    target: PublishTarget,
    processed_date: date,
    run_started_at: datetime,
    logger: logging.Logger,
) -> None:
    with engine.begin() as connection:
        if date_fully_published(connection, processed_date, target.table_names):
            logger.info("publish_skip already_published date=%s", processed_date)
            return
        for table_name in target.table_names:
            insert_manifest(
                connection,
                table_name=table_name,
                processed_date=processed_date,
                record_count=0,
                start_time=run_started_at,
                end_time=datetime.now(),
                status="SUCCESS",
                error_message="empty OpenSky state-vector day",
            )


def publish_dataframes(
    engine: Engine,
    target: PublishTarget,
    main_df: pd.DataFrame,
    inf_df: pd.DataFrame,
    gndinf_df: pd.DataFrame,
    processed_date: date,
    run_started_at: datetime,
    logger: logging.Logger,
) -> PublishCounts:
    counts = PublishCounts(len(main_df), len(inf_df), len(gndinf_df))
    frames = (main_df, inf_df, gndinf_df)

    with engine.begin() as connection:
        if date_fully_published(connection, processed_date, target.table_names):
            logger.info("publish_skip already_published date=%s", processed_date)
            return counts

        before_counts = {
            table_name: count_rows(connection, table_name)
            for table_name in target.table_names
        }

        for table_name, frame in zip(target.table_names, frames):
            logger.info(
                "upload_table_start table=%s rows=%s",
                table_name,
                len(frame),
            )
            if not frame.empty:
                frame.to_sql(con=connection, name=table_name, if_exists="append")
            logger.info("upload_table_end table=%s", table_name)

        after_counts = {
            table_name: count_rows(connection, table_name)
            for table_name in target.table_names
        }
        for table_name, expected_delta in zip(
            target.table_names, counts.by_table_position
        ):
            actual_delta = after_counts[table_name] - before_counts[table_name]
            if actual_delta != expected_delta:
                raise PublishVerificationError(
                    f"{table_name} expected {expected_delta} new rows, "
                    f"observed {actual_delta}"
                )

        for table_name, record_count in zip(
            target.table_names, counts.by_table_position
        ):
            insert_manifest(
                connection,
                table_name=table_name,
                processed_date=processed_date,
                record_count=record_count,
                start_time=run_started_at,
                end_time=datetime.now(),
                status="SUCCESS",
            )

    return counts


def insert_manifest(
    connection: Connection,
    table_name: str,
    processed_date: date,
    record_count: int,
    start_time: datetime,
    end_time: datetime,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    duration_sec = int((end_time - start_time).total_seconds())
    connection.execute(
        text(
            "INSERT INTO manifest "
            "(table_name, processed_date, record_count, start_time, end_time, "
            "duration_sec, status, error_message) "
            "VALUES (:table_name, :processed_date, :record_count, :start_time, "
            ":end_time, :duration_sec, :status, :error_message)"
        ),
        {
            "table_name": table_name,
            "processed_date": processed_date,
            "record_count": record_count,
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": duration_sec,
            "status": status,
            "error_message": error_message,
        },
    )


def count_rows(connection: Connection, table_name: str) -> int:
    _validate_identifier(table_name)
    result = connection.execute(text(f"SELECT COUNT(*) AS n FROM `{table_name}`"))
    return int(result.scalar_one())


def reload_pythonanywhere(
    creds: Mapping[str, str],
    timeout_seconds: int,
    logger: logging.Logger,
) -> int:
    url = "https://{host}/api/v0/user/{username}/webapps/{domain}/reload/".format(
        host=creds["PYA_host"],
        username=creds["PYA_username"],
        domain=creds["PYA_domain"],
    )
    response = requests.post(
        url,
        headers={"Authorization": "Token {token}".format(token=creds["PYA_token"])},
        timeout=timeout_seconds,
    )
    logger.info("pythonanywhere_reload status_code=%s", response.status_code)
    response.raise_for_status()
    return response.status_code


def _manifest_success_count(
    connection: Connection, table_name: str, processed_date: date
) -> int:
    result = connection.execute(
        text(
            "SELECT COUNT(*) AS n FROM manifest "
            "WHERE table_name = :table_name "
            "AND processed_date = :processed_date "
            "AND status = 'SUCCESS'"
        ),
        {"table_name": table_name, "processed_date": processed_date},
    )
    return int(result.scalar_one())


def _validate_identifier(name: str) -> None:
    if not _IDENTIFIER_RE.match(name):
        raise PublishProtectionError(f"unsafe SQL identifier: {name}")

