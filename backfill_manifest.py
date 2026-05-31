#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import paramiko
from sqlalchemy import create_engine

from lac_pipeline.manifest_backfill import (
    execute_manifest_backfill,
    plan_manifest_backfill,
)
from lac_pipeline.publishing import build_publish_target, mysql_engine_via_tunnel

LOGGER = logging.getLogger("backfill_manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual manifest backfill tool.")
    parser.add_argument("--start-date", required=True, help="First date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Last date, YYYY-MM-DD.")
    parser.add_argument("--target", choices=("test", "prod"), default="test")
    parser.add_argument("--execute", action="store_true", help="Actually write manifest rows.")
    parser.add_argument(
        "--confirm-production",
        action="store_true",
        help="Required with --execute --target prod.",
    )
    parser.add_argument(
        "--force-duplicate",
        action="store_true",
        help="Insert rows even when SUCCESS manifest entries already exist.",
    )
    parser.add_argument(
        "--reason",
        default="manual manifest backfill",
        help="Reason stored in manifest.error_message for inserted rows.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end_date < start_date:
        parser.error("--end-date must be on or after --start-date")

    if args.execute and args.target == "prod" and not args.confirm_production:
        parser.error(
            "production manifest backfill requires "
            "--execute --target prod --confirm-production"
        )

    creds = _load_mysql_creds() if args.execute else {}
    target = build_publish_target(creds, args.target)
    LOGGER.info(
        "manifest_backfill_start target=%s start_date=%s end_date=%s execute=%s force=%s",
        target.name,
        start_date,
        end_date,
        args.execute,
        args.force_duplicate,
    )

    if args.execute:
        ssh_key = paramiko.Ed25519Key(filename="./.ssh/id_ed25519")
        with mysql_engine_via_tunnel(creds, ssh_key, target) as engine:
            with engine.begin() as connection:
                actions = plan_manifest_backfill(
                    connection,
                    target,
                    start_date,
                    end_date,
                    force=args.force_duplicate,
                )
                _log_actions(actions)
                inserted = execute_manifest_backfill(
                    connection,
                    actions,
                    started_at=datetime.now(),
                    reason=args.reason,
                )
        LOGGER.info("manifest_backfill_complete inserted=%s", inserted)
        return 0

    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        _create_empty_manifest(connection)
        actions = plan_manifest_backfill(
            connection,
            target,
            start_date,
            end_date,
            force=args.force_duplicate,
        )
    _log_actions(actions)
    LOGGER.info(
        "manifest_backfill_complete mode=dry-run planned_inserts=%s",
        _count_inserts(actions),
    )
    return 0


def _load_mysql_creds() -> dict:
    with Path("mysql_secrets.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _log_actions(actions) -> None:
    for action in actions:
        LOGGER.info(
            "manifest_action date=%s table=%s action=%s existing_success_count=%s",
            action.processed_date,
            action.table_name,
            action.action,
            action.existing_success_count,
        )


def _count_inserts(actions) -> int:
    return sum(1 for action in actions if action.action == "insert")


def _create_empty_manifest(connection) -> None:
    connection.exec_driver_sql(
        "CREATE TABLE manifest ("
        "table_name TEXT, processed_date DATE, record_count INTEGER, "
        "start_time DATETIME, end_time DATETIME, duration_sec INTEGER, "
        "status TEXT, error_message TEXT)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
