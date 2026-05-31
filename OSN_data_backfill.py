#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

LOGGER = logging.getLogger("OSN_data_backfill")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manual ADS-B recovery wrapper. Plans by default; executes the "
            "hardened OSN_data_update.py entrypoint only with --execute."
        )
    )
    parser.add_argument("--start-date", required=True, help="First date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Last date, YYYY-MM-DD.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run per-date pipeline commands.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Pass --publish to OSN_data_update.py.",
    )
    parser.add_argument("--target", choices=("test", "prod"), default="test")
    parser.add_argument(
        "--confirm-production",
        action="store_true",
        help="Required with --execute --publish --target prod.",
    )
    parser.add_argument(
        "--skip-reload",
        action="store_true",
        help="Pass --skip-reload to child runs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing later dates after a child run fails.",
    )
    parser.add_argument("--max-runtime-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument("--query-attempts", type=int, default=2)
    parser.add_argument("--query-retry-delay-seconds", type=int, default=60)
    parser.add_argument("--http-timeout-seconds", type=int, default=30)
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

    if args.execute and args.publish and args.target == "prod" and not args.confirm_production:
        parser.error(
            "production publish requires "
            "--execute --publish --target prod --confirm-production"
        )

    repo_root = Path(__file__).resolve().parent
    update_script = repo_root / "OSN_data_update.py"
    dates = list(_date_range(start_date, end_date))
    LOGGER.info(
        "backfill_plan dates=%s execute=%s publish=%s target=%s",
        len(dates),
        args.execute,
        args.publish,
        args.target,
    )

    failures: list[tuple[str, int]] = []
    for current_date in dates:
        command = build_child_command(update_script, current_date.isoformat(), args)
        LOGGER.info("planned_command date=%s command=%s", current_date, " ".join(command))

        if not args.execute:
            continue

        completed = subprocess.run(command, cwd=repo_root, check=False)
        if completed.returncode != 0:
            failures.append((current_date.isoformat(), completed.returncode))
            LOGGER.error(
                "backfill_child_failed date=%s returncode=%s",
                current_date,
                completed.returncode,
            )
            if not args.continue_on_error:
                break

    if failures:
        LOGGER.error("backfill_complete status=failed failures=%s", failures)
        return 1

    LOGGER.info("backfill_complete status=%s", "executed" if args.execute else "planned")
    return 0


def build_child_command(
    update_script: Path, target_date: str, args: argparse.Namespace
) -> list[str]:
    command = [
        sys.executable,
        str(update_script),
        "--date",
        target_date,
        "--target",
        args.target,
        "--max-runtime-seconds",
        str(args.max_runtime_seconds),
        "--query-attempts",
        str(args.query_attempts),
        "--query-retry-delay-seconds",
        str(args.query_retry_delay_seconds),
        "--http-timeout-seconds",
        str(args.http_timeout_seconds),
        "--log-level",
        args.log_level,
    ]
    if args.publish:
        command.append("--publish")
    if args.target == "prod" and args.confirm_production:
        command.append("--confirm-production")
    if args.skip_reload:
        command.append("--skip-reload")
    return command


def _date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


if __name__ == "__main__":
    raise SystemExit(main())
