#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import paramiko

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LOGGER = logging.getLogger("reprocess_production_day")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manual production recovery tool. Dry-run by default. With explicit "
            "confirmation, deletes one pipeline date from production tables and "
            "republishes it with OSN_data_update.py."
        )
    )
    parser.add_argument("--date", required=True, help="Europe/Paris pipeline date, YYYY-MM-DD.")
    parser.add_argument("--mysql-secrets", default="mysql_secrets.json")
    parser.add_argument("--ssh-key", default="./.ssh/id_ed25519")
    parser.add_argument("--execute", action="store_true", help="Actually delete and republish.")
    parser.add_argument(
        "--confirm-production",
        action="store_true",
        help="Required together with --execute.",
    )
    parser.add_argument(
        "--confirmation-token",
        help=(
            "Optional non-interactive confirmation token. Must exactly match "
            "'DELETE AND REPROCESS YYYY-MM-DD'."
        ),
    )
    parser.add_argument(
        "--skip-reload",
        action="store_true",
        help="Pass --skip-reload to the republish step.",
    )
    parser.add_argument(
        "--show-child-output",
        action="store_true",
        help="Stream dry-run/publish child logs instead of buffering them.",
    )
    parser.add_argument("--query-attempts", type=int, default=2)
    parser.add_argument("--query-retry-delay-seconds", type=int, default=60)
    parser.add_argument("--max-runtime-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument("--http-timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from lac_pipeline.publishing import build_publish_target, mysql_engine_via_tunnel
    from lac_pipeline.reprocess import (
        delete_production_day,
        format_delete_results,
        format_replacement_plan,
        plan_production_day_replacement,
    )

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    processed_date = date.fromisoformat(args.date)
    if args.execute and not args.confirm_production:
        parser.error("--execute requires --confirm-production")

    creds = load_mysql_creds(args.mysql_secrets)
    target = build_publish_target(creds, "prod")
    ssh_key = paramiko.Ed25519Key(filename=args.ssh_key)

    with mysql_engine_via_tunnel(creds, ssh_key, target) as engine:
        with engine.connect() as connection:
            plan = plan_production_day_replacement(connection, target, processed_date)
    print(format_replacement_plan(processed_date, plan))
    print()
    print("Dry-run mode by default: no production rows were changed.")

    if not args.execute:
        print("Add --execute --confirm-production to delete and republish this date.")
        return 0

    print()
    print("Preflight: running the new pipeline in dry-run mode before deleting production rows.")
    preflight_code = run_preflight(args, processed_date)
    if preflight_code != 0:
        LOGGER.error("preflight_failed returncode=%s", preflight_code)
        return preflight_code

    if not confirmation_accepted(args, processed_date):
        LOGGER.error("confirmation_failed")
        return 2

    with mysql_engine_via_tunnel(creds, ssh_key, target) as engine:
        with engine.begin() as connection:
            results = delete_production_day(connection, target, processed_date)
    print(format_delete_results(processed_date, results))

    publish_code = run_publish(args, processed_date)
    if publish_code != 0:
        LOGGER.error(
            "republish_failed returncode=%s production_date_now_needs_manual_recovery=%s",
            publish_code,
            processed_date,
        )
        return publish_code

    LOGGER.info("reprocess_complete date=%s", processed_date)
    return 0


def load_mysql_creds(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def run_preflight(args: argparse.Namespace, processed_date: date) -> int:
    from lac_pipeline.metrics import read_metrics_json

    with tempfile.NamedTemporaryFile(
        prefix=f"obstaclecheck-reprocess-preflight-{processed_date}-",
        suffix=".json",
        delete=False,
    ) as metrics_file:
        metrics_path = Path(metrics_file.name)
    try:
        command = base_update_command(args, processed_date) + [
            "--target",
            "test",
            "--validation-metrics-json",
            str(metrics_path),
        ]
        code = run_child(command, args.show_child_output)
        if code != 0:
            return code
        metrics_date, metrics = read_metrics_json(metrics_path)
        if metrics_date != processed_date:
            raise RuntimeError(
                f"preflight metrics date {metrics_date} did not match {processed_date}"
            )
        LOGGER.info(
            "preflight_metrics main_rows=%s inf_rows=%s gndinf_rows=%s",
            metrics["main_data"].rows,
            metrics["inf_data"].rows,
            metrics["gndinf_data"].rows,
        )
        return 0
    finally:
        metrics_path.unlink(missing_ok=True)


def run_publish(args: argparse.Namespace, processed_date: date) -> int:
    command = base_update_command(args, processed_date) + [
        "--publish",
        "--target",
        "prod",
        "--confirm-production",
    ]
    if args.skip_reload:
        command.append("--skip-reload")
    return run_child(command, args.show_child_output)


def base_update_command(args: argparse.Namespace, processed_date: date) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "OSN_data_update.py"),
        "--date",
        processed_date.isoformat(),
        "--query-attempts",
        str(args.query_attempts),
        "--query-retry-delay-seconds",
        str(args.query_retry_delay_seconds),
        "--max-runtime-seconds",
        str(args.max_runtime_seconds),
        "--http-timeout-seconds",
        str(args.http_timeout_seconds),
        "--log-level",
        args.log_level,
    ]


def run_child(command: list[str], stream_output: bool) -> int:
    LOGGER.info("running_child command=%s", " ".join(command))
    if stream_output:
        return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr)
    return completed.returncode


def confirmation_accepted(args: argparse.Namespace, processed_date: date) -> bool:
    expected = f"DELETE AND REPROCESS {processed_date.isoformat()}"
    if args.confirmation_token is not None:
        return args.confirmation_token == expected
    print()
    print("This will DELETE production rows and manifest entries for one date,")
    print("then republish that date with the current pipeline code.")
    print(f"Type exactly: {expected}")
    try:
        entered = input("> ")
    except EOFError:
        return False
    return entered == expected


if __name__ == "__main__":
    raise SystemExit(main())
