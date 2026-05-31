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


LOGGER = logging.getLogger("compare_dry_run_to_prod")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pipeline dry-run for one date and compare validation metrics "
            "against read-only production table aggregates."
        )
    )
    parser.add_argument("--date", required=True, help="Europe/Paris pipeline date, YYYY-MM-DD.")
    parser.add_argument("--mysql-secrets", default="mysql_secrets.json")
    parser.add_argument("--ssh-key", default="./.ssh/id_ed25519")
    parser.add_argument("--query-attempts", type=int, default=2)
    parser.add_argument("--query-retry-delay-seconds", type=int, default=60)
    parser.add_argument("--max-runtime-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument("--http-timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--show-dry-run-output",
        action="store_true",
        help="Stream the child dry-run logs instead of printing them only on failure.",
    )
    parser.add_argument(
        "--keep-metrics-json",
        action="store_true",
        help="Keep the temporary dry-run metrics JSON file and print its path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from lac_pipeline.metrics import read_metrics_json
    from lac_pipeline.production_compare import compare_metrics, format_comparison_report

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    processed_date = date.fromisoformat(args.date)
    with tempfile.NamedTemporaryFile(
        prefix=f"obstaclecheck-dry-run-metrics-{processed_date}-",
        suffix=".json",
        delete=False,
    ) as metrics_file:
        metrics_path = Path(metrics_file.name)

    try:
        dry_run_code = run_dry_run_metrics(args, metrics_path)
        if dry_run_code != 0:
            LOGGER.error("dry_run_failed returncode=%s", dry_run_code)
            return dry_run_code

        metrics_date, dry_run_metrics = read_metrics_json(metrics_path)
        if metrics_date != processed_date:
            raise RuntimeError(
                f"dry-run metrics date {metrics_date} did not match requested {processed_date}"
            )

        production_metrics = read_production_metrics(args, processed_date)
        comparisons = compare_metrics(dry_run_metrics, production_metrics)
        print(format_comparison_report(processed_date, comparisons))
        return 0
    finally:
        if args.keep_metrics_json:
            LOGGER.info("dry_run_metrics_json path=%s", metrics_path)
        else:
            metrics_path.unlink(missing_ok=True)


def run_dry_run_metrics(args: argparse.Namespace, metrics_path: Path) -> int:
    command = build_dry_run_command(args.date, metrics_path, args)
    LOGGER.info("running_dry_run command=%s", " ".join(command))
    if args.show_dry_run_output:
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    else:
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


def build_dry_run_command(
    target_date: str,
    metrics_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "OSN_data_update.py"),
        "--date",
        target_date,
        "--target",
        "test",
        "--validation-metrics-json",
        str(metrics_path),
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


def read_production_metrics(
    args: argparse.Namespace,
    processed_date: date,
):
    from lac_pipeline.production_compare import fetch_production_metrics
    from lac_pipeline.publishing import build_publish_target, mysql_engine_via_tunnel

    with Path(args.mysql_secrets).open("r", encoding="utf-8") as fh:
        creds = json.load(fh)
    target = build_publish_target(creds, "prod")
    ssh_key = paramiko.Ed25519Key(filename=args.ssh_key)
    LOGGER.info(
        "reading_production_metrics database=%s tables=%s date=%s",
        target.database_name,
        target.table_names,
        processed_date,
    )
    with mysql_engine_via_tunnel(creds, ssh_key, target) as engine:
        with engine.connect() as connection:
            return fetch_production_metrics(connection, target, processed_date)


if __name__ == "__main__":
    raise SystemExit(main())
