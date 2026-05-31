import argparse
import sys
from datetime import date

from scripts.reprocess_production_day import (
    REPO_ROOT,
    base_update_command,
    confirmation_accepted,
)


def test_base_update_command_builds_dated_pipeline_command():
    args = argparse.Namespace(
        query_attempts=1,
        query_retry_delay_seconds=0,
        max_runtime_seconds=60,
        http_timeout_seconds=10,
        log_level="INFO",
    )

    command = base_update_command(args, date(2026, 5, 29))

    assert command[:4] == [
        sys.executable,
        str(REPO_ROOT / "OSN_data_update.py"),
        "--date",
        "2026-05-29",
    ]
    assert "--publish" not in command


def test_confirmation_token_must_match_exact_phrase():
    good = argparse.Namespace(confirmation_token="DELETE AND REPROCESS 2026-05-29")
    bad = argparse.Namespace(confirmation_token="yes")

    assert confirmation_accepted(good, date(2026, 5, 29)) is True
    assert confirmation_accepted(bad, date(2026, 5, 29)) is False
