import argparse
import sys
from pathlib import Path

from OSN_data_backfill import build_child_command


def test_build_child_command_defaults_to_child_dry_run():
    args = argparse.Namespace(
        target="test",
        max_runtime_seconds=60,
        query_attempts=1,
        query_retry_delay_seconds=0,
        http_timeout_seconds=10,
        log_level="INFO",
        publish=False,
        confirm_production=False,
        skip_reload=False,
    )

    command = build_child_command(Path("OSN_data_update.py"), "2026-05-29", args)

    assert command[:4] == [sys.executable, "OSN_data_update.py", "--date", "2026-05-29"]
    assert "--publish" not in command
    assert "--confirm-production" not in command


def test_build_child_command_for_confirmed_production_publish():
    args = argparse.Namespace(
        target="prod",
        max_runtime_seconds=60,
        query_attempts=1,
        query_retry_delay_seconds=0,
        http_timeout_seconds=10,
        log_level="INFO",
        publish=True,
        confirm_production=True,
        skip_reload=True,
    )

    command = build_child_command(Path("OSN_data_update.py"), "2026-05-29", args)

    assert "--publish" in command
    assert "--target" in command
    assert "prod" in command
    assert "--confirm-production" in command
    assert "--skip-reload" in command

