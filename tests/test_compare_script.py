import argparse
import sys

from scripts.compare_dry_run_to_prod import REPO_ROOT, build_dry_run_command


def test_build_dry_run_command_uses_hidden_metrics_export(tmp_path):
    args = argparse.Namespace(
        query_attempts=1,
        query_retry_delay_seconds=0,
        max_runtime_seconds=60,
        http_timeout_seconds=10,
        log_level="INFO",
    )
    metrics_path = tmp_path / "metrics.json"

    command = build_dry_run_command("2026-05-29", metrics_path, args)

    assert command[:4] == [
        sys.executable,
        str(REPO_ROOT / "OSN_data_update.py"),
        "--date",
        "2026-05-29",
    ]
    assert "--publish" not in command
    assert "--validation-metrics-json" in command
    assert str(metrics_path) in command
