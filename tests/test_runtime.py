import json
import logging
import os
import time

import pytest

from lac_pipeline.runtime import LockError, PipelineLock, parse_runtime_settings


def test_parse_runtime_settings_defaults_to_dry_run():
    settings = parse_runtime_settings([])

    assert settings.dry_run is True
    assert settings.publish is False
    assert settings.target == "test"


def test_parse_runtime_settings_requires_explicit_publish_flag():
    settings = parse_runtime_settings(["--publish", "--target", "prod", "--confirm-production"])

    assert settings.dry_run is False
    assert settings.publish is True
    assert settings.target == "prod"
    assert settings.confirm_production is True


def test_parse_runtime_settings_accepts_show_results_aliases():
    underscore = parse_runtime_settings(["--show_results", "3"])
    hyphen = parse_runtime_settings(["--show-results", "4"])

    assert underscore.show_results == 3
    assert hyphen.show_results == 4


def test_pipeline_lock_acquire_and_release(tmp_path):
    lock_path = tmp_path / "nightly.lock"
    logger = logging.getLogger("test")
    lock = PipelineLock(lock_path, "run-1", 60, logger)

    lock.acquire()
    assert lock_path.exists()

    lock.release()
    assert not lock_path.exists()


def test_pipeline_lock_blocks_running_pid(tmp_path):
    lock_path = tmp_path / "nightly.lock"
    lock_path.write_text(json.dumps({"pid": os.getpid(), "run_id": "other"}))
    logger = logging.getLogger("test")
    lock = PipelineLock(lock_path, "run-2", 1, logger)

    with pytest.raises(LockError):
        lock.acquire()


def test_pipeline_lock_recovers_stale_dead_pid(tmp_path):
    lock_path = tmp_path / "nightly.lock"
    lock_path.write_text(json.dumps({"pid": 999999, "run_id": "old"}))
    old = time.time() - 3600
    os.utime(lock_path, (old, old))
    logger = logging.getLogger("test")
    lock = PipelineLock(lock_path, "run-3", 1, logger)

    lock.acquire()

    assert json.loads(lock_path.read_text())["run_id"] == "run-3"
    lock.release()

