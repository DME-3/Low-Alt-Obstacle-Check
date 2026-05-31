import os
import time

from lac_pipeline.retention import cleanup_candidates, find_retention_candidates


def test_find_retention_candidates_filters_by_age_and_pattern(tmp_path):
    old_pickle = tmp_path / "old.pkl"
    fresh_pickle = tmp_path / "fresh.pkl"
    old_json = tmp_path / "old.json"
    old_pickle.write_text("old")
    fresh_pickle.write_text("fresh")
    old_json.write_text("json")

    now = time.time()
    old = now - 10 * 24 * 60 * 60
    fresh = now - 1 * 24 * 60 * 60
    for path in (old_pickle, old_json):
        path.touch()
        path.chmod(0o644)
        os.utime(path, (old, old))
    os.utime(fresh_pickle, (fresh, fresh))

    candidates = find_retention_candidates(
        tmp_path,
        patterns=["*.pkl"],
        older_than_days=7,
        now=now,
    )

    assert [candidate.path for candidate in candidates] == [old_pickle]


def test_cleanup_candidates_dry_run_keeps_files(tmp_path):
    old_file = tmp_path / "old.pkl"
    old_file.write_text("old")
    candidates = find_retention_candidates(
        tmp_path,
        patterns=["*.pkl"],
        older_than_days=0,
        now=time.time(),
    )

    removed_count, removed_bytes = cleanup_candidates(candidates, execute=False)

    assert removed_count == 0
    assert removed_bytes == 0
    assert old_file.exists()


def test_cleanup_candidates_execute_removes_files(tmp_path):
    old_file = tmp_path / "old.pkl"
    old_file.write_text("old")
    candidates = find_retention_candidates(
        tmp_path,
        patterns=["*.pkl"],
        older_than_days=0,
        now=time.time(),
    )

    removed_count, removed_bytes = cleanup_candidates(candidates, execute=True)

    assert removed_count == 1
    assert removed_bytes == 3
    assert not old_file.exists()

