# Operations

## Dry Run

Dry-run is the default and performs no MySQL upload and no PythonAnywhere reload:

```bash
cd /home/dimitri/obstaclecheck
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_update.py
```

For a specific Europe/Paris pipeline date:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_update.py --date 2026-05-29
```

Dry-runs still query OpenSky and may write local pickles. The pipeline date is a Cologne local calendar day, so validation accepts the corresponding UTC boundary rows.

To print detected candidate-infraction counts plus the first `N` rows of the `inf` and `gndinf` result tables during a dry-run:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_update.py \
  --date 2026-05-29 \
  --show_results 5
```

## Dry Run / Production Comparison

To compare a fresh dry-run's validation metrics with read-only production table aggregates for the same Europe/Paris pipeline date:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python scripts/compare_dry_run_to_prod.py \
  --date 2026-05-29
```

The script reports row counts and unique `icao24` counts for `main_data`, `inf_data`, and `gndinf_data`, plus dry-run minus production differences. It does not publish or modify production tables.

## Test Upload

Upload to test tables and skip web reload:

```bash
cd /home/dimitri/obstaclecheck
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_update.py \
  --date 2026-05-29 \
  --publish \
  --target test \
  --skip-reload
```

## Production Upload

Production upload is opt-in:

```bash
cd /home/dimitri/obstaclecheck
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_update.py \
  --publish \
  --target prod \
  --confirm-production
```

## Recommended Cron

Recommended production cron entry:

```cron
0 2 * * * cd /home/dimitri/obstaclecheck/ && /home/dimitri/obstaclecheck/.venv/bin/python /home/dimitri/obstaclecheck/OSN_data_update.py --publish --target prod --confirm-production --log-file /home/dimitri/OSN_pipeline.log >> /home/dimitri/OSN_log.txt 2>&1
```

The command keeps the existing cron-compatible entry point, adds explicit production confirmation, keeps the legacy stdout/stderr log, and also writes a rotating structured log.

## Locking And Runtime

Default lock file:

```text
/tmp/obstaclecheck-nightly.lock
```

Default max runtime:

```text
21600 seconds
```

If a previous process is still running, the next cron run exits with code `75`. If a lock file remains for a dead process and is older than the stale-lock threshold, it is recovered.

## Failure Recovery

1. Check `/home/dimitri/OSN_log.txt` and `/home/dimitri/OSN_pipeline.log`.
2. Check whether a process is still alive:

```bash
ps -ef | grep OSN_data_update.py
```

3. If no process is alive but `/tmp/obstaclecheck-nightly.lock` remains, the next run should recover it after the stale threshold.
4. Verify manifest rows for the target date and all three tables before re-running with `--publish`.
5. If only some tables have `SUCCESS` manifest rows, automatic publishing now fails fast; perform manual recovery before retrying.
6. If OpenSky/Trino returns an availability HTTP error, including the OSN-down `404 Not Found` case, the pipeline logs `opensky_unavailable`, releases the lock, and exits with code `75` without a Python traceback. Retry after the service recovers.
7. Prefer a test upload before a production re-run when the failure happened after data processing.

## Production Day Reprocessing

A production day can be replaced manually after review. The tool is dry-run by default and first reports the rows and manifest entries that would be deleted:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python scripts/reprocess_production_day.py \
  --date 2026-05-29
```

Execution requires explicit flags and a typed confirmation phrase. It runs a preflight dry-run, deletes the date from all three production tables plus matching manifest rows, then republishes the same date through `OSN_data_update.py`:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python scripts/reprocess_production_day.py \
  --date 2026-05-29 \
  --execute \
  --confirm-production
```

For non-interactive manual use, pass the exact confirmation token:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python scripts/reprocess_production_day.py \
  --date 2026-05-29 \
  --execute \
  --confirm-production \
  --confirmation-token "DELETE AND REPROCESS 2026-05-29"
```

If republishing fails after deletion, that date needs immediate manual recovery before the site should be considered complete for the day.

## Manual Recovery Tools

`OSN_data_backfill.py` is now a manual recovery wrapper around the hardened nightly
entry point. By default it only prints the per-date commands it would run.

Backfill plan only, no OpenSky queries and no writes:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_backfill.py \
  --start-date 2026-05-24 \
  --end-date 2026-05-29
```

Execute per-date dry-run child pipelines, still no DB writes:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_backfill.py \
  --start-date 2026-05-24 \
  --end-date 2026-05-29 \
  --execute
```

Execute test-table publishes and skip reload:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_backfill.py \
  --start-date 2026-05-24 \
  --end-date 2026-05-29 \
  --execute \
  --publish \
  --target test \
  --skip-reload
```

Execute production publishes:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_backfill.py \
  --start-date 2026-05-24 \
  --end-date 2026-05-29 \
  --execute \
  --publish \
  --target prod \
  --confirm-production
```

Manifest backfill dry-run for test target:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python backfill_manifest.py \
  --start-date 2026-05-24 \
  --end-date 2026-05-29 \
  --target test
```

Manifest backfill execution for test target:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python backfill_manifest.py \
  --start-date 2026-05-24 \
  --end-date 2026-05-29 \
  --target test \
  --execute
```

Manifest backfill execution for production:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python backfill_manifest.py \
  --start-date 2026-05-24 \
  --end-date 2026-05-29 \
  --target prod \
  --execute \
  --confirm-production
```

Use `--force-duplicate` only when deliberately inserting additional manifest rows for an audited recovery.

## Artifact Retention

Generated pickle and dataframe artifacts can be scanned without deleting anything:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python scripts/cleanup_artifacts.py \
  --older-than-days 120
```

Deletion requires explicit `--execute`:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python scripts/cleanup_artifacts.py \
  --older-than-days 120 \
  --execute
```

Review the dry-run output before executing cleanup on the production server.

## Rollback

To return to the previous production behavior, deploy the previous commit on `main` and restore the old cron entry. Before rollback, check whether any partial publish occurred and inspect manifest state for the affected date.

