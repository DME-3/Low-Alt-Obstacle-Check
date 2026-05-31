# Operations

## Dry Run

Dry-run is the default and performs no MySQL upload and no PythonAnywhere reload:

```bash
cd /home/dimitri/obstaclecheck
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_update.py
```

For a specific date:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_update.py --date 2026-05-29
```

Dry-runs still query OpenSky and may write local pickles.

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
5. Prefer a test upload before a production re-run when the failure happened after data processing.

## Manual Recovery Tools

Backfill dry-run:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_backfill.py \
  --start-date 2026-05-24 \
  --end-date 2026-05-29
```

Backfill execution:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_backfill.py \
  --start-date 2026-05-24 \
  --end-date 2026-05-29 \
  --execute \
  --confirm-production
```

Manifest backfill dry-run:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python backfill_manifest.py \
  --start-date 2026-05-24 \
  --end-date 2026-05-29
```

Manifest backfill execution:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python backfill_manifest.py \
  --start-date 2026-05-24 \
  --end-date 2026-05-29 \
  --execute \
  --confirm-production
```

## Rollback

To return to the previous production behavior, deploy the previous commit on `main` and restore the old cron entry. Before rollback, check whether any partial publish occurred and inspect manifest state for the affected date.

