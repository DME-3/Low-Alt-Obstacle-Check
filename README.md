# Low-Alt-Obstacle-Check

Production ADS-B data pipeline for the Low Alt Cologne site. The nightly job imports OpenSky ADS-B data, enriches it with terrain/population/obstacle data, identifies candidate low-altitude occurrences, publishes MySQL tables for the PythonAnywhere site, and reloads the site after a successful publish.

## Current Entry Point

```bash
/home/dimitri/obstaclecheck/.venv/bin/python /home/dimitri/obstaclecheck/OSN_data_update.py
```

The script is cron-compatible, but it now defaults to dry-run mode. Production upload requires explicit flags.

## Dry Run

```bash
cd /home/dimitri/obstaclecheck
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_update.py
```

Dry-run mode performs no MySQL upload and no PythonAnywhere reload.

## Test Upload

```bash
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_update.py \
  --date 2026-05-29 \
  --publish \
  --target test \
  --skip-reload
```

## Production Upload

```bash
/home/dimitri/obstaclecheck/.venv/bin/python OSN_data_update.py \
  --publish \
  --target prod \
  --confirm-production
```

## Recommended Cron

```cron
0 2 * * * cd /home/dimitri/obstaclecheck/ && /home/dimitri/obstaclecheck/.venv/bin/python /home/dimitri/obstaclecheck/OSN_data_update.py --publish --target prod --confirm-production --log-file /home/dimitri/OSN_pipeline.log >> /home/dimitri/OSN_log.txt 2>&1
```

## Safety Features

- Single-run lock with stale-lock recovery.
- Process max-runtime guard.
- Dry-run by default.
- Production write protection.
- Early manifest/idempotency check.
- Transaction-scoped publish path with row-count verification.
- PythonAnywhere reload timeout.
- Structured stage logging with run IDs.
- Validation before publish.

## Tests And Lint

```bash
/home/dimitri/obstaclecheck/.venv/bin/python -m pytest
/home/dimitri/obstaclecheck/.venv/bin/python -m ruff check OSN_data_update.py lac_pipeline tests
```

## Documentation

- `docs/audit.md`
- `docs/architecture.md`
- `docs/security.md`
- `docs/operations.md`
- `docs/migration-notes.md`
- `docs/domain-correctness.md`
- `docs/performance.md`

## Important Limitations

Results should be treated as candidate or possible low-altitude occurrences, not definitive legal infractions. Vertical datum assumptions, congested-area classification, and ADS-B quality filtering need further domain validation before stronger claims are made.
