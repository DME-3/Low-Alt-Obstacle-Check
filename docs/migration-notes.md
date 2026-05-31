# Migration Notes

## Database Schema

No schema migration is required.

The site-facing table names and expected columns are preserved. The safer publishing path appends to the same tables and writes the existing `manifest` table after upload verification.

## Behavior Change

`OSN_data_update.py` now defaults to dry-run mode. The existing cron entry will no longer publish unless it is updated with explicit publish flags.

Required production flags:

```bash
--publish --target prod --confirm-production
```

This is intentional: production writes are now opt-in and protected.

## Deployment Steps

1. Deploy this branch.
2. Install updated requirements if needed:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python -m pip install -r requirements.txt
```

3. Run tests:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python -m pytest
```

4. Run lint:

```bash
/home/dimitri/obstaclecheck/.venv/bin/python -m ruff check OSN_data_update.py lac_pipeline tests
```

5. Run a dry-run for a known date.
6. Run a test-table upload with `--target test --skip-reload`.
7. Update cron only after the test upload is verified.

## Compatibility Notes

- Dry-runs no longer require MySQL/PythonAnywhere secrets.
- Publishing still requires `mysql_secrets.json`, `PYA_secrets.json`, and `.ssh/id_ed25519`.
- PythonAnywhere reload happens only after successful publish and is skipped for test upload when `--skip-reload` is passed.
- Empty OpenSky days are handled without building invalid `IN ()` queries.

