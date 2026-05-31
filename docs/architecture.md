# Architecture

## Current Shape

`OSN_data_update.py` is now a thin cron-compatible wrapper around `lac_pipeline.nightly.main`. The highest-risk operational concerns live in small helper modules under `lac_pipeline/`:

- `lac_pipeline.nightly`: orchestrates the nightly pipeline as an import-safe callable entry point.
- `lac_pipeline.runtime`: CLI parsing, structured logging, run IDs, single-run lock, stale-lock recovery, max-runtime guard, stage timing, bounded retries.
- `lac_pipeline.opensky`: OpenSky query-window construction, ICAO filter validation, query string builders, and bounded Trino fetch retries.
- `lac_pipeline.publishing`: publish-target selection, production-write protection, SSH tunnel engine lifecycle, manifest checks, transactional publish sequencing, row-count verification, PythonAnywhere reload timeout.
- `lac_pipeline.validation`: required-column checks, date-range checks, duplicate event-reference checks, and validation summaries.
- `lac_pipeline.events`: candidate obstacle/ground event table construction with stable empty-frame schemas.
- `lac_pipeline.transforms`: shared ADS-B merge helpers that avoid repeated per-ICAO dataframe concatenation.
- `lac_pipeline.geospatial`: context-managed raster sampling and coordinate projection helpers.
- `lac_pipeline.trajectory`: trajectory splitting and along-track distance helpers.
- `lac_pipeline.obstacles`: obstacle/ground clearance calculations and candidate flags.

The production wrapper is intentionally tiny; ADS-B orchestration and domain transformations now live under `lac_pipeline/`.

## Nightly Flow

1. Parse CLI flags.
2. Configure logging and run ID.
3. Enforce publish protection.
4. Acquire a single-run lock.
5. Install the process max-runtime guard.
6. Compute the target processing date as a Europe/Paris local calendar day.
7. If publishing, check manifest success for the exact target date before expensive OpenSky work.
8. Build validated OpenSky/Trino queries and fetch them with bounded retry attempts.
9. If the source day is empty, record a zero-row success only when explicitly publishing.
10. Build enriched ADS-B dataframes and candidate event tables.
11. Validate output frames.
12. Dry-run exits before database writes.
13. Publish mode uploads main, obstacle-event, and ground-event tables inside one transaction scope.
14. Insert manifest rows only after upload verification succeeds.
15. Reload PythonAnywhere only after successful publish, unless `--skip-reload` is set.

## Preserved Compatibility

No database schema migration is introduced. Existing table names and columns are preserved. Production table names remain:

- `main_data`
- `inf_data`
- `gndinf_data`

Test table names remain:

- `main_data_test`
- `inf_data_test`
- `gndinf_data_test`

## Remaining Refactor Work

The next architecture step should compare end-to-end outputs against baseline fixtures, then isolate the remaining orchestration in `lac_pipeline.nightly` into smaller service functions without changing event results.

