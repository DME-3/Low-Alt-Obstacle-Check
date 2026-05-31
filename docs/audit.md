# ADS-B Pipeline Audit

Audit date: 2026-05-31

Branch: `refactor/adsb-pipeline-modernization`

This document records the baseline audit taken before the modernization changes.
Current branch behavior is described in `docs/architecture.md`,
`docs/operations.md`, and `docs/security.md`.

## Scope

This audit covers the production nightly ADS-B pipeline in `OSN_data_update.py`, the manual recovery tooling in `OSN_data_backfill.py` and `backfill_manifest.py`, repository structure, git history, data flow, database interactions, restart logic, configuration, secrets handling, logging, process lifecycle, failure modes, validation, and operational risks.

The baseline production cron entry was:

```cron
0 2 * * * cd /home/dimitri/obstaclecheck/ && /home/dimitri/obstaclecheck/.venv/bin/python /home/dimitri/obstaclecheck/OSN_data_update.py >> /home/dimitri/OSN_log.txt 2>&1
```

## Repository Structure At Audit Start

- `OSN_data_update.py`: production nightly entry point. At audit start, it performed all work at module import time.
- `OSN_data_backfill.py`: manual date-range recovery script. At audit start, it duplicated most production logic and contained hard-coded dates.
- `backfill_manifest.py`: manual manifest-population tool. It plans by default,
  targets test tables by default, and requires explicit confirmation before
  production writes.
- `resources/`: tracked static DEM, population raster, and obstacle CSV. Current size: about 1.2M.
- `OSN_pickles/`: ignored generated OpenSky pickles. Current local size: about 4.9G.
- `dataframes/`: ignored generated JSON dataframe output. Current local size: about 246M.
- `data_baseline/`: ignored comparison/baseline data. Current local size: about 142M.
- `reprocessing/`: ignored recovery/reprocessing artifacts. Current local size: about 77M.
- `*.json` secret files and `.ssh/` keys are present locally and ignored by `.gitignore`.

There is no package structure, test suite, lint configuration, or operational documentation beyond the short README.

## Git History And Churn

Recent history shows high churn in:

- `OSN_data_update.py`: repeated fixes for empty event dataframes, manifest updates, and `min_hgt` behavior.
- `OSN_data_backfill.py`: added later as a large duplicated manual script.
- `backfill_manifest.py`: support utility added for manifest backfills.
- `.gitignore`: repeated cleanup around generated artifacts and local tools.

Recurring patterns:

- Fixes are applied directly in the large production script.
- Backfill tooling copies production logic instead of sharing a tested pipeline module.
- Empty dataframe cases have needed multiple fixes, suggesting weak validation and no regression tests.
- Manifest handling was bolted on after data upload logic and still records each table independently.

## Architecture Overview

At audit start, the nightly flow was linear and monolithic:

1. Load MySQL and PythonAnywhere secrets at import time.
2. Compute the target day as two days before current server time.
3. Query OpenSky/Trino `state_vectors_data4`.
4. Build an ICAO list from returned rows.
5. Query `operational_status_data4` and merge status fields into state vectors.
6. Query `position_data4` and merge NIC values into the merged dataframe.
7. Write raw OpenSky query outputs to `OSN_pickles/`.
8. Add DEM ground elevation from `resources/Cologne_DEM_merged_from_LAS_25x25.tif`.
9. Add population density from `resources/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_Cologne.tif`.
10. Split aircraft trajectories by time gaps.
11. Compute along-track distances and obstacle/ground minimum heights.
12. Build main, obstacle-event, and ground-event dataframes.
13. Query the production manifest for the latest processed date.
14. Append all three dataframes to production tables.
15. Insert one manifest entry per table.
16. Reload the PythonAnywhere web app.

All stages live in one script, most state is global, and there is no safe callable pipeline boundary.

## Data Flow

Inputs:

- OpenSky Network historical ADS-B data through `pyopensky.trino.Trino`.
- Local DEM GeoTIFF.
- Local population GeoTIFF.
- Local obstacle CSV.
- Local secret JSON files.
- Local SSH private key.

Outputs:

- Pickles in `OSN_pickles/`.
- Rows appended to PythonAnywhere MySQL tables:
  - `main_data`
  - `inf_data`
  - `gndinf_data`
- Rows appended to `manifest`.
- PythonAnywhere web app reload API call.

Known test/staging table names exist in configuration, but production scripts write to production table names by default.

## Operational Risks

### Critical

- No single-run guard. Cron can start a second process while a previous run is still active.
- No max-runtime guard. If OpenSky, SSH, MySQL, raster IO, or upload blocks, cron leaves the Python process alive.
- Production writes are the default path. There is no dry-run mode and no explicit production opt-in.
- Upload is not atomic across `main_data`, `inf_data`, and `gndinf_data`.
- Manifest entries are inserted after each table append, so a partial upload can look partially successful.
- The PythonAnywhere reload request has no timeout.
- Trino queries have no explicit timeout or bounded retry policy.
- SQLAlchemy engines are not explicitly disposed.
- Raster files are opened without context managers and are not closed.
- Empty OpenSky days are not handled in the production script before building `icao24 IN (...)`.
- The manifest check happens after all expensive processing and pickle writing.
- The manifest check uses only `max(processed_date)`, which does not prove that the specific target day and all three related tables were successfully published.

### High

- Backfill scripts can write production data with hard-coded dates and no operator confirmation.
- Manual manifest backfill remains a powerful recovery tool; it now avoids SQLAlchemy engine echo logging and requires explicit execution/production confirmation.
- Secrets are loaded at import time and stored in module globals.
- Exceptions around manifest writes are swallowed with `print`, so operational failure can be hidden.
- `exit()` is called directly from the script, which complicates wrapping and testing.
- There is no structured logging, run ID, stage duration, row count summary, or predictable exit-code model.

### Medium

- Generated pickles accumulate without retention management and already consume about 4.9G locally.
- Repeated per-ICAO loops and repeated dataframe concatenation can increase runtime and memory pressure.
- Raster band data is read repeatedly per row via `dem.read(1)` / `pop_src.read(1)`.
- There is no schema validation before upload.
- There are no regression tests for empty dataframe cases, duplicate days, or production-write protection.

## Failure Modes

- Overlapping cron invocations can duplicate uploads or compete for memory/network resources.
- If Trino returns no state vector rows, `icao24 IN ()` can be produced for subsequent queries.
- If a network call hangs, cron redirection only captures output; it does not enforce a deadline or cleanup.
- If `main_data` upload succeeds but `inf_data` or `gndinf_data` fails, the site may see inconsistent data.
- If manifest insertion fails after upload, the next run may reprocess and append duplicates.
- If manifest reports a newer day but a specific older target date is missing, the production script exits incorrectly.
- If the web app reload call hangs, the pipeline can remain alive after successful uploads.
- If the DEM or population rasters fail to close, repeated runs can leak file descriptors.

## Database And Publishing Findings

- Production database is addressed through an SSH tunnel to PythonAnywhere MySQL.
- Connection strings are assembled by string concatenation from secrets.
- Writes use `DataFrame.to_sql(..., if_exists="append")` without explicit transactions across all related tables.
- Manifest updates reflect the metadata table each time, adding overhead and another failure point.
- The current manifest status model records table-level success but not an all-or-nothing publish batch.
- There is no pre-upload schema comparison against the destination tables.
- There is no row-count verification after upload.
- There is no duplicate check for the target date per table before uploading.
- Test table/database constants exist in secrets but production scripts do not default to them.

## Restart Logic Findings

- PythonAnywhere reload is performed unconditionally after the upload block, except when the script exits on manifest date.
- The reload call has no timeout.
- Non-200 responses print response content directly. This should avoid logging secrets; response bodies should be summarized.
- Reload should run only after validation and all related uploads succeed.

## Configuration And Secrets Findings

- `mysql_secrets.json`, `PYA_secrets.json`, `trino_secrets.json`, `OSN_secrets.json`, and `.ssh/` keys are ignored by git.
- No secret values were read or printed during this audit.
- Secrets are loaded from fixed relative paths, so the script depends on the current working directory.
- Runtime mode, target database, dry-run behavior, timeout budgets, lock path, and retention policy are not configurable.

## Logging And Observability Findings

- Operational logging is print-based.
- There is no run ID.
- There are no structured stage start/end records.
- There are no durations by stage.
- There is no consolidated row-count report.
- Exceptions are not consistently logged with tracebacks.
- Cron redirects all output to a single growing log file outside the repo.
- There is no log rotation policy in the repository.

## Process Lifecycle Findings

- No lock file or PID file exists.
- No stale-lock recovery exists.
- No signal handling exists.
- No top-level `try/finally` cleanup boundary exists.
- SSH tunnel context managers exist around individual DB sections, but the script opens two separate tunnels and does not explicitly dispose SQLAlchemy engines.
- Raster datasets are not context-managed.
- Network operations are not covered by an overall deadline.

## Data Validation Findings

Validation is currently implicit and late. Missing coverage includes:

- Required input columns.
- Expected output columns.
- Target schema compatibility.
- Duplicate keys or duplicate target-day rows.
- Null-rate thresholds for critical fields such as `gnd_elev`, `callsign`, `geoaltitude`, `lat`, and `lon`.
- Date-range consistency for all rows.
- Empty source day behavior.
- Row-count verification after upload.
- Manifest consistency across all related tables.

## Performance Findings

- Merging status and position data loops over each ICAO and repeatedly concatenates dataframes. This can be replaced with sorted `merge_asof(..., by="icao24")`.
- `dem.read(1)` and `pop_src.read(1)` are called inside row-wise functions. Reading the raster band once or using vectorized sampling would reduce IO.
- Along-track distance is computed with nested Python loops and dataframe `.loc` writes.
- Obstacle distance checks iterate point by point against every obstacle. This may be acceptable with the current small obstacle set but should be isolated and tested.
- Repeated dataframe copies and global mutation make memory behavior hard to reason about.

## Domain-Correctness Findings

- `GEOID_HEIGHT_M = 47` is added to ground and obstacle heights before comparing to ADS-B `geoaltitude`. This may be a vertical datum correction, but the script does not prove whether ADS-B geometric altitude, DEM elevation, obstacle elevations, `gnd_elev`, and `z` are all in compatible datums.
- Comments state `min_hgt` is referenced to the geoid, while ADS-B `geoaltitude` may be ellipsoidal depending on source semantics. This needs explicit documentation and regression tests.
- `pop_density > 2` is used as a congested-area proxy. This is not equivalent to a legal SERA congested-area determination.
- Current output should be described as candidate/possible low-altitude occurrences, not definitive legal infractions.
- ADS-B quality indicators are retrieved (`positionnac`, `sourceintegritylevel`, `nic`, `geometricverticalaccuracy` in source status) but are not used to filter or score events.
- `CPA_MARGIN_M`, `DIP_MARGIN_M`, and `N_MIN` are defined but not applied.
- EPSG:25832 is used for obstacle-distance calculations and should be retained unless a validated reason emerges.

## Security Findings

- Secret files and SSH keys are correctly ignored by git.
- Production scripts still rely on secrets in predictable local relative paths.
- Backfill tooling now defaults to planning/dry-run behavior, has target selection, and requires explicit production confirmation; keep those guarantees in future recovery tools.
- SQL construction for OpenSky queries is string-based, but values are internally generated from constants and timestamps. ICAO values come from prior query results and should still be treated carefully.
- MySQL table names come from secrets/config. They should be validated against an allowlist before use.
- Logging should avoid full HTTP response bodies and connection strings.
- No unsafe subprocess usage was found in the reviewed Python scripts.

## Bugs And Edge Cases

- `inf_result = pd.DataFrame` assigns the class, not an empty dataframe, before it is later overwritten. Current code usually overwrites it, but the pattern is fragile.
- Empty `inf_pt_df` can create empty grouped dataframes and then access expected columns. Prior fixes suggest this has broken before.
- Empty `svdata4_df` is handled in the backfill script but not in the production script.
- `max_date < two_days_ago.date()` can fail if `max_date` is `NULL` or a different type.
- `max(processed_date)` can skip missing dates before the max date.
- At audit start, `OSN_data_backfill.py` used `if True` in the upload gate, bypassing the manifest duplicate check.
- `backfill_manifest.py` inserts recovery `SUCCESS` rows with zero counts. It now skips existing manifest successes unless `--force-duplicate` is provided, but operators should still audit real table contents before executing it.

## Technical Debt

- The main script combines configuration, external IO, transformations, validation, upload, and reload.
- Backfill tooling duplicates production logic.
- No tests, no fixtures, and no dry-run contract exist.
- No deployment or rollback instructions exist.
- The production entry point cannot be imported safely because it executes immediately.
- The README still describes manual date entry even though the production script computes two days ago.

## Recommended Roadmap

1. Add a safe runtime wrapper around the production entry point:
   - CLI options
   - dry-run by default
   - production upload opt-in
   - lock file with stale-lock recovery
   - max-runtime guard
   - structured logging
   - predictable exit codes
2. Move upload, manifest, reload, locking, and validation into small testable modules.
3. Change publishing to build/validate first, then upload all related tables as a guarded publish batch.
4. Make the manifest check target-date and table-specific, and record success only after all tables verify.
5. Move expensive processing after an early manifest/idempotency check when publish mode is enabled.
6. Add focused tests for locking, dry-run, production protection, validation, empty days, and publish sequencing.
7. Add manual-tool safeguards for backfill and manifest scripts.
8. Improve performance by replacing per-ICAO merge/concat loops and context-managing raster access.
9. Document vertical datum assumptions, congested-area limitations, and event terminology.
10. Add retention management for generated pickles and dataframe artifacts.

