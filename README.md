# Low-Alt-Obstacle-Check

Production ADS-B data pipeline for the Low Alt Cologne site. The nightly job imports OpenSky ADS-B data, enriches it with terrain/population/obstacle data, identifies candidate low-altitude occurrences, publishes MySQL tables for the PythonAnywhere site, and reloads the site after a successful publish.

## General

This Python script is used to generate the data for the [Low Alt Cologne](https://www.lowaltcologne.org) website.

The process comprises the steps described below.

Sources of data other than OSN may be used, as long as the data format is preserved.

The script includes uploading the results to a MySQL server.

requirements.txt lists required packages for all scripts (clean-up needed since more than the packages strictly necessary is included).

### 1 - Importing OSN data

Requisite:
- Functionnal Trino connection

User has to enter the appropriate start and end dates for querying the OSN database.

The script connects to the Trino server, and queries data from 3 tables for the date corresponding to 2 days in the past. All results are merged in a single dataframe.

Results are saved as pickles.

### 2 - Adding DEM Ground Elevation

We now use a DEM obtained from processed LiDAR Airborne Surveys (LAS), available openly.

The ground elevation at each point is simply read from the DEM GeoTIFF.

### 3 - Adding population density

Population density is read and added from a Copernicus GeoTIFF. It will be used to approximate whether the aircraft position is in a congested area or not.

### 4 - Trajectory processing

Aircraft data is split into flights, and cumulative along-track distance information added (first point corresponds to distance = 0).

### 5 - Load obstacle information and check min height

Obstacle data is loaded. Obstacles are determined from processing of LAS data, which is more accurate than the AIP provided data. This way, we can also consider all obstacles from 60 m of height.

For each trajectory point, the highest obstacle within an alerting perimeter defined in the rules of the air is determined. We then determine the minimum height allowable at that point, depending on whether the area is congested or not.

Finally, for all points, we determine a "dip", i.e. the difference between the minimum height and the actual aircraft height.

This diagram illustrates the various heights and and position parameters used:

![diagram](https://github.com/DME-3/Low-Alt-Obstacle-Check/raw/main/LowAltCologne_Definitions.png)

### 6 - Process events

We find all "events" (possible infractions), for both obstacles and ground clearance. Separate dataframes are created for easier processing.

### 7 - Upload data

Results are added to a MySQL database, for use by the web application.

### Limitations

Known limitations:

- Some flights are split and appear as distinct flights in the data (distinct 'ref' identifiers), although the time difference does not exceed the detection threshold.
- Obstacles are considered punctual, their horizontal extent is not taken into account.

## Current Entry Point

```bash
/home/dimitri/obstaclecheck/.venv/bin/python /home/dimitri/obstaclecheck/OSN_data_update.py
```

The script is a thin cron-compatible wrapper around `lac_pipeline.nightly.main`. It defaults to dry-run mode. Production upload requires explicit flags.

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
