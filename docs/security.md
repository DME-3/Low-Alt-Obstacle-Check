# Security Review

## Secrets

Secret values were not read, printed, rewritten, or committed during this work.

The following local secret/key paths are ignored by git:

- `mysql_secrets.json`
- `PYA_secrets.json`
- `trino_secrets.json`
- `OSN_secrets.json`
- `.ssh/id_ed25519`
- `.ssh/id_ed25519.pub`
- `.ssh/known_hosts`

Nightly dry-runs do not load MySQL/PythonAnywhere secrets or the SSH key. Those files are loaded only when `--publish` is set.

## Production Write Protection

`OSN_data_update.py` defaults to dry-run mode. Database writes require `--publish`.

Production writes require all of:

```bash
--publish --target prod --confirm-production
```

Test-table uploads use:

```bash
--publish --target test
```

Manual tools also default to dry-run and require explicit execution plus production confirmation.

## SQL And Table Safety

The publishing layer validates configured database and table identifiers with an allowlist-style identifier check before using them in SQL. Data values are passed through SQLAlchemy/pandas APIs rather than interpolated into MySQL statements.

OpenSky/Trino query strings still interpolate internally generated timestamps, bounds, and ICAO values returned by OpenSky. Future work should isolate query construction and add tests around generated SQL.

## Network Operations

PythonAnywhere reload now uses an HTTP timeout. SSH tunnel timeouts remain configurable through CLI flags. Trino query exceptions are retried with bounded attempts; the process-level max-runtime guard is the hard stop for network calls that hang below the Python API.

## Logging

Logs include timestamps and run IDs. Secrets and connection strings are not intentionally logged. PythonAnywhere reload logs the status code, not the response body.

## Remaining Risks

- The OpenSky/Trino library does not currently have a per-query timeout wired through this code path.
- The monolithic transformation code still performs most work at module scope.
- Manual backfill tools are safer but still duplicate production logic and write directly to production when explicitly confirmed.
- The PythonAnywhere MySQL transaction semantics depend on the destination tables using a transactional engine.

