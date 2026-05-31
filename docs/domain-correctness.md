# Domain Correctness Notes

## Event Terminology

Pipeline outputs should be described as candidate events, possible events, or low-altitude occurrences. They are not definitive legal infractions because the pipeline does not model all legal exceptions, authorizations, aerodromes, heliports, takeoff/landing context, or full operational circumstances.

## Vertical Datum Assumptions

The current calculation preserves the existing behavior:

```text
dip = min_hgt - geoaltitude
```

`min_hgt` is built from `GEOID_HEIGHT_M`, DEM ground elevation, obstacle height/elevation, and the SERA clearance delta. The code assumes this makes the comparison compatible with ADS-B `geoaltitude`, but this has not been proven from source metadata alone.

Risk to keep visible:

- ADS-B `geoaltitude` may be geometric/ellipsoidal depending on source semantics.
- DEM and obstacle elevations may be orthometric or locally processed heights.
- `GEOID_HEIGHT_M = 47` is a coarse Cologne-area correction.

No datum-changing refactor should be made without fixture-backed before/after comparisons.

## Congested-Area Classification

The current heuristic remains:

```text
congested = pop_density > 2
```

This is only a proxy and should not be represented as a legal SERA congested-area classification. Future work should either use an authoritative land-use/legal classification source or continue labeling this as an approximation.

## ADS-B Quality Indicators

The pipeline carries quality-related fields such as `positionnac`, `sourceintegritylevel`, `nic`, and other operational status data, but it does not yet use them to filter or weight candidate events.

Recommended next step:

1. Keep all candidate rows for site compatibility.
2. Add a separate quality flag or confidence tier.
3. Document any threshold before using it to suppress rows.

## Trajectory Splitting

The trajectory split threshold remains `TIME_BETWEEN_TRAJS = 30` seconds. The implementation now computes previous timestamps per ICAO instead of relying on a global dataframe shift, which better matches the intended per-aircraft split logic.

## Unused Margins

`CPA_MARGIN_M`, `DIP_MARGIN_M`, and `N_MIN` remain defined but not applied. They should be either introduced intentionally with tests and migration notes or removed in a separate cleanup after confirming the site does not depend on their implied semantics.

