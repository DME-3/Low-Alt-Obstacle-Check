# Performance Notes

## Improvements Made

The two ADS-B enrichment merges no longer loop over every ICAO and repeatedly concatenate dataframes. They now use a shared `merge_asof_by_icao` helper that performs a single `pandas.merge_asof(..., by="icao24")` operation and returns rows sorted by ICAO/time for downstream trajectory logic.

The candidate-event builders no longer use dataframe groupby/apply to select representative rows. They use group index selection (`idxmin` / `idxmax`) and always return schema-compatible empty frames.

## Remaining Hotspots

- Raster lookup still uses row-wise `apply` and reads raster band data repeatedly.
- Coordinate transformation still uses row-wise `apply`.
- Along-track distance is still computed with nested Python loops and dataframe `.loc` writes.
- Obstacle proximity is still point-by-point against all obstacles.
- OpenSky query result pickle retention is not enforced automatically.

## Recommended Next Steps

1. Use rasterio `sample()` or read raster arrays once per run.
2. Vectorize coordinate transforms with `Transformer.transform` over arrays.
3. Replace along-track distance loops with grouped vectorized calculations where possible.
4. Add a retention command for `OSN_pickles/` before enabling automatic cleanup.
5. Compare event output against baseline JSON before deeper performance changes.

