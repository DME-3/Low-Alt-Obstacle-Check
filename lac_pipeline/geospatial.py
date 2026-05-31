from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import rasterio
from pyproj import Transformer


def add_projected_coordinates(
    frame: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    x_col: str = "etrs89_x",
    y_col: str = "etrs89_y",
    src_crs: int = 4326,
    dst_crs: int = 25832,
) -> pd.DataFrame:
    result = frame.copy()
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs, ys = transformer.transform(result[lon_col].tolist(), result[lat_col].tolist())
    result[x_col] = xs
    result[y_col] = ys
    return result


def add_ground_elevation(
    frame: pd.DataFrame,
    dem_path: str | Path,
    x_col: str = "etrs89_x",
    y_col: str = "etrs89_y",
    output_col: str = "gnd_elev",
) -> pd.DataFrame:
    result = frame.copy()
    coords = zip(result[x_col], result[y_col])
    result[output_col] = sample_raster_file(dem_path, coords, default_value=None)
    return result


def add_population_density(
    frame: pd.DataFrame,
    population_path: str | Path,
    lon_col: str = "lon",
    lat_col: str = "lat",
    output_col: str = "pop_density",
) -> pd.DataFrame:
    result = frame.copy()
    coords = zip(result[lon_col], result[lat_col])
    result[output_col] = sample_raster_file(population_path, coords, default_value=None)
    return result


def sample_raster_file(
    path: str | Path,
    coords: Iterable[tuple[float, float]],
    default_value: float | None = None,
) -> pd.Series:
    with rasterio.open(path) as src:
        return sample_raster_dataset(src, coords, default_value=default_value)


def sample_raster_dataset(
    dataset: rasterio.io.DatasetReader,
    coords: Iterable[tuple[float, float]],
    default_value: float | None = None,
) -> pd.Series:
    band = dataset.read(1)
    height, width = band.shape
    values = []

    for x, y in coords:
        try:
            row, col = dataset.index(float(x), float(y))
        except (TypeError, ValueError):
            values.append(default_value)
            continue

        if row < 0 or col < 0 or row >= height or col >= width:
            values.append(default_value)
        else:
            values.append(band[row, col])

    return pd.Series(values)

