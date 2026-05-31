import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from lac_pipeline.geospatial import (
    add_ground_elevation,
    add_population_density,
    add_projected_coordinates,
    sample_raster_file,
)


def test_sample_raster_file_returns_values_and_default_for_out_of_bounds(tmp_path):
    raster_path = _write_test_raster(tmp_path)

    values = sample_raster_file(
        raster_path,
        coords=[(0.5, 1.5), (1.5, 0.5), (99.0, 99.0)],
        default_value=None,
    )

    assert values.iloc[0] == 1
    assert values.iloc[1] == 4
    assert pd.isna(values.iloc[2])


def test_add_projected_coordinates_can_use_identity_transform():
    frame = pd.DataFrame({"lon": [6.9], "lat": [50.9]})

    result = add_projected_coordinates(frame, src_crs=4326, dst_crs=4326)

    assert result["etrs89_x"].iloc[0] == 6.9
    assert result["etrs89_y"].iloc[0] == 50.9


def test_add_ground_elevation_uses_context_managed_raster(tmp_path):
    raster_path = _write_test_raster(tmp_path)
    frame = pd.DataFrame({"etrs89_x": [0.5], "etrs89_y": [1.5]})

    result = add_ground_elevation(frame, raster_path)

    assert result["gnd_elev"].tolist() == [1]


def test_add_population_density_samples_lon_lat_raster(tmp_path):
    raster_path = _write_test_raster(tmp_path)
    frame = pd.DataFrame({"lon": [1.5], "lat": [0.5]})

    result = add_population_density(frame, raster_path)

    assert result["pop_density"].tolist() == [4]


def _write_test_raster(tmp_path):
    raster_path = tmp_path / "test.tif"
    values = np.array([[1, 2], [3, 4]], dtype=np.float32)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
    ) as dataset:
        dataset.write(values, 1)
    return raster_path

