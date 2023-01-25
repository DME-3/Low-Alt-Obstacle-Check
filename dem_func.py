# DEM Functions

import rasterio

def get_elev(array, coords, transformer, dem_src):
  coords_xy = WGS84_to_ETRS89(coords[1], coords[0], transformer)
  row, col = rasterio.transform.rowcol(dem_src.transform, coords_xy[0], coords_xy[1])
  elev = array[row, col]
  return elev

# CRS transform OK
def WGS84_to_ETRS89(lon, lat, transformer):
  x2,y2 = transformer.transform(lon, lat)
  return x2, y2