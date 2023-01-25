# DEM Functions

import rasterio

def get_elev(array, coords, transformer, dem_file):
  coords_xy = WGS84_to_ETRS89(coords[1], coords[0], transformer)
  row, col = get_row_col(coords_xy, dem_file)
  elev = array[row, col]
  return elev

def get_row_col(pos, dem_file):   # this is correct
  with rasterio.open(dem_file) as src:
      row, col = rasterio.transform.rowcol(src.transform, pos[0], pos[1])
  return row, col

# CRS transform OK
def WGS84_to_ETRS89(lon, lat, transformer):
  x2,y2 = transformer.transform(lon, lat)
  return x2, y2