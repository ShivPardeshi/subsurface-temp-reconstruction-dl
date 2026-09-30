"""
Land/Ocean masking and bathymetry derivation module.
Uses GEBCO elevation dataset to generate:
- Channel #20: Log-scaled Bathymetry
- Channel #21: Binary Land/Ocean Mask
"""

import os
import numpy as np
import xarray as xr
from typing import Tuple

from src.data.harmonize.regrid import regrid_2d_array
from src.utils.io_utils import open_netcdf_safe
from src.utils.logging_config import setup_logger

logger = setup_logger("landmask")

LAND_FILL_VALUE = 0.0

def load_and_regrid_gebco(
    gebco_path: str,
    target_lat: np.ndarray,
    target_lon: np.ndarray
) -> np.ndarray:
    """
    Loads GEBCO elevation NetCDF, regrids onto target (lat, lon), and returns regridded elevation in meters.
    """
    if not os.path.exists(gebco_path):
        raise FileNotFoundError(f"GEBCO file not found at: {gebco_path}")

    logger.info(f"Loading GEBCO elevation from {gebco_path}...")
    ds = open_netcdf_safe(gebco_path)

    elev_var = "elevation" if "elevation" in ds else list(ds.data_vars.keys())[0]
    lat_var = "lat" if "lat" in ds.coords else "latitude"
    lon_var = "lon" if "lon" in ds.coords else "longitude"

    src_lat = ds[lat_var].values
    src_lon = ds[lon_var].values

    # Determine subset bounding box with margin
    lat_min_q = float(target_lat.min()) - 0.5
    lat_max_q = float(target_lat.max()) + 0.5
    lon_min_q = float(target_lon.min()) - 0.5
    lon_max_q = float(target_lon.max()) + 0.5

    # Latitude slice indices
    if src_lat[0] < src_lat[-1]:
        lat_idx = np.where((src_lat >= lat_min_q) & (src_lat <= lat_max_q))[0]
    else:
        lat_idx = np.where((src_lat <= lat_max_q) & (src_lat >= lat_min_q))[0]

    # Longitude slice indices
    if src_lon[0] < src_lon[-1]:
        lon_idx = np.where((src_lon >= lon_min_q) & (src_lon <= lon_max_q))[0]
    else:
        lon_idx = np.where((src_lon <= lon_max_q) & (src_lon >= lon_min_q))[0]

    # Stride downsample if resolution is ultra-dense (e.g. 15 arcseconds)
    # Target resolution is 0.25° (~27 km), GEBCO is ~0.004° (~450m). Stride of 4 (~0.016°) provides pristine accuracy with 16x speedup.
    stride = 4
    lat_idx_strided = lat_idx[::stride]
    lon_idx_strided = lon_idx[::stride]

    subset_lat = src_lat[lat_idx_strided]
    subset_lon = src_lon[lon_idx_strided]
    subset_elev = ds[elev_var].isel({lat_var: lat_idx_strided, lon_var: lon_idx_strided}).values

    ds.close()

    logger.info(f"Regridding GEBCO (subset shape {subset_elev.shape}) to target grid ({len(target_lat)}, {len(target_lon)})...")
    regridded_elev = regrid_2d_array(
        subset_elev,
        subset_lat,
        subset_lon,
        target_lat,
        target_lon,
        method="linear",
        fill_value=0.0
    )
    return regridded_elev

def derive_land_ocean_mask_and_bathymetry(
    elevation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derives:
    1. Log-scaled Bathymetry (Channel #20): log10(max(1.0, -elevation)) for ocean, 0 for land.
    2. Land/Ocean Mask (Channel #21): 1 for Ocean (elevation < 0), 0 for Land (elevation >= 0).

    Args:
        elevation: 2D array of topography/bathymetry elevation (meters)

    Returns:
        log_bathymetry (np.ndarray): Shape (H, W)
        ocean_mask (np.ndarray): Shape (H, W), dtype float32 (1=ocean, 0=land)
    """
    ocean_mask = (elevation < 0.0).astype(np.float32)

    # Ocean depth is positive depth below surface: -elevation
    ocean_depth = np.maximum(0.0, -elevation)
    # Log scale: log10(1 + depth) so that 0 depth -> 0
    log_bathymetry = np.where(ocean_mask > 0.5, np.log10(1.0 + ocean_depth), 0.0).astype(np.float32)

    return log_bathymetry, ocean_mask
