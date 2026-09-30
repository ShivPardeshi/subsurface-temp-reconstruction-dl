"""
Regridding module for OceanEmbed.
Standardized interpolation mapping any source grid to the target 0.25° domain.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from typing import Union, Tuple, Optional

_GRID_WEIGHTS_CACHE = {}

def regrid_2d_array(
    src_data: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    method: str = "linear",
    fill_value: float = np.nan
) -> np.ndarray:
    """
    Regrids a 2D numpy array from (src_lat, src_lon) to (target_lat, target_lon).

    Args:
        src_data: 2D numpy array of shape (len(src_lat), len(src_lon))
        src_lat: 1D array of source latitudes
        src_lon: 1D array of source longitudes
        target_lat: 1D array of target latitudes
        target_lon: 1D array of target longitudes
        method: Interpolation method ('linear' for bilinear, 'nearest' for nearest-neighbor)
        fill_value: Value used for points outside source convex hull

    Returns:
        regridded_data: 2D numpy array of shape (len(target_lat), len(target_lon))
    """
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)

    # Grid cache key for fast bilinear interpolation
    cache_key = (
        len(src_lat), round(float(src_lat[0]), 4), round(float(src_lat[-1]), 4),
        len(src_lon), round(float(src_lon[0]), 4), round(float(src_lon[-1]), 4),
        len(target_lat), round(float(target_lat[0]), 4), round(float(target_lat[-1]), 4),
        len(target_lon), round(float(target_lon[0]), 4), round(float(target_lon[-1]), 4)
    )

    if method == "linear":
        if cache_key not in _GRID_WEIGHTS_CACHE:
            lat_sort_idx = np.argsort(src_lat)
            lon_sort_idx = np.argsort(src_lon)
            sorted_lat = src_lat[lat_sort_idx]
            sorted_lon = src_lon[lon_sort_idx]

            lat_idx = np.clip(np.searchsorted(sorted_lat, target_lat) - 1, 0, len(sorted_lat) - 2)
            lat_span = sorted_lat[lat_idx + 1] - sorted_lat[lat_idx]
            lat_span = np.where(lat_span == 0, 1e-6, lat_span)
            lat_frac = np.clip((target_lat - sorted_lat[lat_idx]) / lat_span, 0.0, 1.0)

            lon_idx = np.clip(np.searchsorted(sorted_lon, target_lon) - 1, 0, len(sorted_lon) - 2)
            lon_span = sorted_lon[lon_idx + 1] - sorted_lon[lon_idx]
            lon_span = np.where(lon_span == 0, 1e-6, lon_span)
            lon_frac = np.clip((target_lon - sorted_lon[lon_idx]) / lon_span, 0.0, 1.0)

            w00 = ((1.0 - lat_frac[:, None]) * (1.0 - lon_frac[None, :])).astype(np.float32)
            w10 = (lat_frac[:, None] * (1.0 - lon_frac[None, :])).astype(np.float32)
            w01 = ((1.0 - lat_frac[:, None]) * lon_frac[None, :]).astype(np.float32)
            w11 = (lat_frac[:, None] * lon_frac[None, :]).astype(np.float32)

            _GRID_WEIGHTS_CACHE[cache_key] = (lat_sort_idx, lon_sort_idx, lat_idx, lon_idx, w00, w10, w01, w11)

        lat_sort_idx, lon_sort_idx, lat_idx, lon_idx, w00, w10, w01, w11 = _GRID_WEIGHTS_CACHE[cache_key]
        sorted_src_data = src_data[lat_sort_idx, :][:, lon_sort_idx]

        nan_mask = np.isnan(sorted_src_data)
        if np.any(nan_mask):
            clean_src_data = np.where(nan_mask, 0.0, sorted_src_data).astype(np.float32)
            weight_mask = np.where(nan_mask, 0.0, 1.0).astype(np.float32)

            val_interp = (
                w00 * clean_src_data[lat_idx[:, None], lon_idx[None, :]] +
                w10 * clean_src_data[(lat_idx + 1)[:, None], lon_idx[None, :]] +
                w01 * clean_src_data[lat_idx[:, None], (lon_idx + 1)[None, :]] +
                w11 * clean_src_data[(lat_idx + 1)[:, None], (lon_idx + 1)[None, :]]
            )
            weight_interp = (
                w00 * weight_mask[lat_idx[:, None], lon_idx[None, :]] +
                w10 * weight_mask[(lat_idx + 1)[:, None], lon_idx[None, :]] +
                w01 * weight_mask[lat_idx[:, None], (lon_idx + 1)[None, :]] +
                w11 * weight_mask[(lat_idx + 1)[:, None], (lon_idx + 1)[None, :]]
            )
            regridded = np.where(weight_interp > 0.5, val_interp / np.maximum(weight_interp, 1e-6), fill_value)
        else:
            sorted_src_data = sorted_src_data.astype(np.float32)
            regridded = (
                w00 * sorted_src_data[lat_idx[:, None], lon_idx[None, :]] +
                w10 * sorted_src_data[(lat_idx + 1)[:, None], lon_idx[None, :]] +
                w01 * sorted_src_data[lat_idx[:, None], (lon_idx + 1)[None, :]] +
                w11 * sorted_src_data[(lat_idx + 1)[:, None], (lon_idx + 1)[None, :]]
            )
        return np.asarray(regridded, dtype=np.float32)

    # Fallback to Scipy RegularGridInterpolator for non-linear methods
    lat_sort_idx = np.argsort(src_lat)
    lon_sort_idx = np.argsort(src_lon)
    sorted_src_lat = src_lat[lat_sort_idx]
    sorted_src_lon = src_lon[lon_sort_idx]
    sorted_src_data = src_data[lat_sort_idx, :][:, lon_sort_idx]

    interp_func = RegularGridInterpolator(
        (sorted_src_lat, sorted_src_lon),
        sorted_src_data,
        method=method,
        bounds_error=False,
        fill_value=fill_value
    )
    lon_mesh, lat_mesh = np.meshgrid(target_lon, target_lat)
    query_points = np.stack([lat_mesh.ravel(), lon_mesh.ravel()], axis=-1)
    regridded = interp_func(query_points).reshape(len(target_lat), len(target_lon))
    return np.asarray(regridded, dtype=np.float32)

def regrid_dataset_to_target(
    ds: xr.Dataset,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    method: str = "linear"
) -> xr.Dataset:
    """
    Regrids all data variables in an xarray Dataset onto the target grid.
    """
    regridded_vars = {}
    src_lat = ds[lat_dim].values
    src_lon = ds[lon_dim].values

    for var_name, data_array in ds.data_vars.items():
        dims = data_array.dims
        data = data_array.values

        if lat_dim in dims and lon_dim in dims:
            lat_axis = dims.index(lat_dim)
            lon_axis = dims.index(lon_dim)

            if len(dims) == 2:
                # 2D array
                if lat_axis > lon_axis:
                    data = data.T
                arr_out = regrid_2d_array(data, src_lat, src_lon, target_lat, target_lon, method=method)
                out_dims = ("lat", "lon")
                regridded_vars[var_name] = (out_dims, arr_out)

            elif len(dims) == 3:
                # 3D array (e.g. time, lat, lon)
                other_dim = [d for d in dims if d not in (lat_dim, lon_dim)][0]
                n_other = data.shape[dims.index(other_dim)]
                outs = []
                for i in range(n_other):
                    slice_2d = np.take(data, i, axis=dims.index(other_dim))
                    if lat_axis > lon_axis:
                        slice_2d = slice_2d.T
                    outs.append(regrid_2d_array(slice_2d, src_lat, src_lon, target_lat, target_lon, method=method))
                arr_out = np.stack(outs, axis=0)
                out_dims = (other_dim, "lat", "lon")
                regridded_vars[var_name] = (out_dims, arr_out)

    coords = {"lat": target_lat, "lon": target_lon}
    for c in ds.coords:
        if c not in (lat_dim, lon_dim):
            coords[c] = ds[c]

    return xr.Dataset(regridded_vars, coords=coords, attrs=ds.attrs)
