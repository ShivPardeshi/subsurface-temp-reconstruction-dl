"""
Harmonic Climatology Fitting Module.
Fits 2-harmonic annual and semi-annual seasonal cycle per grid cell and depth level:
  Clim(x, y, d, doy) = a0 + a1*cos(2pi*doy/365) + b1*sin(2pi*doy/365) + a2*cos(4pi*doy/365) + b2*sin(4pi*doy/365)

STRICT ANTI-LEAKAGE REQUIREMENT:
Climatology is fit EXCLUSIVELY on training years. Validation and test periods are forbidden.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple, Dict, Any, Optional

from src.utils.grid import get_target_grid, get_depth_levels, load_domain_config
from src.utils.io_utils import open_netcdf_safe
from src.utils.logging_config import setup_logger

logger = setup_logger("fit_climatology")

OMEGA = 2.0 * np.pi / 365.25

def construct_harmonic_design_matrix(day_of_year: np.ndarray) -> np.ndarray:
    """
    Constructs the 2-harmonic design matrix X of shape (N, 5):
    [1, cos(omega*t), sin(omega*t), cos(2*omega*t), sin(2*omega*t)]
    """
    doy = np.asarray(day_of_year, dtype=np.float64)
    c1 = np.cos(OMEGA * doy)
    s1 = np.sin(OMEGA * doy)
    c2 = np.cos(2.0 * OMEGA * doy)
    s2 = np.sin(2.0 * OMEGA * doy)
    ones = np.ones_like(doy)

    X = np.stack([ones, c1, s1, c2, s2], axis=-1) # (N, 5)
    return X

def fit_harmonic_climatology(
    temp_series: np.ndarray,
    dates: pd.DatetimeIndex,
    training_end_date: Optional[str] = None
) -> np.ndarray:
    """
    Fits 2-harmonic regression using Ordinary Least Squares across time dimension (axis 0).

    Args:
        temp_series: Array of shape (T, D, H, W) or (T,) of temperatures
        dates: DatetimeIndex of length T
        training_end_date: Optional cutoff date. If any date > training_end_date, raises ValueError.

    Returns:
        coeffs: Array of shape (5, D, H, W) or (5,) with parameters [a0, a1, b1, a2, b2]
    """
    # Strict anti-leakage verification
    if training_end_date is not None:
        max_allowed_dt = pd.to_datetime(training_end_date)
        if (dates > max_allowed_dt).any():
            raise ValueError(
                f"Data leakage detected! Climatology fitting received dates up to {dates.max()}, "
                f"which exceeds the configured training cutoff {training_end_date}."
            )

    doy = dates.dayofyear.values
    X = construct_harmonic_design_matrix(doy) # (T, 5)

    # Solve OLS: beta = (X^T X)^-1 X^T Y
    # Shape of temp_series: (T, ...)
    orig_shape = temp_series.shape
    T = orig_shape[0]
    spatial_shape = orig_shape[1:]

    # Reshape spatial dims to 1D
    y_flat = temp_series.reshape(T, -1) # (T, N_points)

    # Handle NaNs: compute least squares on valid points
    coeffs_flat = np.zeros((5, y_flat.shape[1]), dtype=np.float32)

    # Normal equation matrix
    XtX = X.T @ X # (5, 5)
    XtX_inv = np.linalg.pinv(XtX) # (5, 5)

    XtY = X.T @ y_flat # (5, N_points)
    coeffs_flat = (XtX_inv @ XtY).astype(np.float32) # (5, N_points)

    coeffs = coeffs_flat.reshape((5,) + spatial_shape)
    return coeffs

def fit_climatology_from_glorys(
    glorys_path: str,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_depths: np.ndarray,
    training_end_date: Optional[str] = None,
    output_nc_path: Optional[str] = None
) -> xr.Dataset:
    """
    Loads GLORYS 3D subsurface temperature from training period, regrids onto target (lat, lon, depth),
    fits 2-harmonic climatology per grid cell and depth, and saves coefficients.
    """
    logger.info(f"Fitting harmonic climatology from GLORYS: {glorys_path}...")
    ds = open_netcdf_safe(glorys_path)

    time_vals = pd.to_datetime(ds["time"].values)
    logger.info(f"GLORYS temporal range: {time_vals.min()} to {time_vals.max()} ({len(time_vals)} steps)")

    # Anti-leakage slice
    if training_end_date is not None:
        max_dt = pd.to_datetime(training_end_date)
        train_mask = time_vals <= max_dt
        if not np.any(train_mask):
            raise ValueError(f"No GLORYS observations before training cutoff {training_end_date}")
        ds = ds.isel(time=np.where(train_mask)[0])
        time_vals = pd.to_datetime(ds["time"].values)
        logger.info(f"Restricted climatology fit to training window: {time_vals.min()} to {time_vals.max()}")

    temp_var = "thetao" if "thetao" in ds else list(ds.data_vars.keys())[0]
    depth_var = "depth" if "depth" in ds.coords else list(ds.coords.keys())[1]
    lat_var = "latitude" if "latitude" in ds.coords else "lat"
    lon_var = "longitude" if "longitude" in ds.coords else "lon"

    # Interpolate vertical depth to the 15 canonical depths
    logger.info(f"Interpolating GLORYS temperature to {len(target_depths)} canonical depths...")
    ds_depth = ds[[temp_var]].interp({depth_var: target_depths}, method="linear")

    # Regrid spatially to target grid
    from src.data.harmonize.regrid import regrid_2d_array
    src_lat = ds_depth[lat_var].values
    src_lon = ds_depth[lon_var].values
    raw_temp = ds_depth[temp_var].values # (T, D, src_H, src_W)

    T, D = raw_temp.shape[0], raw_temp.shape[1]
    H, W = len(target_lat), len(target_lon)

    regridded_temp = np.zeros((T, D, H, W), dtype=np.float32)
    for t_i in range(T):
        for d_i in range(D):
            slice_2d = raw_temp[t_i, d_i]
            regridded_temp[t_i, d_i] = regrid_2d_array(slice_2d, src_lat, src_lon, target_lat, target_lon)

    ds.close()

    # Fit 2-harmonic coefficients
    logger.info("Solving ordinary least squares for 2-harmonic seasonal cycle...")
    coeffs = fit_harmonic_climatology(regridded_temp, time_vals, training_end_date=training_end_date)

    ds_coeffs = xr.Dataset(
        data_vars={
            "coefficients": (("param", "depth", "lat", "lon"), coeffs)
        },
        coords={
            "param": ["a0", "a1", "b1", "a2", "b2"],
            "depth": target_depths,
            "lat": target_lat,
            "lon": target_lon
        },
        attrs={
            "title": "OceanEmbed 2-Harmonic Seasonal Climatology Coefficients",
            "formula": "Clim(x,y,d,doy) = a0 + a1*cos(w*doy) + b1*sin(w*doy) + a2*cos(2w*doy) + b2*sin(2w*doy)",
            "training_period_end": str(training_end_date) if training_end_date else str(time_vals.max())
        }
    )

    if output_nc_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_nc_path)), exist_ok=True)
        ds_coeffs.to_netcdf(output_nc_path)
        logger.info(f"Climatology coefficients saved to: {output_nc_path}")

    return ds_coeffs
