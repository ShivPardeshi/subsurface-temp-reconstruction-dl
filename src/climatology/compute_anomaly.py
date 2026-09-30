"""
Anomaly target computation module.
Evaluates 2-harmonic climatology and calculates temperature anomaly target:
  anomaly(x, y, d, t) = GLORYS_true(x, y, d, t) - Clim(x, y, d, doy(t))
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, Optional

OMEGA = 2.0 * np.pi / 365.25

def evaluate_climatology(
    coeffs: np.ndarray,
    day_of_year: Union[int, float, np.ndarray]
) -> np.ndarray:
    """
    Evaluates climatology temperature at day_of_year.

    Args:
        coeffs: Parameter array of shape (5, ...) where axis 0 has [a0, a1, b1, a2, b2]
        day_of_year: Integer or array of day-of-year (1 to 366)

    Returns:
        clim_temp: Array of shape (D, H, W) or (T, D, H, W)
    """
    doy = np.asarray(day_of_year, dtype=np.float64)

    # Coeffs shape: (5, D, H, W)
    a0 = coeffs[0]
    a1 = coeffs[1]
    b1 = coeffs[2]
    a2 = coeffs[3]
    b2 = coeffs[4]

    if doy.ndim == 0:
        c1 = np.cos(OMEGA * doy)
        s1 = np.sin(OMEGA * doy)
        c2 = np.cos(2.0 * OMEGA * doy)
        s2 = np.sin(2.0 * OMEGA * doy)
        clim = a0 + a1 * c1 + b1 * s1 + a2 * c2 + b2 * s2
    else:
        # Vectorized over time dimension
        c1 = np.cos(OMEGA * doy)[:, np.newaxis, np.newaxis, np.newaxis]
        s1 = np.sin(OMEGA * doy)[:, np.newaxis, np.newaxis, np.newaxis]
        c2 = np.cos(2.0 * OMEGA * doy)[:, np.newaxis, np.newaxis, np.newaxis]
        s2 = np.sin(2.0 * OMEGA * doy)[:, np.newaxis, np.newaxis, np.newaxis]

        clim = a0[np.newaxis, ...] + a1[np.newaxis, ...] * c1 + b1[np.newaxis, ...] * s1 + \
               a2[np.newaxis, ...] * c2 + b2[np.newaxis, ...] * s2

    return np.asarray(clim, dtype=np.float32)

def compute_temperature_anomaly(
    glorys_temp: np.ndarray,
    coeffs: np.ndarray,
    dates: pd.DatetimeIndex
) -> np.ndarray:
    """
    Computes anomaly target: GLORYS_temp - Climatology.

    Args:
        glorys_temp: Array of shape (T, 15, H, W)
        coeffs: Climatology coefficients array of shape (5, 15, H, W)
        dates: DatetimeIndex of length T

    Returns:
        anomaly: Array of shape (T, 15, H, W)
    """
    doy = dates.dayofyear.values
    clim = evaluate_climatology(coeffs, doy) # (T, 15, H, W)
    anomaly = (glorys_temp - clim).astype(np.float32)
    return anomaly
