"""
Wind-derived physical features module.
Computes:
1. Wind stress curl: curl = d(tau_y)/dx - d(tau_x)/dy
2. Wind-mixing energy: E_mix = |v|^3
"""

import numpy as np
from typing import Tuple

# Physical constants
AIR_DENSITY = 1.225         # kg/m^3
DRAG_COEFF = 1.3e-3         # Standard constant drag coefficient (simplification)
EARTH_RADIUS = 6.371e6      # meters

def compute_wind_stress_and_curl(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes wind stress components (tau_x, tau_y) and wind stress curl.

    Args:
        wind_u: Zonal wind component at 10m (m/s), shape (H, W)
        wind_v: Meridional wind component at 10m (m/s), shape (H, W)
        lat: 1D array of latitudes (degrees), shape (H,)
        lon: 1D array of longitudes (degrees), shape (W,)

    Returns:
        tau_x: Zonal wind stress (N/m^2)
        tau_y: Meridional wind stress (N/m^2)
        curl: Wind stress curl (N/m^3)
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lon_mesh_rad, lat_mesh_rad = np.meshgrid(lon_rad, lat_rad)

    # Wind speed
    wind_speed = np.sqrt(np.square(wind_u) + np.square(wind_v))

    # Wind stress: tau = rho_air * C_d * |wind| * wind_vec
    tau_x = AIR_DENSITY * DRAG_COEFF * wind_speed * wind_u
    tau_y = AIR_DENSITY * DRAG_COEFF * wind_speed * wind_v

    # Metric step sizes
    d_lat_rad = np.gradient(lat_rad)
    d_lon_rad = np.gradient(lon_rad)

    dy = EARTH_RADIUS * d_lat_rad[:, np.newaxis]
    dx = EARTH_RADIUS * np.cos(lat_mesh_rad) * d_lon_rad[np.newaxis, :]

    # Curl: d(tau_y)/dx - d(tau_x)/dy
    d_tau_y_dy, d_tau_y_dx = np.gradient(tau_y)
    d_tau_x_dy, d_tau_x_dx = np.gradient(tau_x)

    d_tau_y_dx = d_tau_y_dx / dx
    d_tau_x_dy = d_tau_x_dy / dy

    curl = (d_tau_y_dx - d_tau_x_dy).astype(np.float32)
    curl = np.nan_to_num(curl, nan=0.0, posinf=0.0, neginf=0.0)

    return tau_x.astype(np.float32), tau_y.astype(np.float32), curl

def compute_wind_mixing_energy(wind_u: np.ndarray, wind_v: np.ndarray) -> np.ndarray:
    """
    Computes wind-mixing energy proportional to turbulent kinetic energy: |wind|^3.
    """
    speed_sq = np.square(wind_u) + np.square(wind_v)
    energy = (speed_sq * np.sqrt(speed_sq)).astype(np.float32)
    return np.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
