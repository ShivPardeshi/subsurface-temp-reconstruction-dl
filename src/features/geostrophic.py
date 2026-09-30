"""
Geostrophic current computation module.
Computes geostrophic velocity components (u_g, v_g) from Sea Surface Height (SSH/SLA)
with smooth equatorial tapering to handle the f -> 0 singularity.
"""

import numpy as np
from typing import Tuple

# Physical constants
GRAVITY = 9.81              # m/s^2
EARTH_RADIUS = 6.371e6      # meters
OMEGA = 7.2921e-5           # Earth rotation rate in rad/s

def compute_geostrophic_currents(
    ssh: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    equatorial_blend_deg: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes geostrophic velocity components from Sea Surface Height (SSH / SLA).

    u_g = -(g / f) * d_eta / d_y
    v_g =  (g / f) * d_eta / d_x

    Args:
        ssh: 2D array of Sea Surface Height in meters, shape (H, W)
        lat: 1D array of latitudes in degrees, shape (H,)
        lon: 1D array of longitudes in degrees, shape (W,)
        equatorial_blend_deg: Width in degrees over which geostrophic velocity is smoothly tapered to zero near equator

    Returns:
        u_g (np.ndarray): Zonal geostrophic velocity (m/s), shape (H, W)
        v_g (np.ndarray): Meridional geostrophic velocity (m/s), shape (H, W)
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    lon_mesh_rad, lat_mesh_rad = np.meshgrid(lon_rad, lat_rad)
    lat_mesh_deg = np.rad2deg(lat_mesh_rad)

    # Grid step sizes in meters
    d_lat_rad = np.gradient(lat_rad) # (H,)
    d_lon_rad = np.gradient(lon_rad) # (W,)

    # Spherical metrics
    dy = EARTH_RADIUS * d_lat_rad[:, np.newaxis] # (H, 1)
    dx = EARTH_RADIUS * np.cos(lat_mesh_rad) * d_lon_rad[np.newaxis, :] # (H, W)

    # Gradients of SSH (eta)
    d_eta_dy, d_eta_dx = np.gradient(ssh)
    d_eta_dy = d_eta_dy / dy
    d_eta_dx = d_eta_dx / dx

    # Coriolis parameter: f = 2 * Omega * sin(phi)
    f = 2.0 * OMEGA * np.sin(lat_mesh_rad)

    # Avoid exact zero division
    f_safe = np.where(np.abs(f) < 1e-10, np.sign(f + 1e-15) * 1e-10, f)

    u_g_raw = -(GRAVITY / f_safe) * d_eta_dy
    v_g_raw =  (GRAVITY / f_safe) * d_eta_dx

    # Smooth equatorial damping factor: tanh(|lat| / blend_deg)
    # Approaches 0 at equator, 1 away from equator smoothly
    damping = np.tanh(np.abs(lat_mesh_deg) / max(0.5, equatorial_blend_deg))

    u_g = (u_g_raw * damping).astype(np.float32)
    v_g = (v_g_raw * damping).astype(np.float32)

    # Replace any NaNs or infinities with 0
    u_g = np.nan_to_num(u_g, nan=0.0, posinf=0.0, neginf=0.0)
    v_g = np.nan_to_num(v_g, nan=0.0, posinf=0.0, neginf=0.0)

    return u_g, v_g
