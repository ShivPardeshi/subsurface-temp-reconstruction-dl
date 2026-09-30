"""
GBM River Plume Influence Field module.
Computes distance-decay plume influence field from Ganges-Brahmaputra-Meghna delta outflow:
  plume_influence(x, y, t) = Q(t) * exp(-distance(x, y, mouth) / decay_length_scale)
"""

import numpy as np
from typing import Tuple, Optional

# Physical constants
EARTH_RADIUS_KM = 6371.0
GBM_MOUTH_LAT = 21.5        # Approximate main delta outflow latitude (°N)
GBM_MOUTH_LON = 89.5        # Approximate main delta outflow longitude (°E)
DEFAULT_DECAY_SCALE_KM = 250.0  # e-folding decay length scale in km

def compute_haversine_distance_km(
    lat: np.ndarray,
    lon: np.ndarray,
    ref_lat: float = GBM_MOUTH_LAT,
    ref_lon: float = GBM_MOUTH_LON
) -> np.ndarray:
    """
    Computes 2D distance field (in km) from reference coordinates (ref_lat, ref_lon)
    to all grid points (lat, lon).
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    ref_lat_rad = np.deg2rad(ref_lat)
    ref_lon_rad = np.deg2rad(ref_lon)

    lon_mesh_rad, lat_mesh_rad = np.meshgrid(lon_rad, lat_rad)

    dlat = lat_mesh_rad - ref_lat_rad
    dlon = lon_mesh_rad - ref_lon_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(ref_lat_rad) * np.cos(lat_mesh_rad) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    distance_km = EARTH_RADIUS_KM * c

    return distance_km

def compute_river_plume_field(
    lat: np.ndarray,
    lon: np.ndarray,
    discharge_m3_s: float = 30000.0,
    decay_scale_km: float = DEFAULT_DECAY_SCALE_KM,
    ref_mouth_lat: float = GBM_MOUTH_LAT,
    ref_mouth_lon: float = GBM_MOUTH_LON
) -> np.ndarray:
    """
    Computes the 2D river plume influence field.

    Args:
        lat: 1D array of latitudes (H,)
        lon: 1D array of longitudes (W,)
        discharge_m3_s: Real or estimated daily discharge in m^3/s (default ~30,000 m^3/s)
        decay_scale_km: Spatial e-folding decay scale in km
        ref_mouth_lat: River mouth latitude (°N)
        ref_mouth_lon: River mouth longitude (°E)

    Returns:
        plume_field: 2D array of shape (H, W), normalized proxy value (0 to ~1)
    """
    dist_km = compute_haversine_distance_km(lat, lon, ref_mouth_lat, ref_mouth_lon)

    # Normalize discharge magnitude relative to typical peak (~50,000 m^3/s)
    q_norm = np.clip(discharge_m3_s / 50000.0, 0.0, 2.0)

    # Exponential radial decay proxy
    plume_field = (q_norm * np.exp(-dist_km / decay_scale_km)).astype(np.float32)

    return np.nan_to_num(plume_field, nan=0.0, posinf=0.0, neginf=0.0)
