"""
Grid utilities for OceanEmbed.
Single source of truth for target grid generation.
"""

import os
import yaml
import numpy as np
from typing import Tuple, Dict, Any

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "domain_config.yaml")
)

CANONICAL_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

def load_domain_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the domain configuration YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Domain configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_target_grid(config: Dict[str, Any] = None, toy_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate target 1D latitude and longitude coordinate arrays.

    Returns:
        lat (np.ndarray): 1D array of latitudes (e.g. shape (112,) for full domain, or (40,) for toy mode).
        lon (np.ndarray): 1D array of longitudes (e.g. shape (240,) for full domain, or (40,) for toy mode).
    """
    if config is None:
        config = load_domain_config()

    domain_cfg = config.get("domain", {})
    res = float(domain_cfg.get("resolution", 0.25))

    if toy_mode:
        # 10° x 10° sub-crop focused on the Bay of Bengal core (e.g. 12°N - 22°N, 82°E - 92°E)
        lat_min = 12.0
        lat_max = 22.0
        lon_min = 82.0
        lon_max = 92.0
    else:
        lat_min = float(domain_cfg.get("lat_min", 2.0))
        lat_max = float(domain_cfg.get("lat_max", 30.0))
        lon_min = float(domain_cfg.get("lon_min", 45.0))
        lon_max = float(domain_cfg.get("lon_max", 105.0))

    # Generate coordinate arrays with fixed step resolution
    n_lat = int(round((lat_max - lat_min) / res))
    n_lon = int(round((lon_max - lon_min) / res))

    lat = np.linspace(lat_min, lat_min + (n_lat - 1) * res, n_lat, dtype=np.float32)
    lon = np.linspace(lon_min, lon_min + (n_lon - 1) * res, n_lon, dtype=np.float32)

    return lat, lon

def get_depth_levels(config: Dict[str, Any] = None) -> np.ndarray:
    """Return the 15 canonical depth levels in meters as a numpy array."""
    if config is None:
        config = load_domain_config()
    depths = config.get("depth_levels_m", [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])
    return np.array(depths, dtype=np.float32)

def get_grid_mesh(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return 2D meshgrid of latitude and longitude (lat_mesh, lon_mesh)."""
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    return lat_mesh, lon_mesh
