"""
Region membership maps module.
Generates four soft distance-blended spatial membership channels:
- Channel #22: Arabian Sea membership
- Channel #23: Bay of Bengal membership
- Channel #24: 8–10°N Confluence Zone membership
- Channel #25: Open-Ocean / Equatorial-Edge membership
"""

import numpy as np
from typing import Dict, Any, Tuple
from src.utils.grid import get_grid_mesh, load_domain_config

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Standard numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

def generate_region_masks(
    lat: np.ndarray,
    lon: np.ndarray,
    config: Dict[str, Any] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates the four soft, distance-blended region membership maps (each in range [0, 1]).

    Args:
        lat: 1D array of latitudes (H,)
        lon: 1D array of longitudes (W,)
        config: Domain configuration dictionary

    Returns:
        as_mask, bob_mask, conf_mask, open_mask: Four 2D float32 arrays of shape (H, W)
    """
    if config is None:
        config = load_domain_config()

    reg_cfg = config.get("region_boundaries", {})
    as_lon_max = float(reg_cfg.get("arabian_sea_lon_max", 77.0))
    bob_lon_min = float(reg_cfg.get("bay_of_bengal_lon_min", 80.0))
    conf_lat_range = reg_cfg.get("confluence_zone_lat_range", [8.0, 10.0])
    blend_width = float(reg_cfg.get("blend_distance_degrees", 2.0))

    lat_mesh, lon_mesh = get_grid_mesh(lat, lon)

    # 1. Confluence Zone (centered around 8-10°N)
    conf_center = (conf_lat_range[0] + conf_lat_range[1]) / 2.0
    conf_half_width = (conf_lat_range[1] - conf_lat_range[0]) / 2.0
    dist_to_conf = np.abs(lat_mesh - conf_center)
    conf_mask = np.clip(1.0 - (dist_to_conf - conf_half_width) / blend_width, 0.0, 1.0)
    # Confluence primarily active near southern tip of India / Sri Lanka (lon 73 to 85)
    lon_weight_conf = np.clip(1.0 - np.abs(lon_mesh - 78.5) / 12.0, 0.0, 1.0)
    conf_mask = conf_mask * lon_weight_conf

    # 2. Arabian Sea (North of confluence, West of 77°E)
    north_weight = np.clip((lat_mesh - conf_lat_range[0]) / blend_width, 0.0, 1.0)
    west_weight = np.clip((as_lon_max - lon_mesh) / blend_width, 0.0, 1.0)
    as_mask = north_weight * west_weight * (1.0 - 0.5 * conf_mask)

    # 3. Bay of Bengal (North of confluence, East of 80°E)
    east_weight = np.clip((lon_mesh - bob_lon_min) / blend_width, 0.0, 1.0)
    bob_mask = north_weight * east_weight * (1.0 - 0.5 * conf_mask)

    # 4. Open-Ocean / Equatorial-Edge (South of confluence / open domain)
    south_weight = np.clip((conf_lat_range[1] - lat_mesh) / blend_width, 0.0, 1.0)
    open_mask = south_weight * (1.0 - conf_mask)

    # Normalize across regions so they sum to <= 1.0 smoothly
    total = as_mask + bob_mask + conf_mask + open_mask + 1e-6
    max_total = np.maximum(1.0, total)

    as_mask = (as_mask / max_total).astype(np.float32)
    bob_mask = (bob_mask / max_total).astype(np.float32)
    conf_mask = (conf_mask / max_total).astype(np.float32)
    open_mask = (open_mask / max_total).astype(np.float32)

    return as_mask, bob_mask, conf_mask, open_mask
