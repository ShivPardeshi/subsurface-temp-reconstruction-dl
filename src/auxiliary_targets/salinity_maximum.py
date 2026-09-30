"""
Arabian Sea Subsurface Salinity Maximum computation module.
Tracks Persian Gulf Water (PGW) intrusion signature between 100m and 400m depth:
- Depth of salinity maximum
- Strength anomaly: S_max - S_ambient
Masked to Arabian Sea (as_mask > 0.5).
"""

import numpy as np
from typing import Tuple

def compute_salinity_maximum_profile(
    salinity_profile: np.ndarray,
    depths: np.ndarray,
    min_depth: float = 100.0,
    max_depth: float = 400.0
) -> Tuple[float, float]:
    """
    Computes depth and strength anomaly of subsurface salinity maximum for 1D profile.

    Returns:
        max_depth_m (float): Depth of peak salinity in range [min_depth, max_depth]
        strength_psu (float): Peak salinity excess above ambient background mean in that range
    """
    valid_idx = np.where((depths >= min_depth) & (depths <= max_depth))[0]
    if len(valid_idx) == 0:
        return 0.0, 0.0

    sub_sal = salinity_profile[valid_idx]
    sub_depths = depths[valid_idx]

    if np.all(np.isnan(sub_sal)):
        return 0.0, 0.0

    max_sub_idx = int(np.nanargmax(sub_sal))
    s_max = sub_sal[max_sub_idx]
    s_depth = sub_depths[max_sub_idx]

    ambient_mean = np.nanmean(sub_sal)
    s_strength = max(0.0, float(s_max - ambient_mean))

    return float(s_depth), float(s_strength)

def compute_salinity_maximum_field(
    salinity_3d: np.ndarray,
    depths: np.ndarray,
    as_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2D salinity max depth and strength fields, masked to Arabian Sea (as_mask > 0.5).

    Returns:
        sal_depth_field (np.ndarray): Shape (H, W)
        sal_strength_field (np.ndarray): Shape (H, W)
    """
    D, H, W = salinity_3d.shape
    depth_field = np.zeros((H, W), dtype=np.float32)
    strength_field = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            if as_mask[i, j] > 0.5:
                s_prof = salinity_3d[:, i, j]
                d_val, s_val = compute_salinity_maximum_profile(s_prof, depths)
                depth_field[i, j] = d_val
                strength_field[i, j] = s_val

    return depth_field, strength_field
