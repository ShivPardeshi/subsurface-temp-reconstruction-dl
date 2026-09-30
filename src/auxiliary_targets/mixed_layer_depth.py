"""
Mixed Layer Depth (MLD) computation module.
Standard de Boyer Montégut definition: depth where temperature drops by 0.2°C from 10m reference,
calculated via continuous vertical linear interpolation.
"""

import numpy as np
from typing import Union

def compute_mld_profile(
    temp_profile: np.ndarray,
    depths: np.ndarray,
    delta_t_threshold: float = 0.2,
    ref_depth: float = 10.0
) -> float:
    """
    Computes continuous Mixed Layer Depth for a 1D vertical temperature profile.

    Args:
        temp_profile: 1D array of temperatures in °C or K
        depths: 1D array of positive depths in meters (e.g. [0, 5, 10, ...])
        delta_t_threshold: Temperature drop threshold (default 0.2°C)
        ref_depth: Reference depth in meters (default 10.0m)

    Returns:
        mld: Continuous interpolated depth in meters where |T(z) - T_ref| >= threshold
    """
    if len(temp_profile) < 2 or np.all(np.isnan(temp_profile)):
        return 0.0

    # Find 10m reference temperature
    ref_idx = int(np.argmin(np.abs(depths - ref_depth)))
    t_ref = temp_profile[ref_idx]

    if np.isnan(t_ref):
        t_ref = temp_profile[0] # Fallback to surface

    # Search for crossing below reference depth
    for i in range(ref_idx, len(depths) - 1):
        t_curr = temp_profile[i]
        t_next = temp_profile[i + 1]
        z_curr = depths[i]
        z_next = depths[i + 1]

        if np.isnan(t_curr) or np.isnan(t_next):
            continue

        diff_curr = np.abs(t_curr - t_ref)
        diff_next = np.abs(t_next - t_ref)

        if diff_next >= delta_t_threshold:
            # Linear interpolation between z_curr and z_next
            if np.abs(diff_next - diff_curr) < 1e-6:
                return float(z_curr)
            frac = (delta_t_threshold - diff_curr) / (diff_next - diff_curr)
            frac = np.clip(frac, 0.0, 1.0)
            mld = z_curr + frac * (z_next - z_curr)
            return float(mld)

    # If threshold not reached, return maximum profile depth
    return float(depths[-1])

def compute_mld_field(
    temp_3d: np.ndarray,
    depths: np.ndarray,
    delta_t_threshold: float = 0.2,
    ref_depth: float = 10.0
) -> np.ndarray:
    """
    Computes 2D MLD field from 3D temperature tensor (depths, H, W).
    """
    D, H, W = temp_3d.shape
    mld_field = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            prof = temp_3d[:, i, j]
            mld_field[i, j] = compute_mld_profile(prof, depths, delta_t_threshold, ref_depth)

    return mld_field
