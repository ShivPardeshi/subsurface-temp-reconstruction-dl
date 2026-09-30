"""
Missingness tracking module.
Produces binary valid-data masks and the overall missingness channel (Channel #19).
"""

import numpy as np
from typing import Dict, List, Optional

def compute_channel_mask(data: np.ndarray, ocean_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Computes binary mask: 1 = valid ocean observation, 0 = missing/NaN/land.
    """
    valid_mask = (~np.isnan(data) & ~np.isinf(data)).astype(np.float32)
    if ocean_mask is not None:
        valid_mask = valid_mask * ocean_mask.astype(np.float32)
    return valid_mask

def compute_missingness_channel(
    core_channel_arrays: List[np.ndarray],
    ocean_mask: np.ndarray
) -> np.ndarray:
    """
    Computes Channel #19 (Missingness Mask):
    1 = any core dynamic channel is missing/cloud-obscured at this ocean cell
    0 = all core channels are valid and present
    (Land cells are 0)
    """
    if not core_channel_arrays:
        return np.zeros_like(ocean_mask, dtype=np.float32)

    # Any NaN or invalid value in core channels
    any_missing = np.zeros_like(ocean_mask, dtype=bool)
    for arr in core_channel_arrays:
        missing_here = np.isnan(arr) | np.isinf(arr)
        any_missing = any_missing | missing_here

    # Apply only to ocean pixels
    missingness_channel = (any_missing & (ocean_mask > 0.5)).astype(np.float32)
    return missingness_channel
