"""
Ageostrophic current computation module.
Computes ageostrophic velocity components:
ageo_u = observed_u - geostrophic_u
ageo_v = observed_v - geostrophic_v
"""

import numpy as np
from typing import Tuple

def compute_ageostrophic_currents(
    observed_u: np.ndarray,
    observed_v: np.ndarray,
    geostrophic_u: np.ndarray,
    geostrophic_v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes ageostrophic currents representing wind-driven and Ekman flow components.

    Args:
        observed_u: Total observed zonal surface current (m/s)
        observed_v: Total observed meridional surface current (m/s)
        geostrophic_u: Geostrophic zonal current (m/s)
        geostrophic_v: Geostrophic meridional current (m/s)

    Returns:
        ageo_u, ageo_v: Ageostrophic current components (m/s)
    """
    ageo_u = (observed_u - geostrophic_u).astype(np.float32)
    ageo_v = (observed_v - geostrophic_v).astype(np.float32)

    ageo_u = np.nan_to_num(ageo_u, nan=0.0, posinf=0.0, neginf=0.0)
    ageo_v = np.nan_to_num(ageo_v, nan=0.0, posinf=0.0, neginf=0.0)

    return ageo_u, ageo_v
