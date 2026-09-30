"""
Moisture Flux (E - P) computation module.
Derives evaporation from surface latent heat flux and computes net moisture flux (E - P).
Sign convention:
  Positive = Net Evaporative Loss (salinifying, Arabian Sea regime)
  Negative = Net Freshwater Gain (freshening, Bay of Bengal regime)
"""

import numpy as np
from typing import Tuple

# Physical constants
WATER_DENSITY = 1000.0      # kg/m^3
LATENT_HEAT_VAP = 2.5e6     # J/kg (latent heat of vaporization)

def compute_evaporation_from_latent_heat(
    latent_heat_flux: np.ndarray,
    is_accumulated_joules: bool = True
) -> np.ndarray:
    """
    Computes evaporation rate in mm/day from surface latent heat flux.

    Args:
        latent_heat_flux: 2D array of Surface Latent Heat Flux (J/m^2/day if accumulated, or W/m^2 if instantaneous flux)
        is_accumulated_joules: If True, input is in J/m^2 accumulated daily. If False, input is W/m^2.

    Returns:
        evaporation (np.ndarray): Evaporation rate in mm/day
    """
    # ERA5 SLHF is negative for upward flux (energy leaving ocean surface). Take absolute magnitude.
    abs_slhf = np.abs(latent_heat_flux)

    if is_accumulated_joules:
        # E (kg/m^2/day) = abs_slhf (J/m^2/day) / L_v (J/kg)
        # E (mm/day) = (E in kg/m^2) / (1000 kg/m^3) * 1000 mm/m = abs_slhf / L_v
        evap_mm_day = abs_slhf / LATENT_HEAT_VAP
    else:
        # W/m^2 -> J/m^2/day = W/m^2 * 86400 s/day
        daily_joules = abs_slhf * 86400.0
        evap_mm_day = daily_joules / LATENT_HEAT_VAP

    return np.asarray(evap_mm_day, dtype=np.float32)

def compute_moisture_flux_e_minus_p(
    evaporation_mm_day: np.ndarray,
    precipitation_mm_day: np.ndarray
) -> np.ndarray:
    """
    Computes Net Moisture Flux (E - P) in mm/day.

    E_minus_P = Evaporation (mm/day) - Precipitation (mm/day)

    Returns:
        e_minus_p (np.ndarray): Net moisture flux field in mm/day
    """
    e_minus_p = (evaporation_mm_day - precipitation_mm_day).astype(np.float32)
    return np.nan_to_num(e_minus_p, nan=0.0, posinf=0.0, neginf=0.0)
