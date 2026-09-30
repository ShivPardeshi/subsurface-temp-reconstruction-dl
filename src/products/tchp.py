"""Tropical Cyclone Heat Potential (TCHP) Module.

Implements the verified physical formula:
    TCHP = rho * c_p * integral_0^{D26} (T(z) - 26.0) dz  * 10^-7  [kJ/cm^2]
where:
    rho = 1025.0 kg/m^3 (seawater density)
    c_p = 3990.0 J / (kg K) (specific heat of seawater)
    D26 = depth of the 26°C isotherm in meters (interpolated linearly)
    1 J/m^2 = 1e-7 kJ/cm^2 (unit conversion factor)

If surface temperature < 26.0°C, D26 = 0.0 and TCHP = 0.0 kJ/cm^2.
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np

# Physical Constants & Unit Conversion
DEFAULT_RHO = 1025.0       # kg/m^3
DEFAULT_CP = 3990.0        # J / (kg K)
J_M2_TO_KJ_CM2 = 1.0e-7    # Conversion from J/m^2 to kJ/cm^2
ISOTHERM_TEMP = 26.0       # Reference isotherm for tropical cyclogenesis (°C)
CANONICAL_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def _trapezoid(y: Any, x: Any = None, axis: int = -1) -> Any:
    """NumPy 1.x and 2.x compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def find_d26_isotherm_depth(
    temperatures: np.ndarray,
    depths: Union[List[float], np.ndarray] = CANONICAL_DEPTHS,
) -> float:
    """Find the 26°C isotherm depth via linear interpolation between bounding depths.

    Args:
        temperatures: 1D array of temperatures in °C from surface downward.
        depths: 1D array of canonical depth levels in meters.

    Returns:
        D26 depth in meters. Returns 0.0 if surface temperature < 26.0°C.
    """
    temps = np.asarray(temperatures, dtype=np.float64)
    z = np.asarray(depths, dtype=np.float64)

    if temps[0] < ISOTHERM_TEMP:
        return 0.0

    # Search for crossing point where T drops below 26.0°C
    for i in range(len(temps) - 1):
        t0, t1 = temps[i], temps[i + 1]
        z0, z1 = z[i], z[i + 1]

        if t0 >= ISOTHERM_TEMP and t1 <= ISOTHERM_TEMP:
            if t0 == t1:
                return float(z1)
            # Linear interpolation: T(z) = t0 + (z - z0)/(z1 - z0)*(t1 - t0) = 26.0
            frac = (ISOTHERM_TEMP - t0) / (t1 - t0)
            d26 = z0 + frac * (z1 - z0)
            return float(d26)

    # If entire profile is >= 26.0°C (e.g. shallow warm basin), return deepest level
    if temps[-1] >= ISOTHERM_TEMP:
        return float(z[-1])

    return 0.0


def compute_profile_tchp(
    temperatures: np.ndarray,
    depths: Union[List[float], np.ndarray] = CANONICAL_DEPTHS,
    rho: float = DEFAULT_RHO,
    c_p: float = DEFAULT_CP,
) -> Tuple[float, float]:
    """Compute TCHP and D26 for a single 1D vertical temperature profile.

    Args:
        temperatures: 1D array of temperatures in °C.
        depths: 1D array of canonical depth levels in meters.
        rho: Seawater density (kg/m^3).
        c_p: Specific heat capacity (J / kg K).

    Returns:
        Tuple of (tchp_kj_cm2, d26_meters).
    """
    temps = np.asarray(temperatures, dtype=np.float64)
    z = np.asarray(depths, dtype=np.float64)

    d26 = find_d26_isotherm_depth(temps, z)
    if d26 <= 0.0:
        return 0.0, 0.0

    # Sub-levels within D26
    sub_mask = z <= d26
    z_sub = list(z[sub_mask])
    t_sub = list(temps[sub_mask])

    # Append interpolated D26 crossing point
    if z_sub[-1] < d26:
        z_sub.append(d26)
        t_sub.append(ISOTHERM_TEMP)

    # Temperature excess above 26.0°C
    excess_t = np.maximum(0.0, np.array(t_sub) - ISOTHERM_TEMP)

    # Integrate excess temperature: integral_0^D26 (T(z) - 26) dz
    integral_excess = _trapezoid(excess_t, z_sub)

    # Convert to kJ / cm^2
    tchp = rho * c_p * integral_excess * J_M2_TO_KJ_CM2
    return float(tchp), float(d26)


def compute_tchp_field(
    temperature_field: np.ndarray,
    depths: Union[List[float], np.ndarray] = CANONICAL_DEPTHS,
    mask: Optional[np.ndarray] = None,
    rho: float = DEFAULT_RHO,
    c_p: float = DEFAULT_CP,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D TCHP and D26 maps across a 3D (15, H, W) temperature field.

    Args:
        temperature_field: 3D array (15, H, W) in °C.
        depths: Depth array.
        mask: Optional boolean ocean mask (H, W). True = ocean.
        rho: Seawater density.
        c_p: Specific heat.

    Returns:
        Tuple of:
        - tchp_map: (H, W) in kJ/cm^2
        - d26_map: (H, W) in meters
    """
    field = np.asarray(temperature_field, dtype=np.float32)
    H, W = field.shape[1], field.shape[2]
    tchp_map = np.zeros((H, W), dtype=np.float32)
    d26_map = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            if mask is not None and not mask[i, j]:
                continue
            t_prof = field[:, i, j]
            if np.isnan(t_prof[0]):
                continue
            val_tchp, val_d26 = compute_profile_tchp(t_prof, depths, rho=rho, c_p=c_p)
            tchp_map[i, j] = val_tchp
            d26_map[i, j] = val_d26

    return tchp_map, d26_map


# Convenient alias
compute_tchp_profile = compute_profile_tchp

