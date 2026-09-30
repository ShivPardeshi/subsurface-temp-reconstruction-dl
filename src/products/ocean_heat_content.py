"""Ocean Heat Content (OHC) Calculation Module.

Computes column-integrated thermal energy per unit surface area:
    OHC = rho * c_p * integral_0^D T(z) dz   [J/m^2]
Standard oceanographic values:
    rho = 1025.0 kg/m^3 (mean seawater density)
    c_p = 3990.0 J/(kg K) (specific heat of seawater)
Supports:
1. Standard OHC to fixed reference depth (commonly 700m or 300m).
2. OHC to the 26°C isotherm depth (D26).
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np

# Physical Constants
DEFAULT_RHO = 1025.0       # Seawater density (kg/m^3)
DEFAULT_CP = 3990.0        # Specific heat capacity (J / kg K)
CANONICAL_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def _trapezoid(y: Any, x: Any = None, axis: int = -1) -> Any:
    """NumPy 1.x and 2.x compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def integrate_vertical_heat(
    temperatures: np.ndarray,
    depths: Union[List[float], np.ndarray],
    max_depth: float = 700.0,
    rho: float = DEFAULT_RHO,
    c_p: float = DEFAULT_CP,
) -> float:
    """Trapezoidal integration of thermal energy from 0m to max_depth for a 1D vertical profile.

    Args:
        temperatures: 1D array of temperatures in °C across depths.
        depths: 1D array of canonical depth levels in meters.
        max_depth: Integration depth limit in meters (e.g. 700.0).
        rho: Seawater density (kg/m^3).
        c_p: Specific heat capacity (J / kg K).

    Returns:
        OHC in Joules per square meter (J/m^2).
    """
    temps = np.asarray(temperatures, dtype=np.float64)
    z = np.asarray(depths, dtype=np.float64)

    if len(temps) != len(z):
        raise ValueError(f"Length mismatch: temps {len(temps)} vs depths {len(z)}")

    # Truncate or interpolate at max_depth
    valid_idx = np.where(z <= max_depth)[0]
    if len(valid_idx) == 0:
        return 0.0

    z_sub = list(z[valid_idx])
    t_sub = list(temps[valid_idx])

    # If max_depth is between depths, interpolate temperature at max_depth
    if z_sub[-1] < max_depth and len(valid_idx) < len(z):
        next_idx = valid_idx[-1] + 1
        z0, z1 = z[valid_idx[-1]], z[next_idx]
        t0, t1 = temps[valid_idx[-1]], temps[next_idx]
        frac = (max_depth - z0) / (z1 - z0)
        t_max = t0 + frac * (t1 - t0)
        z_sub.append(max_depth)
        t_sub.append(t_max)

    # Trapezoidal integration: integral T(z) dz
    integral_t = _trapezoid(t_sub, z_sub)
    ohc_jm2 = rho * c_p * integral_t
    return float(ohc_jm2)


def compute_ohc_field(
    temperature_field: np.ndarray,
    depths: Union[List[float], np.ndarray] = CANONICAL_DEPTHS,
    max_depth: float = 700.0,
    mask: Optional[np.ndarray] = None,
    rho: float = DEFAULT_RHO,
    c_p: float = DEFAULT_CP,
) -> np.ndarray:
    """Compute 2D OHC map from a 3D (15, H, W) temperature field.

    Args:
        temperature_field: 3D array (15, H, W) in °C.
        depths: Depth array matching vertical axis.
        max_depth: Maximum integration depth (default 700m).
        mask: Optional boolean ocean mask (H, W). True = ocean.
        rho: Seawater density (kg/m^3).
        c_p: Specific heat capacity (J / kg K).

    Returns:
        2D array (H, W) of OHC in J/m^2 (land masked to 0.0).
    """
    field = np.asarray(temperature_field, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.float32)
    H, W = field.shape[1], field.shape[2]
    ohc_map = np.zeros((H, W), dtype=np.float32)

    # Vectorized trapezoidal integration up to max_depth
    valid_mask = depths <= max_depth
    sub_depths = depths[valid_mask]
    sub_field = field[valid_mask]  # (K, H, W)

    if sub_depths[-1] < max_depth and len(sub_depths) < len(depths):
        next_idx = len(sub_depths)
        z0, z1 = depths[next_idx - 1], depths[next_idx]
        t0, t1 = field[next_idx - 1], field[next_idx]
        frac = (max_depth - z0) / (z1 - z0)
        t_at_max = t0 + frac * (t1 - t0)  # (H, W)
        sub_depths = np.append(sub_depths, max_depth)
        sub_field = np.concatenate([sub_field, t_at_max[None, ...]], axis=0)

    # Trapz along depth axis 0
    integral_2d = _trapezoid(sub_field, sub_depths, axis=0)  # (H, W)
    ohc_map = (rho * c_p * integral_2d).astype(np.float32)

    if mask is not None:
        ohc_map = np.where(mask, ohc_map, 0.0).astype(np.float32)

    return ohc_map
