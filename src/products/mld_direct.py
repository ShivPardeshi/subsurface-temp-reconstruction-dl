"""Direct Mixed Layer Depth (MLD) Profile Analysis Module.

Implements the standard de Boyer Montégut et al. (2004) oceanographic method:
    T(z_mld) = T(10m) - 0.2°C
applied directly to the reconstructed 3D temperature profile.
The exact crossing point is resolved via linear interpolation between bracketing depths.

Auxiliary head predictions are supported as a secondary cross-check, explicitly
flagged as lower-confidence.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

CANONICAL_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
REF_DEPTH_INDEX = 2  # 10m depth index in CANONICAL_DEPTHS
DELTA_T_THRESHOLD = 0.2  # °C threshold from reference depth


def compute_profile_mld_direct(
    temperatures: np.ndarray,
    depths: Union[List[float], np.ndarray] = CANONICAL_DEPTHS,
    ref_depth: float = 10.0,
    delta_t: float = DELTA_T_THRESHOLD,
    aux_head_pred: Optional[float] = None,
) -> Dict[str, Union[float, str]]:
    """Compute Mixed Layer Depth (MLD) directly from a 1D temperature profile.

    Args:
        temperatures: 1D array of temperatures in °C from surface downward.
        depths: 1D array of canonical depth levels in meters.
        ref_depth: Reference depth (meters), default 10.0m.
        delta_t: Temperature drop criterion (°C), default 0.2°C.
        aux_head_pred: Optional scalar prediction from Stage 5 MLD auxiliary head.

    Returns:
        Dictionary containing:
        - 'mld_direct_m': Direct profile MLD in meters (primary high-confidence estimate).
        - 'ref_temp_c': Temperature at reference depth.
        - 'thresh_temp_c': Threshold temperature for MLD definition.
        - 'aux_head_mld_m': Auxiliary prediction (if provided).
        - 'aux_head_confidence': Label note on auxiliary head reliability.
    """
    temps = np.asarray(temperatures, dtype=np.float64)
    z = np.asarray(depths, dtype=np.float64)

    # 1. Find reference temperature at ref_depth (10m)
    ref_idx = np.where(z == ref_depth)[0]
    if len(ref_idx) > 0:
        t_ref = temps[ref_idx[0]]
    else:
        # Interpolate if ref_depth is between depths
        t_ref = np.interp(ref_depth, z, temps)

    t_thresh = t_ref - delta_t

    # 2. Search below ref_depth for temperature dropping below t_thresh
    mld_m = float(z[-1])
    search_start = np.where(z >= ref_depth)[0][0]

    for i in range(search_start, len(temps) - 1):
        t0, t1 = temps[i], temps[i + 1]
        z0, z1 = z[i], z[i + 1]

        if t0 >= t_thresh and t1 <= t_thresh:
            if t0 == t1:
                mld_m = float(z1)
            else:
                frac = (t0 - t_thresh) / (t0 - t1)
                mld_m = float(z0 + frac * (z1 - z0))
            break
    else:
        # If never drops below t_thresh, entire profile is well-mixed
        if temps[-1] >= t_thresh:
            mld_m = float(z[-1])

    result: Dict[str, Union[float, str]] = {
        "mld_direct_m": round(mld_m, 2),
        "ref_temp_c": round(float(t_ref), 3),
        "thresh_temp_c": round(float(t_thresh), 3),
        "method": "de Boyer Montégut (0.2°C drop from 10m)",
    }

    if aux_head_pred is not None:
        result["aux_head_mld_m"] = round(float(aux_head_pred), 2)
        result["aux_head_confidence"] = "secondary cross-check (lower confidence: r=0.036)"

    return result


def compute_mld_field(
    temperature_field: np.ndarray,
    depths: Union[List[float], np.ndarray] = CANONICAL_DEPTHS,
    mask: Optional[np.ndarray] = None,
    ref_depth: float = 10.0,
    delta_t: float = DELTA_T_THRESHOLD,
) -> np.ndarray:
    """Compute 2D MLD map from 3D (15, H, W) temperature field.

    Args:
        temperature_field: (15, H, W) in °C.
        depths: Depth array.
        mask: Optional boolean ocean mask. True = ocean.
        ref_depth: Reference depth in meters.
        delta_t: Temperature drop criterion in °C.

    Returns:
        2D array (H, W) of MLD in meters.
    """
    field = np.asarray(temperature_field, dtype=np.float32)
    H, W = field.shape[1], field.shape[2]
    mld_map = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            if mask is not None and not mask[i, j]:
                continue
            t_prof = field[:, i, j]
            if np.isnan(t_prof[0]):
                continue
            res = compute_profile_mld_direct(t_prof, depths, ref_depth=ref_depth, delta_t=delta_t)
            mld_map[i, j] = res["mld_direct_m"]

    return mld_map


# Convenient alias
compute_mld_direct_profile = compute_profile_mld_direct

