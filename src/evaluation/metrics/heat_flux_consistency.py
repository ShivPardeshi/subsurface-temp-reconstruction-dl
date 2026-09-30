"""Physical Heat Flux Consistency Metric.

Evaluates physical consistency of predicted temperature fields with real
observed ocean circulation by computing meridional heat transport:
  q_v = ρ · c_p · V_input · T

where:
  ρ   = 1025 kg/m³ (seawater reference density)
  c_p = 3990 J/(kg·K) (seawater specific heat capacity)
  V_input = input surface/subsurface meridional velocity field (m/s)
  T = temperature (°C)
"""

from typing import Dict, Optional, Union
import numpy as np
import torch
from .basic_metrics import _to_numpy, compute_rmse, compute_correlation

# Physical constants
RHO_SEAWATER = 1025.0  # kg/m³
CP_SEAWATER = 3990.0   # J/(kg·K)


def compute_meridional_heat_flux(
    v_velocity: Union[np.ndarray, torch.Tensor],
    temperature: Union[np.ndarray, torch.Tensor],
) -> np.ndarray:
    """Compute 2D or 3D meridional heat flux density (W/m²).

    Args:
        v_velocity: Meridional current velocity (m/s).
        temperature: Temperature field (°C).

    Returns:
        Heat flux density q_v (W/m²).
    """
    v = _to_numpy(v_velocity)
    t = _to_numpy(temperature)
    return RHO_SEAWATER * CP_SEAWATER * v * t


def evaluate_heat_flux_consistency(
    v_input: Union[np.ndarray, torch.Tensor],
    t_pred: Union[np.ndarray, torch.Tensor],
    t_true: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """Compare physical meridional heat transport between predicted and true temperature.

    Args:
        v_input: Meridional velocity input field (e.g. geostrophic + ageostrophic v).
        t_pred: Predicted temperature field.
        t_true: Ground truth temperature field.
        mask: Ocean mask.

    Returns:
        Dictionary with:
        - 'heat_flux_rmse': RMSE of heat flux density (W/m²).
        - 'heat_flux_correlation': Pearson r of heat flux fields.
        - 'zonal_mean_relative_error': Relative error of zonally integrated heat transport.
    """
    q_pred = compute_meridional_heat_flux(v_input, t_pred)
    q_true = compute_meridional_heat_flux(v_input, t_true)

    rmse = compute_rmse(q_pred, q_true, mask=mask)

    m = _to_numpy(mask).astype(bool) if mask is not None else np.isfinite(q_true)
    if m.ndim < q_true.ndim:
        m = np.broadcast_to(m, q_true.shape)

    # Physical reference baseline magnitude
    true_valid = q_true[m]
    true_mean_magnitude = float(np.nanmean(np.abs(true_valid))) if len(true_valid) > 0 else 1.0
    relative_rmse_pct = float((rmse / true_mean_magnitude) * 100.0) if true_mean_magnitude > 0 else 0.0

    # Unpooled per-depth correlation if 3D, or 2D spatial correlation
    if q_pred.ndim >= 3 and q_pred.shape[0] > 1:
        corrs = []
        for d in range(q_pred.shape[0]):
            c = compute_correlation(q_pred[d], q_true[d], mask=mask)
            if np.isfinite(c):
                corrs.append(c)
        corr = float(np.mean(corrs)) if corrs else float(compute_correlation(q_pred, q_true, mask=mask))
    else:
        corr = float(compute_correlation(q_pred, q_true, mask=mask))

    # Zonal mean transport across latitude lines (axis -2)
    q_p_masked = np.where(m, q_pred, 0.0)
    q_t_masked = np.where(m, q_true, 0.0)

    # Sum along longitude axis (-1)
    zonal_pred = np.sum(q_p_masked, axis=-1)
    zonal_true = np.sum(q_t_masked, axis=-1)

    denom = np.sum(np.abs(zonal_true))
    if denom > 1e-6:
        rel_error = float(np.sum(np.abs(zonal_pred - zonal_true)) / denom)
    else:
        rel_error = 0.0

    return {
        "heat_flux_rmse_wm2": float(rmse),
        "heat_flux_correlation": float(corr),
        "heat_flux_true_ref_wm2": float(true_mean_magnitude),
        "heat_flux_relative_rmse_pct": float(relative_rmse_pct),
        "zonal_transport_relative_error": float(rel_error),
        "zonal_profile_pred": zonal_pred,
        "zonal_profile_true": zonal_true,
    }
