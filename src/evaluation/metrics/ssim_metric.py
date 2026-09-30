"""Structural Similarity Index (SSIM) Metric.

Evaluates structural fidelity of 2D spatial ocean temperature fields per depth
level following Asefi et al. and Wang et al. (2004).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from scipy.ndimage import uniform_filter
from .basic_metrics import _to_numpy


def compute_ssim_2d(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    win_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: Optional[float] = None,
) -> float:
    """Compute 2D Structural Similarity Index (SSIM) between spatial fields.

    Args:
        pred: 2D array of shape (H, W).
        target: 2D array of shape (H, W).
        mask: Optional 2D boolean mask of shape (H, W) (True = valid ocean).
        win_size: Size of sliding window (default 7x7).
        k1, k2: Stability constants.
        data_range: Dynamic range of values. If None, derived from target.

    Returns:
        SSIM index float in [-1, 1] (1.0 = identical structure).
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)

    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch in SSIM: {p.shape} vs {t.shape}")

    # Determine dynamic range
    if data_range is None:
        if mask is not None:
            m = _to_numpy(mask).astype(bool)
            t_valid = t[m & np.isfinite(t)]
        else:
            t_valid = t[np.isfinite(t)]
        if len(t_valid) > 0:
            data_range = float(np.ptp(t_valid))
            if data_range < 1e-4:
                data_range = 1.0
        else:
            data_range = 1.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Replace NaNs/Infs with local or global mean for continuous filtering
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        p_clean = p.copy()
        t_clean = t.copy()
        valid_m_p = m & np.isfinite(p)
        valid_m_t = m & np.isfinite(t)
        valid_mean_p = float(np.mean(p[valid_m_p])) if np.any(valid_m_p) else 0.0
        valid_mean_t = float(np.mean(t[valid_m_t])) if np.any(valid_m_t) else 0.0
        p_clean[~valid_m_p] = valid_mean_p
        t_clean[~valid_m_t] = valid_mean_t
    else:
        p_clean = np.nan_to_num(p)
        t_clean = np.nan_to_num(t)

    # Local means via uniform box filter
    mu_p = uniform_filter(p_clean, size=win_size)
    mu_t = uniform_filter(t_clean, size=win_size)

    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t

    # Local variances and covariance
    sigma_p_sq = uniform_filter(p_clean * p_clean, size=win_size) - mu_p_sq
    sigma_t_sq = uniform_filter(t_clean * t_clean, size=win_size) - mu_t_sq
    sigma_pt = uniform_filter(p_clean * t_clean, size=win_size) - mu_pt

    # Numerical stability
    sigma_p_sq = np.maximum(0.0, sigma_p_sq)
    sigma_t_sq = np.maximum(0.0, sigma_t_sq)

    # Full SSIM map
    ssim_map = ((2.0 * mu_pt + c1) * (2.0 * sigma_pt + c2)) / (
        (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)
    )

    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        # Only average over valid ocean region
        valid_ssim = ssim_map[m & np.isfinite(ssim_map)]
        if len(valid_ssim) == 0:
            return float("nan")
        return float(np.mean(valid_ssim))

    finite_ssim = ssim_map[np.isfinite(ssim_map)]
    return float(np.mean(finite_ssim)) if len(finite_ssim) > 0 else float("nan")


def compute_profile_ssim(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    depths: Optional[List[float]] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    win_size: int = 7,
) -> Dict[str, float]:
    """Compute SSIM for each depth level in a 3D (D, H, W) or 4D (B, D, H, W) profile.

    Returns:
        Dictionary mapping depth strings (e.g. '0m', '50m') to SSIM values.
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)

    if p.ndim == 4:
        p = np.mean(p, axis=0)  # average over batch
        t = np.mean(t, axis=0)

    num_depths = p.shape[0]
    ssim_by_depth = {}

    for d in range(num_depths):
        p_d = p[d]
        t_d = t[d]
        depth_key = f"{depths[d]}m" if depths and d < len(depths) else f"depth_{d}"
        ssim_val = compute_ssim_2d(p_d, t_d, mask=mask, win_size=win_size)
        ssim_by_depth[depth_key] = ssim_val

    return ssim_by_depth
