"""Basic evaluation metrics: RMSE, Correlation, Bias, MAE, and R² vs Climatology.

All metrics support 2D spatial fields, 3D cubes (depth, H, W), and masked arrays.
"""

from typing import Dict, Optional, Union
import numpy as np
import torch


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert input tensor or array to float64 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _apply_mask(
    arr: np.ndarray, mask: Optional[Union[np.ndarray, torch.Tensor]]
) -> np.ndarray:
    """Flatten array or apply broadcasted boolean mask."""
    if mask is None:
        return arr.flatten()
    m = _to_numpy(mask).astype(bool)
    if m.shape != arr.shape:
        # Broadcast e.g. 2D spatial mask (H, W) to 3D (D, H, W) or 4D (B, D, H, W)
        m = np.broadcast_to(m, arr.shape)
    return arr[m]


def compute_rmse(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Compute Root Mean Squared Error (RMSE).

    Args:
        pred: Predicted array.
        target: Ground truth array.
        mask: Optional boolean mask (True = valid/ocean, False = ignore/land).

    Returns:
        RMSE scalar float.
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)

    p = _apply_mask(p, mask)
    t = _apply_mask(t, mask)

    valid = np.isfinite(p) & np.isfinite(t)
    if not np.any(valid):
        return float("nan")

    p = p[valid]
    t = t[valid]
    return float(np.sqrt(np.mean((p - t) ** 2)))


def compute_mae(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Compute Mean Absolute Error (MAE)."""
    p = _to_numpy(pred)
    t = _to_numpy(target)

    p = _apply_mask(p, mask)
    t = _apply_mask(t, mask)

    valid = np.isfinite(p) & np.isfinite(t)
    if not np.any(valid):
        return float("nan")

    p = p[valid]
    t = t[valid]
    return float(np.mean(np.abs(p - t)))


def compute_bias(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Compute Mean Bias Error (pred - target)."""
    p = _to_numpy(pred)
    t = _to_numpy(target)

    p = _apply_mask(p, mask)
    t = _apply_mask(t, mask)

    valid = np.isfinite(p) & np.isfinite(t)
    if not np.any(valid):
        return float("nan")

    p = p[valid]
    t = t[valid]
    return float(np.mean(p - t))


def compute_correlation_1d(p: np.ndarray, t: np.ndarray) -> float:
    """Compute Pearson correlation on 1D vectors."""
    valid = np.isfinite(p) & np.isfinite(t)
    if np.sum(valid) < 3:
        return float("nan")
    p_v = p[valid]
    t_v = t[valid]
    std_p = np.std(p_v)
    std_t = np.std(t_v)
    if std_p < 1e-9 or std_t < 1e-9:
        return 0.0
    corr = np.corrcoef(p_v, t_v)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def compute_correlation(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pool_depths: bool = False,
) -> float:
    """Compute Pearson Correlation Coefficient.

    CRITICAL OCEANOGRAPHIC DISCIPLINE (Requirement #5):
    If input is 3D (D, H, W) or 4D (B, D, H, W), spatial correlation is computed
    individually per depth level and averaged across depths.
    Pooling across depths is explicitly avoided by default because the 24°C
    vertical lapse rate artificially inflates pooled correlation to >0.98.
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)

    # 1D or 2D inputs, or explicit pooling requested
    if p.ndim <= 2 or pool_depths:
        p_flat = _apply_mask(p, mask)
        t_flat = _apply_mask(t, mask)
        return compute_correlation_1d(p_flat, t_flat)

    # 3D (D, H, W) or 4D (B, D, H, W)
    if p.ndim == 4:
        p = np.mean(p, axis=0)  # average across batch
        t = np.mean(t, axis=0)

    num_depths = p.shape[0]
    corrs = []
    m_2d = _to_numpy(mask) if mask is not None else None
    if m_2d is not None and m_2d.ndim > 2:
        m_2d = m_2d[0] if m_2d.ndim == 3 else m_2d[0, 0]

    for d in range(num_depths):
        p_d = _apply_mask(p[d], m_2d)
        t_d = _apply_mask(t[d], m_2d)
        r_d = compute_correlation_1d(p_d, t_d)
        if np.isfinite(r_d):
            corrs.append(r_d)

    return float(np.mean(corrs)) if corrs else float("nan")


def compute_r2(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    clim: Optional[Union[np.ndarray, torch.Tensor]] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Compute Coefficient of Determination (R²) against climatology baseline.

    R² = 1 - ( sum((target - pred)^2) / sum((target - clim)^2) )
    If clim is None, uses target.mean() as the reference baseline.
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)

    p = _apply_mask(p, mask)
    t = _apply_mask(t, mask)

    if clim is not None:
        c = _to_numpy(clim)
        c = _apply_mask(c, mask)
    else:
        c = None

    valid = np.isfinite(p) & np.isfinite(t)
    if c is not None:
        valid = valid & np.isfinite(c)

    if not np.any(valid):
        return float("nan")

    p = p[valid]
    t = t[valid]
    if c is not None:
        c = c[valid]
        ss_tot = np.sum((t - c) ** 2)
    else:
        ss_tot = np.sum((t - np.mean(t)) ** 2)

    ss_res = np.sum((t - p) ** 2)

    if ss_tot < 1e-9:
        return 1.0 if ss_res < 1e-9 else 0.0

    return float(1.0 - (ss_res / ss_tot))


def compute_all_basic_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    clim: Optional[Union[np.ndarray, torch.Tensor]] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Compute complete suite of basic metrics."""
    return {
        "rmse": compute_rmse(pred, target, mask=mask),
        "mae": compute_mae(pred, target, mask=mask),
        "bias": compute_bias(pred, target, mask=mask),
        "correlation": compute_correlation(pred, target, mask=mask),
        "r2": compute_r2(pred, target, clim=clim, mask=mask),
    }
