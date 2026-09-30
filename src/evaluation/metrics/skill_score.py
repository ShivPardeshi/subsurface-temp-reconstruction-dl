"""Murphy Skill Score Metric.

Computes skill score relative to an operational reference baseline (climatology):
SS = 1 - (MSE_model / MSE_climatology)

Interpretation:
  SS > 0: Model outperforms the climatological reference baseline.
  SS = 1: Perfect forecast.
  SS = 0: Model has equal performance to simply predicting climatology.
  SS < 0: Model performs worse than the climatological baseline.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
from .basic_metrics import _to_numpy, _apply_mask


def compute_murphy_skill_score(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    clim: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Compute global or sliced Murphy Skill Score.

    Args:
        pred: Predicted temperature field.
        target: Ground truth temperature field.
        clim: Climatology baseline field.
        mask: Optional boolean mask (True = valid, False = masked out).

    Returns:
        Murphy skill score float.
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)
    c = _to_numpy(clim)

    p = _apply_mask(p, mask)
    t = _apply_mask(t, mask)
    c = _apply_mask(c, mask)

    valid = np.isfinite(p) & np.isfinite(t) & np.isfinite(c)
    if not np.any(valid):
        return float("nan")

    p = p[valid]
    t = t[valid]
    c = c[valid]

    mse_model = np.mean((p - t) ** 2)
    mse_clim = np.mean((c - t) ** 2)

    if mse_clim < 1e-9:
        # If climatology has zero error, skill score is 1.0 if model is also zero, else 0.0
        return 1.0 if mse_model < 1e-9 else 0.0

    return float(1.0 - (mse_model / mse_clim))


def compute_depthwise_skill_score(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    clim: Union[np.ndarray, torch.Tensor],
    depths: Optional[List[float]] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Compute Murphy Skill Score individually for each depth level.

    Expected input shapes: (D, H, W) or (B, D, H, W).
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)
    c = _to_numpy(clim)

    # If batch dimension exists, average over batch
    if p.ndim == 4:
        # (B, D, H, W)
        num_depths = p.shape[1]
    elif p.ndim == 3:
        num_depths = p.shape[0]
    else:
        raise ValueError(f"Expected 3D or 4D array for depthwise skill score, got shape {p.shape}")

    scores = {}
    for d in range(num_depths):
        p_d = p[:, d, ...] if p.ndim == 4 else p[d, ...]
        t_d = t[:, d, ...] if t.ndim == 4 else t[d, ...]
        c_d = c[:, d, ...] if c.ndim == 4 else c[d, ...]

        depth_key = f"{depths[d]}m" if depths and d < len(depths) else f"depth_{d}"
        score = compute_murphy_skill_score(p_d, t_d, c_d, mask=mask)
        scores[depth_key] = score

    return scores
