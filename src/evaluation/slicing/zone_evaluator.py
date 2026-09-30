"""Generic Zone Evaluator Module.

Applies any metric function identically across the 7 canonical priority zones
(per Section 8 of Phase 5 specification).
"""

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from .priority_zones import PriorityZone, get_priority_zones
from src.evaluation.metrics.basic_metrics import _to_numpy


def evaluate_metric_by_zone(
    metric_fn: Callable[..., Any],
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    zones: Optional[Dict[str, PriorityZone]] = None,
    dates: Optional[Union[pd.DatetimeIndex, np.ndarray, List[str]]] = None,
    clim: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ocean_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    timestamps: Optional[Union[pd.DatetimeIndex, np.ndarray, List[str]]] = None,
    **metric_kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate any arbitrary metric sliced across all priority zones.

    Args:
        metric_fn: Metric function accepting (pred, target, mask=..., **metric_kwargs).
        pred: (D, H, W) or (B, D, H, W) predictions.
        target: (D, H, W) or (B, D, H, W) ground truth.
        zones: Dictionary of {zone_key: PriorityZone}. If None, loads defaults.
        dates: Optional array of timestamps matching batch dimension B.
        clim: Optional climatology field for skill score or R2.
        ocean_mask: Optional base land/ocean mask (True = ocean).
        timestamps: Alias for dates parameter.
        **metric_kwargs: Extra keyword arguments forwarded to metric_fn.

    Returns:
        Dictionary mapping zone_key -> metric output.
    """
    if dates is None and timestamps is not None:
        dates = timestamps

    p = _to_numpy(pred)
    t = _to_numpy(target)
    c = _to_numpy(clim) if clim is not None else None

    # Handle 3D (D, H, W) vs 4D (B, D, H, W)
    is_batched = (p.ndim == 4)

    if zones is None:
        h, w = p.shape[-2], p.shape[-1]
        toy_mode = (h < 60)
        zones = get_priority_zones(toy_mode=toy_mode)

    base_ocean_mask = _to_numpy(ocean_mask).astype(bool) if ocean_mask is not None else None

    results = {}

    for zone_key, zone in zones.items():
        # 1. Temporal filtering if applicable
        if zone.time_filter_fn is not None:
            if dates is None:
                results[zone_key] = {
                    "status": "dates_required_for_temporal_zone",
                    "error": f"Evaluation inputs must provide timestamps for {zone.name}",
                    "rmse": float("nan"),
                }
                continue

            time_mask = zone.time_filter_fn(dates)
            if not np.any(time_mask):
                # No samples match this time window in the current batch
                results[zone_key] = {
                    "status": "no_events_in_sample",
                    "error": f"No dates in sample coincide with {zone.name}",
                    "rmse": float("nan"),
                }
                continue

            if is_batched:
                p_slice = p[time_mask]
                t_slice = t[time_mask]
                c_slice = c[time_mask] if c is not None else None
            else:
                # Single sample with 1 date matching
                p_slice = p
                t_slice = t
                c_slice = c
        else:
            p_slice = p
            t_slice = t
            c_slice = c

        # 2. Vertical depth filtering
        depth_idx = zone.depth_indices
        if is_batched:
            p_sub = p_slice[:, depth_idx, ...]
            t_sub = t_slice[:, depth_idx, ...]
            c_sub = c_slice[:, depth_idx, ...] if c_slice is not None else None
        else:
            p_sub = p_slice[depth_idx, ...]
            t_sub = t_slice[depth_idx, ...]
            c_sub = c_slice[depth_idx, ...] if c_slice is not None else None

        # 3. Spatial masking
        zone_spatial = zone.spatial_mask
        if base_ocean_mask is not None:
            combined_mask = zone_spatial & base_ocean_mask
        else:
            combined_mask = zone_spatial

        if not np.any(combined_mask):
            results[zone_key] = {
                "status": "empty_spatial_mask",
                "error": f"Zone '{zone.name}' is outside the spatial bounds of the active evaluation grid",
                "rmse": float("nan"),
                "correlation": float("nan"),
                "mae": float("nan"),
            }
            continue

        # Broadcast 2D mask to match sliced shape if needed
        # Sliced spatial dimensions are (-2, -1)
        # Metric functions handle mask broadcasting or 2D masks automatically
        kwargs = dict(metric_kwargs)
        if c_sub is not None:
            kwargs["clim"] = c_sub

        try:
            val = metric_fn(p_sub, t_sub, mask=combined_mask, **kwargs)
            results[zone_key] = val
        except Exception as e:
            results[zone_key] = {"error": str(e)}

    return results
