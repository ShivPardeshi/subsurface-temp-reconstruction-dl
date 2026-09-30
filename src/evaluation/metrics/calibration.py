"""Uncertainty Calibration and Reliability Diagrams.

Evaluates empirical calibration of DDIM ensemble spread against ground truth
observations (GLORYS / ARGO), computing empirical coverage vs nominal confidence
intervals and Expected Calibration Error (ECE).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from .basic_metrics import _to_numpy


def evaluate_ensemble_calibration(
    ensemble_preds: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    confidence_levels: Optional[List[float]] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, Union[float, Dict[str, float], List[float]]]:
    """Evaluate empirical coverage across multiple confidence intervals.

    Args:
        ensemble_preds: (N, ...) array of N ensemble predictions (e.g. N=10 to 20 DDIM runs).
        target: (...) ground truth array matching ensemble spatial/depth dimensions.
        confidence_levels: List of nominal confidence intervals to test (default [0.50, 0.68, 0.80, 0.90, 0.95]).
        mask: Optional boolean mask (True = valid).

    Returns:
        Dictionary containing:
        - 'nominal_levels': list of nominal levels
        - 'empirical_coverages': list of empirical observed coverages
        - 'expected_calibration_error': mean |empirical - nominal|
        - 'calibration_diagnostics': status ('well_calibrated', 'overconfident', 'underconfident')
        - 'coverage_by_level': dict mapping e.g. '80%' -> coverage
    """
    if confidence_levels is None:
        confidence_levels = [0.50, 0.68, 0.80, 0.90, 0.95]

    ens = _to_numpy(ensemble_preds)
    t = _to_numpy(target)

    n_samples = ens.shape[0]
    if n_samples < 3:
        raise ValueError(f"Ensemble size must be at least 3 for quantile evaluation, got {n_samples}")

    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        if m.shape != t.shape:
            m = np.broadcast_to(m, t.shape)
        t_valid = t[m]
        # Slice each ensemble member
        ens_valid = np.array([ens[i][m] for i in range(n_samples)])
    else:
        t_valid = t.flatten()
        ens_valid = np.array([ens[i].flatten() for i in range(n_samples)])

    valid = np.isfinite(t_valid) & np.all(np.isfinite(ens_valid), axis=0)
    if not np.any(valid):
        return {
            "expected_calibration_error": float("nan"),
            "calibration_diagnostics": "invalid_data",
        }

    t_valid = t_valid[valid]
    ens_valid = ens_valid[:, valid]

    empirical_coverages = []
    coverage_by_level = {}
    abs_errors = []

    for level in confidence_levels:
        alpha = 1.0 - level
        q_low = 100.0 * (alpha / 2.0)
        q_high = 100.0 * (1.0 - alpha / 2.0)

        lower_bound = np.percentile(ens_valid, q_low, axis=0)
        upper_bound = np.percentile(ens_valid, q_high, axis=0)

        inside = (t_valid >= lower_bound) & (t_valid <= upper_bound)
        emp_cov = float(np.mean(inside))

        empirical_coverages.append(emp_cov)
        coverage_by_level[f"{int(round(level * 100))}%"] = emp_cov
        abs_errors.append(abs(emp_cov - level))

    ece = float(np.mean(abs_errors))

    # Overall diagnostic assessment
    # If empirical coverage is consistently lower than nominal, intervals are too narrow -> overconfident
    mean_diff = np.mean([emp - nom for emp, nom in zip(empirical_coverages, confidence_levels)])
    if mean_diff < -0.05:
        diagnostic = "overconfident (intervals too narrow)"
    elif mean_diff > 0.05:
        diagnostic = "underconfident (intervals too wide)"
    else:
        diagnostic = "well_calibrated"

    return {
        "nominal_levels": confidence_levels,
        "empirical_coverages": empirical_coverages,
        "expected_calibration_error": ece,
        "calibration_diagnostics": diagnostic,
        "coverage_by_level": coverage_by_level,
    }


def evaluate_depthwise_calibration(
    ensemble_preds: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    depths: Optional[List[float]] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    target_confidence: float = 0.80,
) -> Dict[str, float]:
    """Evaluate 80% CI empirical coverage across individual canonical depths.

    Args:
        ensemble_preds: (N, D, H, W)
        target: (D, H, W)
    """
    ens = _to_numpy(ensemble_preds)
    t = _to_numpy(target)

    num_depths = t.shape[0]
    depth_coverage = {}

    for d in range(num_depths):
        ens_d = ens[:, d, ...]
        t_d = t[d, ...]
        depth_key = f"{depths[d]}m" if depths and d < len(depths) else f"depth_{d}"
        res = evaluate_ensemble_calibration(
            ens_d, t_d, confidence_levels=[target_confidence], mask=mask
        )
        depth_coverage[depth_key] = res["empirical_coverages"][0]

    return depth_coverage
