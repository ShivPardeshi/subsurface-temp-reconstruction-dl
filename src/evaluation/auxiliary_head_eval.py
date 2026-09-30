"""Auxiliary Head Evaluation Module.

Evaluates predictions from Stage 5 Auxiliary physical prediction heads:
1. Mixed Layer Depth (MLD) in meters.
2. Bay of Bengal Barrier Layer Thickness (BLT) in meters (evaluated within BoB).
3. Arabian Sea Subsurface Salinity Maximum depth (m) and strength anomaly (PSU) (evaluated within AS).
"""

from typing import Dict, Optional, Union
import numpy as np
import torch
from src.evaluation.metrics.basic_metrics import compute_mae, compute_rmse, compute_correlation, compute_bias, _to_numpy


def evaluate_auxiliary_predictions(
    pred_aux: Dict[str, Union[np.ndarray, torch.Tensor]],
    true_aux: Dict[str, Union[np.ndarray, torch.Tensor]],
    region_weights: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute physical metrics for all 3 auxiliary prediction heads.

    Args:
        pred_aux: Dict with keys: 'mld', 'blt', 'sal_max_depth', 'sal_max_strength'.
        true_aux: Ground truth dict with matching keys.
        region_weights: Optional dict with 'bob_weight' and 'as_weight' arrays of shape (B,).

    Returns:
        Structured dictionary of evaluation metrics per auxiliary target.
    """
    results = {}

    # 1. Mixed Layer Depth (MLD) - Global
    if "mld" in pred_aux and "mld" in true_aux:
        p_mld = _to_numpy(pred_aux["mld"]).flatten()
        t_mld = _to_numpy(true_aux["mld"]).flatten()
        results["mixed_layer_depth"] = {
            "mae_m": compute_mae(p_mld, t_mld),
            "rmse_m": compute_rmse(p_mld, t_mld),
            "bias_m": compute_bias(p_mld, t_mld),
            "correlation": compute_correlation(p_mld, t_mld),
            "unit": "meters",
        }

    # 2. Bay of Bengal Barrier Layer Thickness (BLT)
    if "blt" in pred_aux and "blt" in true_aux:
        p_blt = _to_numpy(pred_aux["blt"]).flatten()
        t_blt = _to_numpy(true_aux["blt"]).flatten()
        mask = None
        if region_weights and "bob_weight" in region_weights:
            bw = _to_numpy(region_weights["bob_weight"]).flatten()
            mask = bw > 0.25

        results["bob_barrier_layer_thickness"] = {
            "mae_m": compute_mae(p_blt, t_blt, mask=mask),
            "rmse_m": compute_rmse(p_blt, t_blt, mask=mask),
            "bias_m": compute_bias(p_blt, t_blt, mask=mask),
            "correlation": compute_correlation(p_blt, t_blt, mask=mask),
            "domain": "Bay of Bengal (weighted)",
            "unit": "meters",
        }

    # 3. Arabian Sea Salinity Maximum (Depth & Strength)
    as_mask = None
    if region_weights and "as_weight" in region_weights:
        aw = _to_numpy(region_weights["as_weight"]).flatten()
        as_mask = aw > 0.25

    if "sal_max_depth" in pred_aux and "sal_max_depth" in true_aux:
        p_smd = _to_numpy(pred_aux["sal_max_depth"]).flatten()
        t_smd = _to_numpy(true_aux["sal_max_depth"]).flatten()
        p_std = float(np.nanstd(p_smd))
        t_std = float(np.nanstd(t_smd))
        diag = "ok" if p_std >= 0.1 else "head_variance_collapse (near-constant prediction)"
        results["as_salinity_max_depth"] = {
            "mae_m": compute_mae(p_smd, t_smd, mask=as_mask),
            "rmse_m": compute_rmse(p_smd, t_smd, mask=as_mask),
            "bias_m": compute_bias(p_smd, t_smd, mask=as_mask),
            "correlation": compute_correlation(p_smd, t_smd, mask=as_mask),
            "pred_mean_m": float(np.nanmean(p_smd)),
            "true_mean_m": float(np.nanmean(t_smd)),
            "pred_std_m": p_std,
            "true_std_m": t_std,
            "diagnostic": diag,
            "domain": "Arabian Sea (weighted)",
            "unit": "meters",
        }

    if "sal_max_strength" in pred_aux and "sal_max_strength" in true_aux:
        p_sms = _to_numpy(pred_aux["sal_max_strength"]).flatten()
        t_sms = _to_numpy(true_aux["sal_max_strength"]).flatten()
        results["as_salinity_max_strength"] = {
            "mae_psu": compute_mae(p_sms, t_sms, mask=as_mask),
            "rmse_psu": compute_rmse(p_sms, t_sms, mask=as_mask),
            "bias_psu": compute_bias(p_sms, t_sms, mask=as_mask),
            "correlation": compute_correlation(p_sms, t_sms, mask=as_mask),
            "pred_mean_psu": float(np.nanmean(p_sms)),
            "true_mean_psu": float(np.nanmean(t_sms)),
            "domain": "Arabian Sea (weighted)",
            "unit": "PSU",
        }

    return results
