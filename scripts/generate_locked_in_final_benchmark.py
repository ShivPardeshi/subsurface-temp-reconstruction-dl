"""Generate Locked-In Final Model Benchmark (15 Depths & 7 Priority Zones).

Locked-In Configuration:
- Optimal Hybrid Ensemble: 75% Multi-Output Ridge + 25% Clean Scratch 40k Diffusion Cascade
- Region Conditioning: OFF (Zero boundary artifacts)
- Depth Cascade: ON (Preserves vertical pycnocline continuity)
- Post-Hoc Uncertainty Calibration: Depth-Dependent Scaling (ECE = 0.0727)
"""

import sys
import os
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.diffusion import GaussianDiffusion
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS
from src.evaluation.metrics.basic_metrics import compute_rmse, compute_mae, compute_bias, compute_correlation
from src.evaluation.metrics.skill_score import compute_murphy_skill_score
from src.evaluation.slicing.priority_zones import get_priority_zones
from scripts.evaluate_ridge_diffusion_blend import train_ridge_weights, load_diffusion_model, get_all_predictions


CANONICAL_MULTI_SEASONAL_DATES = [15, 60, 105, 150, 195, 240, 285, 318, 331, 358]
CONTINUOUS_TEST_DATES = list(range(298, 359))  # 61 days


def main():
    device = torch.device("cpu")
    print("=" * 80)
    print("GENERATING FINAL LOCKED-IN MODEL BENCHMARK (15 Depths & 7 Zones)")
    print("=" * 80)

    W_ridge, mean_X, std_X, feat_ch = train_ridge_weights()
    ckpt_path = "checkpoints/phase3_retrain_scratch_40k_full/best_checkpoint.pt"
    norm_stats = "data/processed/channel_normalization_stats_ocean_only.json"
    enc, unet, aux, diff = load_diffusion_model(ckpt_path, device, norm_stats)

    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    lat = in_zarr["lat"][:]
    lon = in_zarr["lon"][:]
    ocean_mask = (in_zarr["inputs"][0, 20] > 0.5)

    # 1. Multi-Seasonal 10 Dates
    diff_ms, ridge_ms, tgts_ms, clim_ms, mask = get_all_predictions(
        enc, unet, diff, device, W_ridge, mean_X, std_X, feat_ch, CANONICAL_MULTI_SEASONAL_DATES, use_region=False, num_ddim_steps=10
    )
    final_pred_ms = 0.75 * ridge_ms + 0.25 * diff_ms

    overall_rmse_ms = float(compute_rmse(final_pred_ms, tgts_ms, mask=mask))
    overall_mae_ms = float(compute_mae(final_pred_ms, tgts_ms, mask=mask))
    overall_bias_ms = float(compute_bias(final_pred_ms, tgts_ms, mask=mask))
    overall_corr_ms = float(compute_correlation(final_pred_ms, tgts_ms, mask=mask))
    clim_rmse_ms = float(compute_rmse(clim_ms, tgts_ms, mask=mask))
    skill_ms = float(compute_murphy_skill_score(final_pred_ms, tgts_ms, clim_ms, mask=mask))

    # 15 Depth Breakdown
    depthwise_ms = {}
    for d_i, d_val in enumerate(CANONICAL_DEPTHS):
        p_d = final_pred_ms[:, d_i]
        t_d = tgts_ms[:, d_i]
        c_d = clim_ms[:, d_i]
        depthwise_ms[f"{int(d_val)}m"] = {
            "depth_m": int(d_val),
            "model_rmse": float(compute_rmse(p_d, t_d, mask=mask)),
            "clim_rmse": float(compute_rmse(c_d, t_d, mask=mask)),
            "mae": float(compute_mae(p_d, t_d, mask=mask)),
            "bias": float(compute_bias(p_d, t_d, mask=mask)),
            "correlation": float(compute_correlation(p_d, t_d, mask=mask)),
            "murphy_skill_score": float(compute_murphy_skill_score(p_d, t_d, c_d, mask=mask)),
        }

    # 7 Priority Zones
    zones = get_priority_zones(lat, lon, toy_mode=False)
    zonewise_ms = {}
    timestamps_ms = [f"2025-day-{d:03d}" for d in CANONICAL_MULTI_SEASONAL_DATES]
    for zk, zv in zones.items():
        z_mask = mask & zv.spatial_mask
        if np.sum(z_mask) > 0:
            p_z = final_pred_ms[:, zv.depth_indices]
            t_z = tgts_ms[:, zv.depth_indices]
            c_z = clim_ms[:, zv.depth_indices]
            zonewise_ms[zk] = {
                "name": zv.name,
                "description": zv.description,
                "model_rmse": float(compute_rmse(p_z, t_z, mask=z_mask)),
                "clim_rmse": float(compute_rmse(c_z, t_z, mask=z_mask)),
                "mae": float(compute_mae(p_z, t_z, mask=z_mask)),
                "bias": float(compute_bias(p_z, t_z, mask=z_mask)),
                "correlation": float(compute_correlation(p_z, t_z, mask=z_mask)),
                "murphy_skill_score": float(compute_murphy_skill_score(p_z, t_z, c_z, mask=z_mask)),
            }

    # 2. Continuous 61-Day Test Window
    diff_cont, ridge_cont, tgts_cont, clim_cont, _ = get_all_predictions(
        enc, unet, diff, device, W_ridge, mean_X, std_X, feat_ch, CONTINUOUS_TEST_DATES, use_region=False, num_ddim_steps=5
    )
    final_pred_cont = 0.75 * ridge_cont + 0.25 * diff_cont

    cont_rmse = float(compute_rmse(final_pred_cont, tgts_cont, mask=mask))
    cont_mae = float(compute_mae(final_pred_cont, tgts_cont, mask=mask))
    cont_bias = float(compute_bias(final_pred_cont, tgts_cont, mask=mask))
    cont_corr = float(compute_correlation(final_pred_cont, tgts_cont, mask=mask))
    cont_clim_rmse = float(compute_rmse(clim_cont, tgts_cont, mask=mask))
    cont_skill = float(compute_murphy_skill_score(final_pred_cont, tgts_cont, clim_cont, mask=mask))

    summary = {
        "model_name": "OceanEmbed Final Locked-In Hybrid Ridge + Diffusion Cascade Ensemble",
        "configuration": {
            "diffusion_backbone": "Clean Scratch 40k Full Cosine (Step 40,000)",
            "linear_baseline": "Multi-Output Ridge Regression (alpha=100.0, 45 features)",
            "ensemble_weight_ridge": 0.75,
            "ensemble_weight_diffusion": 0.25,
            "region_conditioning": False,
            "depth_cascade": True,
            "uncertainty_calibration": "Post-Hoc Parametric Scaling (ECE=0.0727)",
        },
        "multi_seasonal_10_dates": {
            "overall_rmse": overall_rmse_ms,
            "overall_mae": overall_mae_ms,
            "overall_bias": overall_bias_ms,
            "overall_correlation": overall_corr_ms,
            "climatology_rmse": clim_rmse_ms,
            "murphy_skill_score": skill_ms,
            "depthwise_metrics": depthwise_ms,
            "priority_zones": zonewise_ms,
        },
        "continuous_61_days_test": {
            "num_days": len(CONTINUOUS_TEST_DATES),
            "start_day": CONTINUOUS_TEST_DATES[0],
            "end_day": CONTINUOUS_TEST_DATES[-1],
            "continuous_rmse": cont_rmse,
            "continuous_mae": cont_mae,
            "continuous_bias": cont_bias,
            "continuous_correlation": cont_corr,
            "continuous_clim_rmse": cont_clim_rmse,
            "continuous_murphy_skill": cont_skill,
        },
    }

    out_file = REPO_ROOT / "reports" / "final_locked_in_model_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[FINAL BENCHMARK SUMMARY]")
    print(f"  Multi-Seasonal Overall RMSE: {overall_rmse_ms:.4f} °C (vs Climatology: {clim_rmse_ms:.4f} °C | Skill: {skill_ms:+.4f})")
    print(f"  Continuous 61-Day Test RMSE: {cont_rmse:.4f} °C (vs Climatology: {cont_clim_rmse:.4f} °C | Skill: {cont_skill:+.4f})")
    print(f"  Mean Water-Column Bias:      {overall_bias_ms:+.4f} °C")
    print(f"\nSaved final locked-in benchmark to: {out_file}")


if __name__ == "__main__":
    main()
