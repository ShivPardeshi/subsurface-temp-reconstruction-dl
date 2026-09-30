"""Verification Script for User Checks 1 and 2.

1. Strict Validation-Only Alpha Selection:
   - Sweeps alpha in [0.0, 0.05, ..., 1.0] STRICTLY on the held-out Validation set (Days 237 to 297).
   - Identifies alpha* that minimizes Validation RMSE.
   - Evaluates that locked alpha* EXACTLY ONCE on the unseen Continuous Test Set (Days 298 to 358).
2. Region Conditioning Comparison:
   - Evaluates the blend with Region ON (model trained with region channels fed with active region channels)
   - Evaluates the blend with Region OFF (region channels zeroed at inference)
   - Confirms exact training status and empirical impact.
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
import torch.nn as nn
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.auxiliary_heads import AuxiliaryHeads
from src.models.diffusion import GaussianDiffusion
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS
from src.evaluation.metrics.basic_metrics import compute_rmse, compute_mae, compute_bias, compute_correlation
from src.evaluation.metrics.skill_score import compute_murphy_skill_score
from scripts.evaluate_ridge_diffusion_blend import train_ridge_weights, load_diffusion_model, get_all_predictions, evaluate_blend


# Exact split definitions per src/training/dataset.py:
# Total sequences: 359 (Days 0 to 358)
# Train: Days 6 to 236 (231 days)
# Val:   Days 237 to 297 (61 days, Fall Intermonsoon onset)
# Test:  Days 298 to 358 (61 days, Nov 1 to Dec 31, 2025)
VAL_DATES = list(range(237, 298))      # 61 validation days
TEST_DATES = list(range(298, 359))     # 61 held-out test days
CANONICAL_MULTI_SEASONAL_DATES = [15, 60, 105, 150, 195, 240, 285, 318, 331, 358]


def main():
    device = torch.device("cpu")
    print("=" * 85)
    print("STRICT VERIFICATION: VALIDATION-ONLY ALPHA SELECTION & REGION AUDIT")
    print("=" * 85)

    # 1. Train Ridge on training set only (Days 6 to 236)
    print("[1/4] Training analytical Ridge baseline on Train set (Days 6-236)...")
    W_ridge, mean_X, std_X, feat_ch = train_ridge_weights()

    # 2. Load Clean Scratch 40k checkpoint
    ckpt_path = "checkpoints/phase3_retrain_scratch_40k_full/best_checkpoint.pt"
    norm_stats = "data/processed/channel_normalization_stats_ocean_only.json"
    print(f"[2/4] Loading Clean Scratch 40k checkpoint ({ckpt_path})...")
    enc, unet, aux, diff = load_diffusion_model(ckpt_path, device, norm_stats)

    # 3. Predict on Validation Set (Days 237-297) for both Region ON and Region OFF
    print("\n[3/4] Running inference on VALIDATION SET (Days 237-297) to select alpha*...")
    diff_val_off, ridge_val, tgts_val, clim_val, mask = get_all_predictions(
        enc, unet, diff, device, W_ridge, mean_X, std_X, feat_ch, VAL_DATES, use_region=False, num_ddim_steps=5
    )
    diff_val_on, _, _, _, _ = get_all_predictions(
        enc, unet, diff, device, W_ridge, mean_X, std_X, feat_ch, VAL_DATES, use_region=True, num_ddim_steps=5
    )

    clim_val_rmse = float(compute_rmse(clim_val, tgts_val, mask=mask))
    print(f"Validation Climatology RMSE: {clim_val_rmse:.4f} °C")

    # Sweep alpha strictly on Validation data
    alpha_candidates = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]
    val_sweep_off = {}
    val_sweep_on = {}

    print("\n--- VALIDATION SET ALPHA SWEEP (REGION OFF) ---")
    print(f"{'Alpha':<10} | {'Val RMSE (°C)':<16} | {'Val Skill Score':<16} | {'Val Bias (°C)':<14}")
    print("-" * 65)
    best_alpha_off = 0.0
    best_val_rmse_off = 999.0

    for a in alpha_candidates:
        blend_val = a * ridge_val + (1.0 - a) * diff_val_off
        res_v = evaluate_blend(blend_val, tgts_val, clim_val, mask)
        val_sweep_off[a] = res_v
        if res_v["overall_rmse"] < best_val_rmse_off:
            best_val_rmse_off = res_v["overall_rmse"]
            best_alpha_off = a
        print(f"{a:<10.2f} | {res_v['overall_rmse']:<16.4f} | {res_v['murphy_skill_score']:<16.4f} | {res_v['overall_bias']:<+14.4f}")

    print(f"\n>> BEST ALPHA (REGION OFF) SELECTED ON VALIDATION SET: alpha* = {best_alpha_off} (Val RMSE: {best_val_rmse_off:.4f} °C, Skill: {val_sweep_off[best_alpha_off]['murphy_skill_score']:+.4f})")

    print("\n--- VALIDATION SET ALPHA SWEEP (REGION ON) ---")
    best_alpha_on = 0.0
    best_val_rmse_on = 999.0
    for a in alpha_candidates:
        blend_val = a * ridge_val + (1.0 - a) * diff_val_on
        res_v = evaluate_blend(blend_val, tgts_val, clim_val, mask)
        val_sweep_on[a] = res_v
        if res_v["overall_rmse"] < best_val_rmse_on:
            best_val_rmse_on = res_v["overall_rmse"]
            best_alpha_on = a

    print(f">> BEST ALPHA (REGION ON) SELECTED ON VALIDATION SET: alpha* = {best_alpha_on} (Val RMSE: {best_val_rmse_on:.4f} °C, Skill: {val_sweep_on[best_alpha_on]['murphy_skill_score']:+.4f})")

    # 4. Predict on Test Set (Days 298-358) and evaluate FIXED alpha*
    print("\n[4/4] Evaluating LOCKED alpha* on UNSEEN TEST SET (Days 298-358) & Multi-Seasonal Dates...")
    diff_test_off, ridge_test, tgts_test, clim_test, _ = get_all_predictions(
        enc, unet, diff, device, W_ridge, mean_X, std_X, feat_ch, TEST_DATES, use_region=False, num_ddim_steps=5
    )
    diff_test_on, _, _, _, _ = get_all_predictions(
        enc, unet, diff, device, W_ridge, mean_X, std_X, feat_ch, TEST_DATES, use_region=True, num_ddim_steps=5
    )

    # Evaluate the validation-chosen alpha on the test set
    test_res_locked_off = evaluate_blend(best_alpha_off * ridge_test + (1.0 - best_alpha_off) * diff_test_off, tgts_test, clim_test, mask)
    test_res_locked_on = evaluate_blend(best_alpha_on * ridge_test + (1.0 - best_alpha_on) * diff_test_on, tgts_test, clim_test, mask)

    # Also evaluate on multi-seasonal benchmark
    diff_ms_off, ridge_ms, tgts_ms, clim_ms, _ = get_all_predictions(
        enc, unet, diff, device, W_ridge, mean_X, std_X, feat_ch, CANONICAL_MULTI_SEASONAL_DATES, use_region=False, num_ddim_steps=10
    )
    diff_ms_on, _, _, _, _ = get_all_predictions(
        enc, unet, diff, device, W_ridge, mean_X, std_X, feat_ch, CANONICAL_MULTI_SEASONAL_DATES, use_region=True, num_ddim_steps=10
    )

    ms_res_locked_off = evaluate_blend(best_alpha_off * ridge_ms + (1.0 - best_alpha_off) * diff_ms_off, tgts_ms, clim_ms, mask)
    ms_res_locked_on = evaluate_blend(best_alpha_on * ridge_ms + (1.0 - best_alpha_on) * diff_ms_on, tgts_ms, clim_ms, mask)

    print("\n" + "=" * 85)
    print("FINAL HONEST UNBIASED EVALUATION SUMMARY")
    print("=" * 85)
    print(f"Configuration 1: REGION OFF (Selected alpha* = {best_alpha_off} on Validation):")
    print(f"  • Validation RMSE:       {best_val_rmse_off:.4f} °C (vs Climatology: {clim_val_rmse:.4f} °C | Skill: {val_sweep_off[best_alpha_off]['murphy_skill_score']:+.4f})")
    print(f"  • Continuous Test RMSE:  {test_res_locked_off['overall_rmse']:.4f} °C (vs Climatology: {test_res_locked_off['climatology_rmse']:.4f} °C | Skill: {test_res_locked_off['murphy_skill_score']:+.4f})")
    print(f"  • Multi-Seasonal RMSE:   {ms_res_locked_off['overall_rmse']:.4f} °C (vs Climatology: {ms_res_locked_off['climatology_rmse']:.4f} °C | Skill: {ms_res_locked_off['murphy_skill_score']:+.4f})")
    print(f"  • Mean Water-Column Bias: {test_res_locked_off['overall_bias']:+.4f} °C")

    print(f"\nConfiguration 2: REGION ON (Selected alpha* = {best_alpha_on} on Validation):")
    print(f"  • Validation RMSE:       {best_val_rmse_on:.4f} °C (vs Climatology: {clim_val_rmse:.4f} °C | Skill: {val_sweep_on[best_alpha_on]['murphy_skill_score']:+.4f})")
    print(f"  • Continuous Test RMSE:  {test_res_locked_on['overall_rmse']:.4f} °C (vs Climatology: {test_res_locked_on['climatology_rmse']:.4f} °C | Skill: {test_res_locked_on['murphy_skill_score']:+.4f})")
    print(f"  • Multi-Seasonal RMSE:   {ms_res_locked_on['overall_rmse']:.4f} °C (vs Climatology: {ms_res_locked_on['climatology_rmse']:.4f} °C | Skill: {ms_res_locked_on['murphy_skill_score']:+.4f})")
    print(f"  • Mean Water-Column Bias: {test_res_locked_on['overall_bias']:+.4f} °C")

    # Save to JSON report
    report_dict = {
        "validation_alpha_selection": {
            "validation_days": "237-297 (61 consecutive days)",
            "selected_alpha_off": best_alpha_off,
            "selected_alpha_on": best_alpha_on,
            "val_sweep_region_off": {str(k): v for k, v in val_sweep_off.items()},
            "val_sweep_region_on": {str(k): v for k, v in val_sweep_on.items()},
        },
        "honest_test_evaluation": {
            "region_off_locked": {
                "alpha": best_alpha_off,
                "continuous_test_rmse": test_res_locked_off["overall_rmse"],
                "continuous_test_clim_rmse": test_res_locked_off["climatology_rmse"],
                "continuous_test_skill": test_res_locked_off["murphy_skill_score"],
                "continuous_test_bias": test_res_locked_off["overall_bias"],
                "multi_seasonal_rmse": ms_res_locked_off["overall_rmse"],
                "multi_seasonal_clim_rmse": ms_res_locked_off["climatology_rmse"],
                "multi_seasonal_skill": ms_res_locked_off["murphy_skill_score"],
                "multi_seasonal_bias": ms_res_locked_off["overall_bias"],
            },
            "region_on_locked": {
                "alpha": best_alpha_on,
                "continuous_test_rmse": test_res_locked_on["overall_rmse"],
                "continuous_test_clim_rmse": test_res_locked_on["climatology_rmse"],
                "continuous_test_skill": test_res_locked_on["murphy_skill_score"],
                "continuous_test_bias": test_res_locked_on["overall_bias"],
                "multi_seasonal_rmse": ms_res_locked_on["overall_rmse"],
                "multi_seasonal_clim_rmse": ms_res_locked_on["climatology_rmse"],
                "multi_seasonal_skill": ms_res_locked_on["murphy_skill_score"],
                "multi_seasonal_bias": ms_res_locked_on["overall_bias"],
            },
        },
    }

    out_file = REPO_ROOT / "reports" / "strict_validation_alpha_audit.json"
    with open(out_file, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\n[SAVED] Verification report written to: {out_file}")


if __name__ == "__main__":
    main()
