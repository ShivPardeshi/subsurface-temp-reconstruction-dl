"""Comprehensive Category A Analysis Script.

Executes:
1. Metric Reconciliation across all models (Multi-Seasonal 10 Dates vs Continuous 61 Days vs Proxy).
2. Granular 15-Depth and 7-Zone Skill Score Breakdown for Clean Scratch 40k, Warm-Started 40k, Baseline 20k.
3. Side-by-Side Uncertainty Calibration Verification for Clean Scratch 40k vs Warm-Started 40k.
4. Continuous Nov-Dec Test Evaluation for Clean Scratch 40k.
"""

import sys
import os
import math
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

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
from src.evaluation.metrics.basic_metrics import (
    compute_rmse,
    compute_mae,
    compute_bias,
    compute_correlation,
    compute_all_basic_metrics,
)
from src.evaluation.metrics.skill_score import compute_murphy_skill_score
from src.evaluation.slicing.priority_zones import get_priority_zones
from src.evaluation.slicing.zone_evaluator import evaluate_metric_by_zone
from src.evaluation.calibration_scaling import PostHocCalibrator
from src.evaluation.metrics.calibration import evaluate_ensemble_calibration


CANONICAL_MULTI_SEASONAL_DATES = [15, 60, 105, 150, 195, 240, 285, 318, 331, 358]
CONTINUOUS_TEST_DATES = list(range(298, 359))  # 61 days: 298 to 358 inclusive


def load_model(
    checkpoint_path: str,
    device: torch.device,
    normalize_inputs: bool,
    norm_stats_path: Optional[str],
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    context_encoder = ContextEncoder(
        in_channels=25,
        hidden_dims=[32, 64, 64],
        normalize_inputs=normalize_inputs,
        norm_stats_path=norm_stats_path,
    ).to(device)
    unet = UNetDenoiser(in_channels=72, stage_channels=[32, 64, 128, 256], cond_in_dim=8).to(device)
    aux_heads = AuxiliaryHeads(in_features=64).to(device)
    diffusion = GaussianDiffusion(timesteps=1000, schedule_type="cosine").to(device)

    if "models" in ckpt:
        context_encoder.load_state_dict(ckpt["models"]["context_encoder"], strict=False)
        unet.load_state_dict(ckpt["models"]["unet"])
        aux_heads.load_state_dict(ckpt["models"]["aux_heads"])
    elif "model_state_dict" in ckpt:
        context_encoder.load_state_dict(ckpt["model_state_dict"].get("context_encoder", {}), strict=False)
        unet.load_state_dict(ckpt["model_state_dict"].get("unet", {}))
        aux_heads.load_state_dict(ckpt["model_state_dict"].get("aux_heads", {}))

    context_encoder.eval()
    unet.eval()
    aux_heads.eval()
    return context_encoder, unet, aux_heads, diffusion, ckpt


def run_full_inference(
    context_encoder: nn.Module,
    unet: nn.Module,
    diffusion: GaussianDiffusion,
    device: torch.device,
    target_dates: List[int],
    num_ddim_timesteps: int = 10,
    eta: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Runs full profile inference over specified dates.
    
    Returns:
        preds_arr: (N_dates, 15, H, W) in physical °C
        targets_arr: (N_dates, 15, H, W) in physical °C
        clim_arr: (N_dates, 15, H, W) in physical °C
        ocean_mask: (H, W) boolean
        timestamps: List of date strings
    """
    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    lat = in_zarr["lat"][:]
    lon = in_zarr["lon"][:]
    ocean_mask_np = (in_zarr["inputs"][0, 20] > 0.5)
    clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)
    static_channels = [20, 19, 21, 22, 23, 24]
    omega = 2.0 * math.pi / 365.25

    ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=num_ddim_timesteps, eta=eta)
    cascade = DepthCascadeSampler(
        context_encoder=context_encoder,
        unet_denoiser=unet,
        ddim_sampler=ddim,
        depths=CANONICAL_DEPTHS,
    )

    all_preds = []
    all_targets = []
    all_clim = []
    timestamps = []

    with torch.no_grad():
        for t_day in target_dates:
            doy = int(scalar_df.loc[t_day, "day_of_year"])
            sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
            cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
            oni = float(scalar_df.loc[t_day, "oni_index"])
            iod = float(scalar_df.loc[t_day, "iod_dmi_index"])
            date_str = str(scalar_df.loc[t_day, "date"]) if "date" in scalar_df.columns else f"2025-day-{t_day:03d}"
            timestamps.append(date_str)

            seq_slice = in_zarr["inputs"][t_day - 6 : t_day + 1]
            seq_slice = np.nan_to_num(seq_slice, nan=0.0)
            x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)
            static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
            scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], device=device, dtype=torch.float32)

            out = cascade.sample_full_profile(
                x_seq=x_seq,
                static_features=static_feats,
                scalar_conditions=scalar_cond,
                use_cascade=True,
            )
            pred_anom = out["anomalies"][0].cpu().numpy()

            clim_t = (
                clim_coeffs[0]
                + clim_coeffs[1] * math.cos(omega * doy)
                + clim_coeffs[2] * math.sin(omega * doy)
                + clim_coeffs[3] * math.cos(2 * omega * doy)
                + clim_coeffs[4] * math.sin(2 * omega * doy)
            )

            true_anom = np.nan_to_num(tgt_zarr["anomaly"][t_day].copy(), nan=0.0)
            true_anom[0] = true_anom[1]

            pred_temp = pred_anom + clim_t
            true_temp = true_anom + clim_t

            all_preds.append(pred_temp)
            all_targets.append(true_temp)
            all_clim.append(clim_t)

    preds_arr = np.stack(all_preds, axis=0)
    targets_arr = np.stack(all_targets, axis=0)
    clim_arr = np.stack(all_clim, axis=0)
    return preds_arr, targets_arr, clim_arr, ocean_mask_np, lat, lon, timestamps


def compute_granular_depth_metrics(
    preds: np.ndarray, targets: np.ndarray, clim: np.ndarray, mask: np.ndarray
) -> Dict[str, Dict[str, float]]:
    depthwise = {}
    for d_idx, d_val in enumerate(CANONICAL_DEPTHS):
        p_d = preds[:, d_idx]
        t_d = targets[:, d_idx]
        c_d = clim[:, d_idx]

        rmse_m = float(compute_rmse(p_d, t_d, mask=mask))
        rmse_c = float(compute_rmse(c_d, t_d, mask=mask))
        mae_m = float(compute_mae(p_d, t_d, mask=mask))
        bias_m = float(compute_bias(p_d, t_d, mask=mask))
        corr_m = float(compute_correlation(p_d, t_d, mask=mask))
        skill = float(compute_murphy_skill_score(p_d, t_d, c_d, mask=mask))

        depthwise[f"{int(d_val)}m"] = {
            "depth_m": int(d_val),
            "model_rmse": rmse_m,
            "clim_rmse": rmse_c,
            "mae": mae_m,
            "bias": bias_m,
            "correlation": corr_m,
            "murphy_skill_score": skill,
        }
    return depthwise


def compute_granular_zone_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    clim: np.ndarray,
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    timestamps: List[str],
) -> Dict[str, Dict[str, float]]:
    zones = get_priority_zones(lat, lon, toy_mode=False)
    zone_results = {}

    for z_key, zone in zones.items():
        z_mask = mask & zone.spatial_mask
        d_indices = zone.depth_indices

        # Check time filter
        if zone.time_filter_fn is not None:
            t_mask = zone.time_filter_fn(timestamps)
            if not np.any(t_mask):
                t_mask = np.ones(len(timestamps), dtype=bool)
        else:
            t_mask = np.ones(len(timestamps), dtype=bool)

        p_sub = preds[t_mask][:, d_indices]
        t_sub = targets[t_mask][:, d_indices]
        c_sub = clim[t_mask][:, d_indices]

        if np.sum(z_mask) == 0:
            continue

        rmse_m = float(compute_rmse(p_sub, t_sub, mask=z_mask))
        rmse_c = float(compute_rmse(c_sub, t_sub, mask=z_mask))
        mae_m = float(compute_mae(p_sub, t_sub, mask=z_mask))
        bias_m = float(compute_bias(p_sub, t_sub, mask=z_mask))
        corr_m = float(compute_correlation(p_sub, t_sub, mask=z_mask))
        skill = float(compute_murphy_skill_score(p_sub, t_sub, c_sub, mask=z_mask))

        zone_results[z_key] = {
            "name": zone.name,
            "description": zone.description,
            "model_rmse": rmse_m,
            "clim_rmse": rmse_c,
            "mae": mae_m,
            "bias": bias_m,
            "correlation": corr_m,
            "murphy_skill_score": skill,
        }
    return zone_results


def evaluate_calibration_for_model(
    context_encoder: nn.Module,
    unet: nn.Module,
    diffusion: GaussianDiffusion,
    device: torch.device,
    target_dates: List[int],
    num_ens: int = 5,
) -> Dict[str, Any]:
    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    ocean_mask_np = (in_zarr["inputs"][0, 20] > 0.5)
    clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)
    static_channels = [20, 19, 21, 22, 23, 24]
    omega = 2.0 * math.pi / 365.25

    all_ens_preds = []  # (N_ens, N_dates, D, H, W)
    all_targets = []    # (N_dates, D, H, W)

    with torch.no_grad():
        for t_day in target_dates:
            doy = int(scalar_df.loc[t_day, "day_of_year"])
            sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
            cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
            oni = float(scalar_df.loc[t_day, "oni_index"])
            iod = float(scalar_df.loc[t_day, "iod_dmi_index"])

            seq_slice = in_zarr["inputs"][t_day - 6 : t_day + 1]
            seq_slice = np.nan_to_num(seq_slice, nan=0.0)
            x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)
            static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
            scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], device=device, dtype=torch.float32)

            clim_t = (
                clim_coeffs[0]
                + clim_coeffs[1] * math.cos(omega * doy)
                + clim_coeffs[2] * math.sin(omega * doy)
                + clim_coeffs[3] * math.cos(2 * omega * doy)
                + clim_coeffs[4] * math.sin(2 * omega * doy)
            )

            true_anom = np.nan_to_num(tgt_zarr["anomaly"][t_day].copy(), nan=0.0)
            true_anom[0] = true_anom[1]
            true_temp = true_anom + clim_t
            all_targets.append(true_temp)

            date_ens = []
            for ens_i in range(num_ens):
                torch.manual_seed(42 + ens_i * 100)
                ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=10, eta=0.3)
                cascade = DepthCascadeSampler(
                    context_encoder=context_encoder,
                    unet_denoiser=unet,
                    ddim_sampler=ddim,
                    depths=CANONICAL_DEPTHS,
                )
                out = cascade.sample_full_profile(
                    x_seq=x_seq,
                    static_features=static_feats,
                    scalar_conditions=scalar_cond,
                    use_cascade=True,
                )
                p_anom = out["anomalies"][0].cpu().numpy()
                p_temp = p_anom + clim_t
                date_ens.append(p_temp)

            all_ens_preds.append(date_ens)

    ens_arr = np.array(all_ens_preds)       # (N_dates, N_ens, D, H, W)
    ens_arr = np.swapaxes(ens_arr, 0, 1)    # (N_ens, N_dates, D, H, W)
    tgt_arr = np.stack(all_targets, axis=0) # (N_dates, D, H, W)

    raw_cal = evaluate_ensemble_calibration(ens_arr, tgt_arr, mask=ocean_mask_np)

    calibrator = PostHocCalibrator(canonical_depths=CANONICAL_DEPTHS)
    fit_params = calibrator.fit(ens_arr, tgt_arr, mask=ocean_mask_np)
    ens_cal = calibrator.calibrate_ensemble(ens_arr)
    cal_cal = evaluate_ensemble_calibration(ens_cal, tgt_arr, mask=ocean_mask_np)

    return {
        "raw_calibration": raw_cal,
        "fitted_parameters": fit_params,
        "calibrated_calibration": cal_cal,
    }


def main():
    device = torch.device("cpu")
    print("=" * 80)
    print("CATEGORY A: FULL METHODOLOGY RECONCILIATION, 15-DEPTH & 7-ZONE BREAKDOWN")
    print("=" * 80)

    models_config = {
        "clean_scratch_40k_full_cosine": {
            "label": "Clean Scratch 40k (Single-Pass Full Cosine)",
            "checkpoint_path": "checkpoints/phase3_retrain_scratch_40k_full/best_checkpoint.pt",
            "normalize_inputs": True,
            "norm_stats_path": "data/processed/channel_normalization_stats_ocean_only.json",
        },
        "warm_started_40k": {
            "label": "Warm-Started 40k (All-Grid Norm)",
            "checkpoint_path": "checkpoints/phase3_retrain_normalized/step40000_checkpoint.pt",
            "normalize_inputs": True,
            "norm_stats_path": "data/processed/channel_normalization_stats.json",
        },
        "baseline_20k": {
            "label": "Baseline 20k (Unnormalized Fix A2)",
            "checkpoint_path": "checkpoints/baseline_20k_fixA2_seed42/best_checkpoint.pt",
            "normalize_inputs": False,
            "norm_stats_path": None,
        },
    }

    full_report = {}

    for m_key, m_info in models_config.items():
        print(f"\n========================================================")
        print(f"EVALUATING: {m_info['label']}")
        print(f"Checkpoint: {m_info['checkpoint_path']}")
        print(f"========================================================")

        enc, unet, aux, diff, _ = load_model(
            m_info["checkpoint_path"],
            device=device,
            normalize_inputs=m_info["normalize_inputs"],
            norm_stats_path=m_info["norm_stats_path"],
        )

        # 1. Multi-Seasonal Evaluation (10 Dates)
        print("\n--- Running Multi-Seasonal 10-Date Evaluation ---")
        preds_ms, tgts_ms, clim_ms, mask, lat, lon, timestamps_ms = run_full_inference(
            enc, unet, diff, device, CANONICAL_MULTI_SEASONAL_DATES, num_ddim_timesteps=10, eta=0.0
        )

        overall_rmse = float(compute_rmse(preds_ms, tgts_ms, mask=mask))
        overall_mae = float(compute_mae(preds_ms, tgts_ms, mask=mask))
        overall_bias = float(compute_bias(preds_ms, tgts_ms, mask=mask))
        overall_corr = float(compute_correlation(preds_ms, tgts_ms, mask=mask))
        clim_rmse = float(compute_rmse(clim_ms, tgts_ms, mask=mask))
        murphy_skill = float(compute_murphy_skill_score(preds_ms, tgts_ms, clim_ms, mask=mask))

        depthwise_metrics = compute_granular_depth_metrics(preds_ms, tgts_ms, clim_ms, mask)
        zone_metrics = compute_granular_zone_metrics(preds_ms, tgts_ms, clim_ms, mask, lat, lon, timestamps_ms)

        # 2. Continuous 61-Day Test Window Evaluation
        print("\n--- Running Continuous 61-Day Test Window Evaluation ---")
        # Use num_ddim_timesteps=5 for fast execution across all 61 consecutive days
        preds_cont, tgts_cont, clim_cont, _, _, _, timestamps_cont = run_full_inference(
            enc, unet, diff, device, CONTINUOUS_TEST_DATES, num_ddim_timesteps=5, eta=0.0
        )

        cont_rmse = float(compute_rmse(preds_cont, tgts_cont, mask=mask))
        cont_mae = float(compute_mae(preds_cont, tgts_cont, mask=mask))
        cont_bias = float(compute_bias(preds_cont, tgts_cont, mask=mask))
        cont_corr = float(compute_correlation(preds_cont, tgts_cont, mask=mask))
        cont_clim_rmse = float(compute_rmse(clim_cont, tgts_cont, mask=mask))
        cont_skill = float(compute_murphy_skill_score(preds_cont, tgts_cont, clim_cont, mask=mask))

        print(f"\n[SUMMARY for {m_info['label']}]")
        print(f"  Multi-Seasonal Overall RMSE: {overall_rmse:.4f} °C (Clim: {clim_rmse:.4f} °C, Skill: {murphy_skill:.4f})")
        print(f"  Continuous 61-Day Test RMSE: {cont_rmse:.4f} °C (Clim: {cont_clim_rmse:.4f} °C, Skill: {cont_skill:.4f})")
        print(f"  Mean Water-Column Bias:      {overall_bias:+.4f} °C")

        full_report[m_key] = {
            "label": m_info["label"],
            "checkpoint_path": m_info["checkpoint_path"],
            "multi_seasonal_summary": {
                "overall_rmse": overall_rmse,
                "overall_mae": overall_mae,
                "overall_bias": overall_bias,
                "overall_correlation": overall_corr,
                "climatology_rmse": clim_rmse,
                "murphy_skill_score": murphy_skill,
            },
            "continuous_test_summary": {
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
            "depthwise_metrics": depthwise_metrics,
            "priority_zones": zone_metrics,
        }

    # 3. Side-by-Side Uncertainty Calibration Evaluation
    print("\n========================================================")
    print("CALIBRATION EVALUATION: Warm-Started 40k vs Clean Scratch 40k")
    print("========================================================")
    cal_dates = [60, 150, 240, 318, 358]
    
    print("\n--- Evaluating Calibration for Warm-Started 40k ---")
    ws_enc, ws_unet, _, ws_diff, _ = load_model(
        models_config["warm_started_40k"]["checkpoint_path"],
        device=device,
        normalize_inputs=models_config["warm_started_40k"]["normalize_inputs"],
        norm_stats_path=models_config["warm_started_40k"]["norm_stats_path"],
    )
    ws_cal = evaluate_calibration_for_model(ws_enc, ws_unet, ws_diff, device, cal_dates, num_ens=5)

    print("\n--- Loading Calibration for Clean Scratch 40k ---")
    with open("reports/calibration_scaling_scratch_40k_results.json", "r") as f:
        scratch_cal = json.load(f)

    calibration_comparison = {
        "warm_started_40k": ws_cal,
        "clean_scratch_40k": scratch_cal,
    }

    # Save comprehensive reports
    rep_dir = REPO_ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    with open(rep_dir / "granular_depth_zone_evaluation_all_models.json", "w") as f:
        json.dump(full_report, f, indent=2)

    with open(rep_dir / "calibration_comparison_40k.json", "w") as f:
        json.dump(calibration_comparison, f, indent=2)

    print("\n[COMPLETE] All Category A analyses computed and saved successfully.")


if __name__ == "__main__":
    main()
