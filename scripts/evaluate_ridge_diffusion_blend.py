"""P0.1: Comprehensive Ridge + Diffusion Ensemble Blending Evaluation.

Evaluates:
1. Ridge Standalone Baseline
2. Clean Scratch 40k Diffusion Standalone Baseline (with Region OFF, Cascade ON)
3. Grid of Uniform Linear Blends: alpha * Ridge + (1-alpha) * Diffusion
4. Depth-Dependent Optimal Blend:
   - Upper Ocean (0-150m): Diffusion-weighted (captures non-linear pycnocline/MLD/stratification)
   - Deep Ocean (200-1000m): Ridge-weighted (eliminates sampling variance in narrow-variance regime)
Evaluated across Multi-Seasonal 10 Dates AND Continuous 61-Day Nov-Dec Test Set.
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
from src.products.mld_direct import compute_profile_mld_direct


CANONICAL_MULTI_SEASONAL_DATES = [15, 60, 105, 150, 195, 240, 285, 318, 331, 358]
CONTINUOUS_TEST_DATES = list(range(298, 359))  # 61 days: 298 to 358


def load_diffusion_model(checkpoint_path: str, device: torch.device, norm_stats_path: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    context_encoder = ContextEncoder(
        in_channels=25,
        hidden_dims=[32, 64, 64],
        normalize_inputs=True,
        norm_stats_path=norm_stats_path,
    ).to(device)
    unet = UNetDenoiser(in_channels=72, stage_channels=[32, 64, 128, 256], cond_in_dim=8).to(device)
    aux_heads = AuxiliaryHeads(in_features=64).to(device)
    diffusion = GaussianDiffusion(timesteps=1000, schedule_type="cosine").to(device)

    context_encoder.load_state_dict(ckpt["models"]["context_encoder"], strict=False)
    unet.load_state_dict(ckpt["models"]["unet"])
    aux_heads.load_state_dict(ckpt["models"]["aux_heads"])

    context_encoder.eval()
    unet.eval()
    aux_heads.eval()
    return context_encoder, unet, aux_heads, diffusion


def train_ridge_weights() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    lat = in_zarr["lat"][:]
    lon = in_zarr["lon"][:]
    ocean_mask = (in_zarr["inputs"][0, 20] > 0.5)
    feature_channels = [0, 1, 2, 3, 4, 11, 14, 15, 19, 21, 22, 23, 24]
    train_days = list(range(6, 237))

    n_ocean = int(np.sum(ocean_mask))
    lat_mesh, lon_mesh = np.meshgrid(lat, lon, indexing="ij")
    lat_pts = lat_mesh[ocean_mask][None, :]
    lon_pts = lon_mesh[ocean_mask][None, :]

    X_list, Y_list = [], []
    for t_day in train_days:
        seq = in_zarr["inputs"][t_day - 6 : t_day + 1]
        anom = tgt_zarr["anomaly"][t_day]
        anom[0] = anom[1]

        sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
        cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
        oni = float(scalar_df.loc[t_day, "oni_index"])
        iod = float(scalar_df.loc[t_day, "iod_dmi_index"])

        curr_feat = seq[-1, feature_channels][:, ocean_mask]
        mean_feat = np.mean(seq[:, feature_channels], axis=0)[:, ocean_mask]
        diff_feat = (seq[-1, feature_channels] - seq[0, feature_channels])[:, ocean_mask]
        scalars = np.array([[sin_doy], [cos_doy], [oni], [iod]]) * np.ones((4, n_ocean))

        x_day = np.vstack([curr_feat, mean_feat, diff_feat, lat_pts, lon_pts, scalars]).T
        y_day = anom[:, ocean_mask].T
        X_list.append(x_day)
        Y_list.append(y_day)

    X_train = np.nan_to_num(np.vstack(X_list), nan=0.0)
    Y_train = np.nan_to_num(np.vstack(Y_list), nan=0.0)

    mean_X = np.mean(X_train, axis=0, keepdims=True)
    std_X = np.std(X_train, axis=0, keepdims=True) + 1e-6
    X_train_norm = (X_train - mean_X) / std_X
    X_train_b = np.hstack([X_train_norm, np.ones((X_train_norm.shape[0], 1))])

    alpha = 100.0
    XtX = X_train_b.T @ X_train_b + alpha * np.eye(X_train_b.shape[1])
    XtY = X_train_b.T @ Y_train
    W = np.linalg.solve(XtX, XtY)
    return W, mean_X, std_X, feature_channels


def get_all_predictions(
    context_encoder: nn.Module,
    unet: nn.Module,
    diffusion: GaussianDiffusion,
    device: torch.device,
    W_ridge: np.ndarray,
    mean_X: np.ndarray,
    std_X: np.ndarray,
    feature_channels: List[int],
    target_dates: List[int],
    use_region: bool = False,
    num_ddim_steps: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    lat = in_zarr["lat"][:]
    lon = in_zarr["lon"][:]
    ocean_mask_np = (in_zarr["inputs"][0, 20] > 0.5)
    n_ocean = int(np.sum(ocean_mask_np))
    clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)
    static_channels = [20, 19, 21, 22, 23, 24]
    omega = 2.0 * math.pi / 365.25

    lat_mesh, lon_mesh = np.meshgrid(lat, lon, indexing="ij")
    lat_pts = lat_mesh[ocean_mask_np][None, :]
    lon_pts = lon_mesh[ocean_mask_np][None, :]

    ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=num_ddim_steps, eta=0.0)
    cascade = DepthCascadeSampler(
        context_encoder=context_encoder,
        unet_denoiser=unet,
        ddim_sampler=ddim,
        depths=CANONICAL_DEPTHS,
    )

    all_diff_preds, all_ridge_preds, all_targets, all_clim = [], [], [], []

    with torch.no_grad():
        for t_day in target_dates:
            doy = int(scalar_df.loc[t_day, "day_of_year"])
            sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
            cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
            oni = float(scalar_df.loc[t_day, "oni_index"])
            iod = float(scalar_df.loc[t_day, "iod_dmi_index"])

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

            # 1. Diffusion prediction (Region OFF per clean finding, Cascade ON)
            seq_slice = in_zarr["inputs"][t_day - 6 : t_day + 1].copy()
            seq_slice = np.nan_to_num(seq_slice, nan=0.0)
            if not use_region:
                seq_slice[:, 21:25] = 0.0

            x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)
            static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
            if not use_region:
                static_feats[:, 2:6] = 0.0

            scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], device=device, dtype=torch.float32)

            out_diff = cascade.sample_full_profile(
                x_seq=x_seq,
                static_features=static_feats,
                scalar_conditions=scalar_cond,
                use_cascade=True,
            )
            diff_anom = out_diff["anomalies"][0].cpu().numpy()
            diff_temp = diff_anom + clim_t

            # 2. Ridge prediction
            curr_feat = seq_slice[-1, feature_channels][:, ocean_mask_np]
            mean_feat = np.mean(seq_slice[:, feature_channels], axis=0)[:, ocean_mask_np]
            diff_feat = (seq_slice[-1, feature_channels] - seq_slice[0, feature_channels])[:, ocean_mask_np]
            scalars = np.array([[sin_doy], [cos_doy], [oni], [iod]]) * np.ones((4, n_ocean))

            x_day = np.vstack([curr_feat, mean_feat, diff_feat, lat_pts, lon_pts, scalars]).T
            x_day_norm = (x_day - mean_X) / std_X
            x_day_b = np.hstack([x_day_norm, np.ones((x_day_norm.shape[0], 1))])

            ridge_anom_ocean = x_day_b @ W_ridge
            ridge_anom = np.zeros((15, len(lat), len(lon)), dtype=np.float32)
            for d in range(15):
                ridge_anom[d, ocean_mask_np] = ridge_anom_ocean[:, d]
            ridge_temp = ridge_anom + clim_t

            all_diff_preds.append(diff_temp)
            all_ridge_preds.append(ridge_temp)
            all_targets.append(true_temp)
            all_clim.append(clim_t)

    return (
        np.stack(all_diff_preds, axis=0),
        np.stack(all_ridge_preds, axis=0),
        np.stack(all_targets, axis=0),
        np.stack(all_clim, axis=0),
        ocean_mask_np,
    )


def evaluate_blend(
    preds_blend: np.ndarray,
    targets: np.ndarray,
    clim: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    shallow_idx = [0, 1, 2, 3, 4]       # 0-30m
    thermo_idx = [6, 7, 8, 9]            # 75-150m
    deep_idx = [10, 11, 12, 13, 14]      # 200-1000m

    rmse = float(compute_rmse(preds_blend, targets, mask=mask))
    mae = float(compute_mae(preds_blend, targets, mask=mask))
    bias = float(compute_bias(preds_blend, targets, mask=mask))
    corr = float(compute_correlation(preds_blend, targets, mask=mask))
    clim_rmse = float(compute_rmse(clim, targets, mask=mask))
    skill = float(compute_murphy_skill_score(preds_blend, targets, clim, mask=mask))

    s_rmse = float(compute_rmse(preds_blend[:, shallow_idx], targets[:, shallow_idx], mask=mask))
    t_rmse = float(compute_rmse(preds_blend[:, thermo_idx], targets[:, thermo_idx], mask=mask))
    d_rmse = float(compute_rmse(preds_blend[:, deep_idx], targets[:, deep_idx], mask=mask))

    return {
        "overall_rmse": rmse,
        "overall_mae": mae,
        "overall_bias": bias,
        "overall_correlation": corr,
        "climatology_rmse": clim_rmse,
        "murphy_skill_score": skill,
        "shallow_rmse": s_rmse,
        "thermo_rmse": t_rmse,
        "deep_rmse": d_rmse,
    }


def main():
    device = torch.device("cpu")
    print("=" * 80)
    print("P0.1: RIDGE + CLEAN SCRATCH 40K DIFFUSION ENSEMBLE BLEND EVALUATION")
    print("=" * 80)

    # 1. Fit Ridge baseline
    W_ridge, mean_X, std_X, feat_ch = train_ridge_weights()

    # 2. Load Clean Scratch 40k Diffusion
    ckpt_path = "checkpoints/phase3_retrain_scratch_40k_full/best_checkpoint.pt"
    norm_stats = "data/processed/channel_normalization_stats_ocean_only.json"
    enc, unet, aux, diff = load_diffusion_model(ckpt_path, device, norm_stats)

    # 3. Generate predictions across Multi-Seasonal 10 Dates
    print("\n--- Evaluating Multi-Seasonal 10 Dates ---")
    diff_ms, ridge_ms, tgts_ms, clim_ms, mask = get_all_predictions(
        enc, unet, diff, device, W_ridge, mean_X, std_X, feat_ch, CANONICAL_MULTI_SEASONAL_DATES, use_region=False
    )

    # Test alpha grid: alpha * Ridge + (1-alpha) * Diffusion
    alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    grid_results_ms = {}

    print(f"{'Alpha (Ridge)':<15} | {'Overall RMSE':<14} | {'Skill Score':<14} | {'Shallow':<10} | {'Thermo':<10} | {'Deep':<10} | {'Bias':<10}")
    print("-" * 95)
    for a in alpha_grid:
        blend = a * ridge_ms + (1.0 - a) * diff_ms
        res = evaluate_blend(blend, tgts_ms, clim_ms, mask)
        grid_results_ms[f"alpha_{a:.1f}"] = res
        print(f"{a:<15.1f} | {res['overall_rmse']:<14.4f} | {res['murphy_skill_score']:<14.4f} | {res['shallow_rmse']:<10.4f} | {res['thermo_rmse']:<10.4f} | {res['deep_rmse']:<10.4f} | {res['overall_bias']:<+10.4f}")

    # 4. Depth-Dependent Optimal Blend:
    # Depths: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]m
    # In shallow & thermocline: alpha_ridge = 0.5 (blend both)
    # In deep (200-1000m): alpha_ridge = 0.85 (smooth linear prior, eliminating stochastic noise)
    alpha_depthwise = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95, 0.95, 0.95])
    a_broad = alpha_depthwise[None, :, None, None]
    blend_depth_opt = a_broad * ridge_ms + (1.0 - a_broad) * diff_ms
    res_depth_opt_ms = evaluate_blend(blend_depth_opt, tgts_ms, clim_ms, mask)

    print("\n--- Depth-Dependent Optimal Blend (Multi-Seasonal) ---")
    print(f"Overall RMSE: {res_depth_opt_ms['overall_rmse']:.4f} °C (Clim: {res_depth_opt_ms['climatology_rmse']:.4f} °C | Skill: {res_depth_opt_ms['murphy_skill_score']:.4f})")
    print(f"Shallow: {res_depth_opt_ms['shallow_rmse']:.4f} °C | Thermo: {res_depth_opt_ms['thermo_rmse']:.4f} °C | Deep: {res_depth_opt_ms['deep_rmse']:.4f} °C | Bias: {res_depth_opt_ms['overall_bias']:+.4f} °C")

    # 5. Continuous 61-Day Test Window Evaluation
    print("\n--- Evaluating Continuous 61-Day Test Window (Nov 1 - Dec 31, 2025) ---")
    diff_cont, ridge_cont, tgts_cont, clim_cont, _ = get_all_predictions(
        enc, unet, diff, device, W_ridge, mean_X, std_X, feat_ch, CONTINUOUS_TEST_DATES, use_region=False, num_ddim_steps=5
    )

    grid_results_cont = {}
    for a in [0.0, 0.5, 0.7, 1.0]:
        blend = a * ridge_cont + (1.0 - a) * diff_cont
        res = evaluate_blend(blend, tgts_cont, clim_cont, mask)
        grid_results_cont[f"alpha_{a:.1f}"] = res
        print(f"Alpha={a:.1f} -> Continuous Test RMSE: {res['overall_rmse']:.4f} °C (Clim: {res['climatology_rmse']:.4f} °C | Skill: {res['murphy_skill_score']:.4f})")

    blend_depth_opt_cont = a_broad * ridge_cont + (1.0 - a_broad) * diff_cont
    res_depth_opt_cont = evaluate_blend(blend_depth_opt_cont, tgts_cont, clim_cont, mask)
    print(f"Depth-Optimal Blend -> Continuous Test RMSE: {res_depth_opt_cont['overall_rmse']:.4f} °C (Clim: {res_depth_opt_cont['climatology_rmse']:.4f} °C | Skill: {res_depth_opt_cont['murphy_skill_score']:.4f})")

    # Save summary
    out_dict = {
        "multi_seasonal_10_dates": {
            "alpha_grid": grid_results_ms,
            "depth_dependent_optimal_blend": res_depth_opt_ms,
        },
        "continuous_61_days": {
            "alpha_grid": grid_results_cont,
            "depth_dependent_optimal_blend": res_depth_opt_cont,
        },
        "locked_weights": {
            "alpha_depthwise": alpha_depthwise.tolist(),
            "canonical_depths": CANONICAL_DEPTHS,
        },
    }

    out_file = REPO_ROOT / "reports" / "p0_1_ridge_diffusion_blend_results.json"
    with open(out_file, "w") as f:
        json.dump(out_dict, f, indent=2)

    print(f"\n[COMPLETE] P0.1 Blend results saved to: {out_file}")


if __name__ == "__main__":
    main()
