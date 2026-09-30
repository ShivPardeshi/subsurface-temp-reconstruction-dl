"""Comprehensive Extended Benchmark Suite for Phase 6 Thermocline Breakthrough 25k Run.

Executes all core and extended tests:
1. Multi-Seasonal 10-Date Benchmark (Winter, Spring, Summer, Autumn)
2. 15 Canonical Depths Vertical Profile Breakdown (RMSE, MAE, Bias, Pearson R, R² vs Spatial Mean, Murphy Skill)
3. 7 Priority Oceanographic Zones (BoB Barrier Layer, Thermocline Core, AS PGW, Confluence, Extreme Cyclones, Monsoon Transitions, Equatorial Edge)
4. Auxiliary Physical Target Accuracy (MLD, BLT, Salinity Max Depth)
5. Physical Oceanographic Soundness (Static Stability Violation Rate, TCHP Error, D20 & D26 Isotherm Depth Errors)
6. Structural & Spectral Quality (Spatial SSIM at surface, thermocline, deep ocean)
7. DDIM Sampling Trajectory Efficiency Comparison (10-step vs 25-step)
"""

import sys
import os
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

import torch
import torch.nn.functional as F
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
from src.utils.grid import CANONICAL_DEPTHS, get_target_grid
from src.evaluation.metrics.basic_metrics import (
    compute_rmse,
    compute_mae,
    compute_bias,
    compute_correlation,
    compute_r2,
    _apply_mask,
)
from src.evaluation.metrics.skill_score import compute_murphy_skill_score
from src.evaluation.metrics.ssim_metric import compute_ssim_2d
from src.products.tchp import compute_tchp_field, find_d26_isotherm_depth

CANONICAL_MULTI_SEASONAL_DATES = [15, 60, 105, 150, 195, 240, 285, 318, 331, 358]


def load_phase6_model(checkpoint_path: str, device: torch.device):
    print(f"Loading Phase 6 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    norm_stats = "data/processed/channel_normalization_stats_ocean_only.json"
    context_encoder = ContextEncoder(
        in_channels=25,
        hidden_dims=[32, 64, 64],
        normalize_inputs=True,
        norm_stats_path=norm_stats,
    ).to(device)

    unet = UNetDenoiser(
        in_channels=74,
        stage_channels=[32, 64, 128, 256],
        cond_in_dim=14,
    ).to(device)

    aux_heads = AuxiliaryHeads(in_features=64).to(device)
    diffusion = GaussianDiffusion(timesteps=1000, schedule_type="cosine").to(device)

    context_encoder.load_state_dict(ckpt["models"]["context_encoder"])
    unet.load_state_dict(ckpt["models"]["unet"])
    aux_heads.load_state_dict(ckpt["models"]["aux_heads"])

    context_encoder.eval()
    unet.eval()
    aux_heads.eval()
    return context_encoder, unet, aux_heads, diffusion


def compute_isotherm_depth(temps: np.ndarray, depths: List[float], iso_temp: float = 20.0) -> np.ndarray:
    """Compute isotherm depth map (H, W) from 3D temperature profile (D, H, W)."""
    D, H, W = temps.shape
    iso_map = np.zeros((H, W), dtype=np.float32)
    depths_arr = np.array(depths, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            col = temps[:, i, j]
            if col[0] < iso_temp:
                iso_map[i, j] = 0.0
                continue
            idx = np.where(col < iso_temp)[0]
            if len(idx) == 0:
                iso_map[i, j] = depths_arr[-1]
            else:
                k = idx[0]
                if k == 0:
                    iso_map[i, j] = 0.0
                else:
                    z0, z1 = depths_arr[k - 1], depths_arr[k]
                    t0, t1 = col[k - 1], col[k]
                    if abs(t1 - t0) < 1e-4:
                        iso_map[i, j] = z0
                    else:
                        iso_map[i, j] = z0 + (iso_temp - t0) * (z1 - z0) / (t1 - t0)
    return iso_map


def main():
    device = torch.device("cpu")
    print("=" * 80)
    print("PHASE 6 THERMOCLINE BREAKTHROUGH COMPREHENSIVE EXTENDED EVALUATION SUITE")
    print("=" * 80)

    ckpt_path = "checkpoints/phase6_thermocline_breakthrough_25k/last_checkpoint.pt"
    context_encoder, unet, aux_heads, diffusion = load_phase6_model(ckpt_path, device)

    # 10 DDIM steps for fast multi-date sweep
    ddim_10 = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=10, eta=0.0, schedule_type="quadratic")
    cascade_10 = DepthCascadeSampler(
        context_encoder=context_encoder,
        unet_denoiser=unet,
        ddim_sampler=ddim_10,
        depths=CANONICAL_DEPTHS,
        rescale_output=True,
        aux_heads=aux_heads,
    )

    # Load datasets
    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    aux_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_auxiliary_targets.zarr", mode="r")
    clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    lat = in_zarr["lat"][:]
    lon = in_zarr["lon"][:]
    ocean_mask_np = (in_zarr["inputs"][0, 20] > 0.5)
    clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)
    static_channels = [20, 19, 21, 22, 23, 24]
    omega = 2.0 * math.pi / 365.25

    # Define 7 Priority Zone spatial masks
    # Zone 1: BoB Barrier Layer (0-30m)
    bob_mask = (in_zarr["inputs"][0, 22] > 0.5) & ocean_mask_np
    # Zone 2: Thermocline Core (75-150m, whole basin)
    # Zone 3: AS PGW (200-300m)
    as_mask = (in_zarr["inputs"][0, 21] > 0.5) & ocean_mask_np
    # Zone 4: Confluence (8-10N)
    lat_mesh, lon_mesh = np.meshgrid(lat, lon, indexing="ij")
    conf_mask = (lat_mesh >= 8.0) & (lat_mesh <= 10.0) & ocean_mask_np
    # Zone 5: Extreme Cyclone (Dates: Nov-Dec / May)
    # Zone 6: Monsoon Transitions (Apr / Oct dates: e.g. 105, 285)
    # Zone 7: Equatorial Domain Edge (2N to 5N)
    eq_edge_mask = (lat_mesh >= 2.0) & (lat_mesh <= 5.0) & ocean_mask_np

    all_pred_anom = []
    all_true_anom = []
    all_pred_temp = []
    all_true_temp = []
    all_clim_temp = []
    aux_eval_data = {"mld_true": [], "mld_pred": [], "blt_true": [], "blt_pred": [], "sal_true": [], "sal_pred": []}

    print(f"\n[Test 1] Running Multi-Seasonal 10-Date Benchmark: {CANONICAL_MULTI_SEASONAL_DATES}...")
    t_start = time.time()

    with torch.no_grad():
        for idx, t_day in enumerate(CANONICAL_MULTI_SEASONAL_DATES):
            t0 = time.time()
            seq = np.nan_to_num(in_zarr["inputs"][t_day - 6 : t_day + 1], nan=0.0)
            static_feats_np = np.nan_to_num(in_zarr["inputs"][t_day, static_channels], nan=0.0)
            true_anom_np = np.nan_to_num(tgt_zarr["anomaly"][t_day].copy(), nan=0.0)
            true_anom_np[0] = true_anom_np[1]  # surface coordinate extrapolation

            doy = int(scalar_df.loc[t_day, "day_of_year"])
            sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
            cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
            oni = float(scalar_df.loc[t_day, "oni_index"])
            iod = float(scalar_df.loc[t_day, "iod_dmi_index"])

            clim_t_np = (
                clim_coeffs[0]
                + clim_coeffs[1] * math.cos(omega * doy)
                + clim_coeffs[2] * math.sin(omega * doy)
                + clim_coeffs[3] * math.cos(2 * omega * doy)
                + clim_coeffs[4] * math.sin(2 * omega * doy)
            )
            true_temp_np = true_anom_np + clim_t_np

            x_seq = torch.from_numpy(seq).unsqueeze(0).float().to(device)
            static_feats = torch.from_numpy(static_feats_np).unsqueeze(0).float().to(device)
            scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], dtype=torch.float32, device=device)

            # Sample 3D profile
            out = cascade_10.sample_full_profile(
                x_seq=x_seq,
                static_features=static_feats,
                scalar_conditions=scalar_cond,
                use_cascade=True,
            )
            pred_anom_np = out["anomalies"].squeeze(0).cpu().numpy()
            pred_temp_np = pred_anom_np + clim_t_np

            all_pred_anom.append(pred_anom_np)
            all_true_anom.append(true_anom_np)
            all_pred_temp.append(pred_temp_np)
            all_true_temp.append(true_temp_np)
            all_clim_temp.append(clim_t_np)

            # Evaluate auxiliary heads
            u_cond = context_encoder(x_seq)
            aux_out = aux_heads(u_cond)
            pred_mld = float(aux_out["mld"].item())
            pred_blt = float(aux_out["blt"].item())
            pred_sal = float(aux_out["sal_max_depth"].item())

            true_aux = aux_zarr["auxiliary_targets"][t_day]  # (4, H, W)
            true_mld = float(np.nanmean(true_aux[0][ocean_mask_np]))
            true_blt = float(np.nanmean(true_aux[1][bob_mask])) if np.sum(bob_mask) > 0 else float(np.nanmean(true_aux[1][ocean_mask_np]))
            true_sal = float(np.nanmean(true_aux[2][as_mask])) if np.sum(as_mask) > 0 else float(np.nanmean(true_aux[2][ocean_mask_np]))

            aux_eval_data["mld_true"].append(true_mld)
            aux_eval_data["mld_pred"].append(pred_mld)
            aux_eval_data["blt_true"].append(true_blt)
            aux_eval_data["blt_pred"].append(pred_blt)
            aux_eval_data["sal_true"].append(true_sal)
            aux_eval_data["sal_pred"].append(pred_sal)

            dt = time.time() - t0
            print(f"  Processed Date {t_day:3d} (DOY {doy:3d}) in {dt:.2f}s | Pred MLD: {pred_mld:.1f}m (True: {true_mld:.1f}m)")

    print(f"Multi-Seasonal sweep completed in {time.time() - t_start:.1f}s.")

    # Convert to stacked numpy arrays: (N, 15, H, W)
    P_anom = np.stack(all_pred_anom, axis=0)
    T_anom = np.stack(all_true_anom, axis=0)
    P_temp = np.stack(all_pred_temp, axis=0)
    T_temp = np.stack(all_true_temp, axis=0)
    C_temp = np.stack(all_clim_temp, axis=0)

    # 1. Overall Multi-Seasonal Metrics
    overall_rmse = float(compute_rmse(P_anom, T_anom, mask=ocean_mask_np))
    overall_mae = float(compute_mae(P_anom, T_anom, mask=ocean_mask_np))
    overall_bias = float(compute_bias(P_anom, T_anom, mask=ocean_mask_np))
    overall_corr = float(compute_correlation(P_temp, T_temp, mask=ocean_mask_np))
    clim_rmse = float(compute_rmse(np.zeros_like(T_anom), T_anom, mask=ocean_mask_np))
    murphy_skill = float(compute_murphy_skill_score(P_temp, T_temp, C_temp, mask=ocean_mask_np))

    # 2. Depthwise Detailed Metrics
    depthwise_metrics = {}
    for d_idx, depth in enumerate(CANONICAL_DEPTHS):
        p_d = P_anom[:, d_idx]
        t_d = T_anom[:, d_idx]
        p_temp_d = P_temp[:, d_idx]
        t_temp_d = T_temp[:, d_idx]
        c_temp_d = C_temp[:, d_idx]

        rmse_d = float(compute_rmse(p_d, t_d, mask=ocean_mask_np))
        mae_d = float(compute_mae(p_d, t_d, mask=ocean_mask_np))
        bias_d = float(compute_bias(p_d, t_d, mask=ocean_mask_np))
        corr_d = float(compute_correlation(p_temp_d, t_temp_d, mask=ocean_mask_np))
        r2_stat = float(compute_r2(p_temp_d, t_temp_d, mask=ocean_mask_np))
        clim_d = float(compute_rmse(np.zeros_like(t_d), t_d, mask=ocean_mask_np))
        skill_d = float(compute_murphy_skill_score(p_temp_d, t_temp_d, c_temp_d, mask=ocean_mask_np))

        depthwise_metrics[f"{int(depth)}m"] = {
            "depth_m": int(depth),
            "model_rmse": round(rmse_d, 4),
            "clim_rmse": round(clim_d, 4),
            "mae": round(mae_d, 4),
            "bias": round(bias_d, 4),
            "correlation_r": round(corr_d, 4),
            "r2_stat": round(r2_stat, 4),
            "murphy_skill_pct": round(skill_d * 100.0, 2),
        }

    # 3. 7 Canonical Priority Oceanographic Zones
    zone_results = {}
    # Zone 1: BoB Barrier Layer (0-30m)
    p_z1 = P_temp[:, :5]
    t_z1 = T_temp[:, :5]
    c_z1 = C_temp[:, :5]
    zone_results["zone1_bob_barrier_layer"] = {
        "name": "Bay of Bengal Barrier Layer (0-30m)",
        "model_rmse": round(float(compute_rmse(p_z1, t_z1, mask=bob_mask)), 4),
        "clim_rmse": round(float(compute_rmse(c_z1, t_z1, mask=bob_mask)), 4),
        "mae": round(float(compute_mae(p_z1, t_z1, mask=bob_mask)), 4),
        "bias": round(float(compute_bias(p_z1, t_z1, mask=bob_mask)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(p_z1, t_z1, c_z1, mask=bob_mask)) * 100.0, 2),
    }

    # Zone 2: Thermocline Core (75-150m, whole basin)
    p_z2 = P_temp[:, 6:10]
    t_z2 = T_temp[:, 6:10]
    c_z2 = C_temp[:, 6:10]
    zone_results["zone2_thermocline_core"] = {
        "name": "Thermocline Core (75-150m, Whole Basin)",
        "model_rmse": round(float(compute_rmse(p_z2, t_z2, mask=ocean_mask_np)), 4),
        "clim_rmse": round(float(compute_rmse(c_z2, t_z2, mask=ocean_mask_np)), 4),
        "mae": round(float(compute_mae(p_z2, t_z2, mask=ocean_mask_np)), 4),
        "bias": round(float(compute_bias(p_z2, t_z2, mask=ocean_mask_np)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(p_z2, t_z2, c_z2, mask=ocean_mask_np)) * 100.0, 2),
    }

    # Zone 3: Arabian Sea PGW (200-300m)
    p_z3 = P_temp[:, 10:12]
    t_z3 = T_temp[:, 10:12]
    c_z3 = C_temp[:, 10:12]
    zone_results["zone3_as_persian_gulf_water"] = {
        "name": "Arabian Sea Persian Gulf Water (200-300m)",
        "model_rmse": round(float(compute_rmse(p_z3, t_z3, mask=as_mask)), 4),
        "clim_rmse": round(float(compute_rmse(c_z3, t_z3, mask=as_mask)), 4),
        "mae": round(float(compute_mae(p_z3, t_z3, mask=as_mask)), 4),
        "bias": round(float(compute_bias(p_z3, t_z3, mask=as_mask)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(p_z3, t_z3, c_z3, mask=as_mask)) * 100.0, 2),
    }

    # Zone 4: Confluence Zone (8-10N, 0-200m)
    p_z4 = P_temp[:, :11]
    t_z4 = T_temp[:, :11]
    c_z4 = C_temp[:, :11]
    zone_results["zone4_confluence_zone"] = {
        "name": "Equatorial Confluence Zone (8-10°N, 0-200m)",
        "model_rmse": round(float(compute_rmse(p_z4, t_z4, mask=conf_mask)), 4),
        "clim_rmse": round(float(compute_rmse(c_z4, t_z4, mask=conf_mask)), 4),
        "mae": round(float(compute_mae(p_z4, t_z4, mask=conf_mask)), 4),
        "bias": round(float(compute_bias(p_z4, t_z4, mask=conf_mask)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(p_z4, t_z4, c_z4, mask=conf_mask)) * 100.0, 2),
    }

    # Zone 5: Extreme Cyclone Windows (Dates 150, 318, 331)
    cyclone_indices = [3, 7, 8]
    p_z5 = P_temp[cyclone_indices]
    t_z5 = T_temp[cyclone_indices]
    c_z5 = C_temp[cyclone_indices]
    zone_results["zone5_extreme_cyclone_windows"] = {
        "name": "Extreme Cyclone Windows (Pre/Post-Monsoon Storms)",
        "model_rmse": round(float(compute_rmse(p_z5, t_z5, mask=ocean_mask_np)), 4),
        "clim_rmse": round(float(compute_rmse(c_z5, t_z5, mask=ocean_mask_np)), 4),
        "mae": round(float(compute_mae(p_z5, t_z5, mask=ocean_mask_np)), 4),
        "bias": round(float(compute_bias(p_z5, t_z5, mask=ocean_mask_np)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(p_z5, t_z5, c_z5, mask=ocean_mask_np)) * 100.0, 2),
    }

    # Zone 6: Monsoon Transitions (Dates 105, 285)
    transition_indices = [2, 6]
    p_z6 = P_temp[transition_indices]
    t_z6 = T_temp[transition_indices]
    c_z6 = C_temp[transition_indices]
    zone_results["zone6_monsoon_transitions"] = {
        "name": "Monsoon Transition Windows (Spring & Autumn Reversals)",
        "model_rmse": round(float(compute_rmse(p_z6, t_z6, mask=ocean_mask_np)), 4),
        "clim_rmse": round(float(compute_rmse(c_z6, t_z6, mask=ocean_mask_np)), 4),
        "mae": round(float(compute_mae(p_z6, t_z6, mask=ocean_mask_np)), 4),
        "bias": round(float(compute_bias(p_z6, t_z6, mask=ocean_mask_np)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(p_z6, t_z6, c_z6, mask=ocean_mask_np)) * 100.0, 2),
    }

    # Zone 7: Equatorial Domain Edge (2N to 5N)
    zone_results["zone7_equatorial_domain_edge"] = {
        "name": "Equatorial Domain Edge (2°N-5°N Boundary Layer)",
        "model_rmse": round(float(compute_rmse(P_temp, T_temp, mask=eq_edge_mask)), 4),
        "clim_rmse": round(float(compute_rmse(C_temp, T_temp, mask=eq_edge_mask)), 4),
        "mae": round(float(compute_mae(P_temp, T_temp, mask=eq_edge_mask)), 4),
        "bias": round(float(compute_bias(P_temp, T_temp, mask=eq_edge_mask)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(P_temp, T_temp, C_temp, mask=eq_edge_mask)) * 100.0, 2),
    }

    # 4. Auxiliary Physical Predictions Accuracy
    mld_rmse = float(np.sqrt(np.mean((np.array(aux_eval_data["mld_pred"]) - np.array(aux_eval_data["mld_true"])) ** 2)))
    mld_mae = float(np.mean(np.abs(np.array(aux_eval_data["mld_pred"]) - np.array(aux_eval_data["mld_true"]))))
    blt_rmse = float(np.sqrt(np.mean((np.array(aux_eval_data["blt_pred"]) - np.array(aux_eval_data["blt_true"])) ** 2)))
    blt_mae = float(np.mean(np.abs(np.array(aux_eval_data["blt_pred"]) - np.array(aux_eval_data["blt_true"]))))
    sal_rmse = float(np.sqrt(np.mean((np.array(aux_eval_data["sal_pred"]) - np.array(aux_eval_data["sal_true"])) ** 2)))
    sal_mae = float(np.mean(np.abs(np.array(aux_eval_data["sal_pred"]) - np.array(aux_eval_data["sal_true"]))))

    aux_results = {
        "mixed_layer_depth_mld": {"rmse_meters": round(mld_rmse, 2), "mae_meters": round(mld_mae, 2)},
        "barrier_layer_thickness_blt": {"rmse_meters": round(blt_rmse, 2), "mae_meters": round(blt_mae, 2)},
        "salinity_max_depth": {"rmse_meters": round(sal_rmse, 2), "mae_meters": round(sal_mae, 2)},
    }

    # 5. Physical Oceanographic Soundness & Diagnostics
    print("\n[Test 2] Computing Physical Oceanographic Diagnostics (Stability, TCHP, D20, D26)...")
    inversion_counts = 0
    total_valid_pairs = 0
    tchp_errors = []
    d26_errors = []
    d20_errors = []

    for i in range(len(CANONICAL_MULTI_SEASONAL_DATES)):
        pred_3d = P_temp[i]  # (15, H, W)
        true_3d = T_temp[i]  # (15, H, W)

        # Check vertical inversions below 30m (depths 30m to 1000m)
        # Physical condition: T(d) <= T(d_prev) + 0.2°C tolerance
        for k in range(5, 14):
            diff_z = pred_3d[k + 1] - pred_3d[k]  # T(deep) - T(shallow)
            inversions = (diff_z > 0.2) & ocean_mask_np
            inversion_counts += int(np.sum(inversions))
            total_valid_pairs += int(np.sum(ocean_mask_np))

        # TCHP & D26
        pred_tchp, pred_d26 = compute_tchp_field(pred_3d, depths=CANONICAL_DEPTHS, mask=ocean_mask_np)
        true_tchp, true_d26 = compute_tchp_field(true_3d, depths=CANONICAL_DEPTHS, mask=ocean_mask_np)

        valid_tchp = ocean_mask_np & (true_tchp > 0.0)
        if np.sum(valid_tchp) > 0:
            tchp_errors.append(float(np.sqrt(np.mean((pred_tchp[valid_tchp] - true_tchp[valid_tchp]) ** 2))))
            d26_errors.append(float(np.sqrt(np.mean((pred_d26[valid_tchp] - true_d26[valid_tchp]) ** 2))))

        # D20
        pred_d20 = compute_isotherm_depth(pred_3d, CANONICAL_DEPTHS, iso_temp=20.0)
        true_d20 = compute_isotherm_depth(true_3d, CANONICAL_DEPTHS, iso_temp=20.0)
        valid_d20 = ocean_mask_np & (true_d20 > 0.0)
        if np.sum(valid_d20) > 0:
            d20_errors.append(float(np.sqrt(np.mean((pred_d20[valid_d20] - true_d20[valid_d20]) ** 2))))

    inversion_rate = (inversion_counts / max(1, total_valid_pairs)) * 100.0
    mean_tchp_rmse = float(np.mean(tchp_errors))
    mean_d26_rmse = float(np.mean(d26_errors))
    mean_d20_rmse = float(np.mean(d20_errors))

    physics_diagnostics = {
        "static_stability_inversion_rate_pct": round(inversion_rate, 3),
        "tchp_rmse_kj_cm2": round(mean_tchp_rmse, 3),
        "d26_isotherm_depth_rmse_meters": round(mean_d26_rmse, 2),
        "d20_isotherm_depth_rmse_meters": round(mean_d20_rmse, 2),
    }

    # 6. Structural & Spatial Similarity (SSIM)
    print("\n[Test 3] Computing Structural Similarity (SSIM)...")
    ssim_surface = float(np.mean([compute_ssim_2d(P_temp[i, 0], T_temp[i, 0], mask=ocean_mask_np) for i in range(len(CANONICAL_MULTI_SEASONAL_DATES))]))
    ssim_thermo = float(np.mean([compute_ssim_2d(P_temp[i, 8], T_temp[i, 8], mask=ocean_mask_np) for i in range(len(CANONICAL_MULTI_SEASONAL_DATES))]))
    ssim_deep = float(np.mean([compute_ssim_2d(P_temp[i, 12], T_temp[i, 12], mask=ocean_mask_np) for i in range(len(CANONICAL_MULTI_SEASONAL_DATES))]))

    structural_results = {
        "ssim_surface_0m": round(ssim_surface, 4),
        "ssim_thermocline_125m": round(ssim_thermo, 4),
        "ssim_deep_500m": round(ssim_deep, 4),
    }

    # 7. DDIM Sampling Trajectory Efficiency Comparison (10 vs 25 steps on Date 358)
    print("\n[Test 4] Comparing DDIM Sampling Trajectory Efficiency (10 vs 25 steps)...")
    t_test = 358
    seq = np.nan_to_num(in_zarr["inputs"][t_test - 6 : t_test + 1], nan=0.0)
    static_feats_np = np.nan_to_num(in_zarr["inputs"][t_test, static_channels], nan=0.0)
    true_anom_np = np.nan_to_num(tgt_zarr["anomaly"][t_test].copy(), nan=0.0)
    true_anom_np[0] = true_anom_np[1]

    doy = int(scalar_df.loc[t_test, "day_of_year"])
    sin_doy = float(scalar_df.loc[t_test, "sin_doy"])
    cos_doy = float(scalar_df.loc[t_test, "cos_doy"])
    oni = float(scalar_df.loc[t_test, "oni_index"])
    iod = float(scalar_df.loc[t_test, "iod_dmi_index"])

    x_seq = torch.from_numpy(seq).unsqueeze(0).float().to(device)
    static_feats = torch.from_numpy(static_feats_np).unsqueeze(0).float().to(device)
    scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], dtype=torch.float32, device=device)

    # 10 steps
    t0_10 = time.time()
    out_10 = cascade_10.sample_full_profile(x_seq=x_seq, static_features=static_feats, scalar_conditions=scalar_cond)
    dur_10 = time.time() - t0_10
    pred_10 = out_10["anomalies"].squeeze(0).cpu().numpy()
    rmse_10 = float(compute_rmse(pred_10, true_anom_np, mask=ocean_mask_np))

    # 25 steps
    ddim_25 = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=25, eta=0.0, schedule_type="quadratic")
    cascade_25 = DepthCascadeSampler(
        context_encoder=context_encoder, unet_denoiser=unet, ddim_sampler=ddim_25, depths=CANONICAL_DEPTHS, rescale_output=True, aux_heads=aux_heads
    )
    t0_25 = time.time()
    out_25 = cascade_25.sample_full_profile(x_seq=x_seq, static_features=static_feats, scalar_conditions=scalar_cond)
    dur_25 = time.time() - t0_25
    pred_25 = out_25["anomalies"].squeeze(0).cpu().numpy()
    rmse_25 = float(compute_rmse(pred_25, true_anom_np, mask=ocean_mask_np))

    trajectory_comparison = {
        "ddim_10_steps": {"rmse": round(rmse_10, 4), "runtime_sec": round(dur_10, 2)},
        "ddim_25_steps": {"rmse": round(rmse_25, 4), "runtime_sec": round(dur_25, 2)},
        "rmse_difference": round(rmse_10 - rmse_25, 4),
    }

    full_suite_results = {
        "model_name": "Phase 6 Thermocline Breakthrough Pure Diffusion (25k Steps)",
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "multi_seasonal_10_dates": {
            "dates_evaluated": CANONICAL_MULTI_SEASONAL_DATES,
            "overall_rmse": round(overall_rmse, 4),
            "overall_mae": round(overall_mae, 4),
            "overall_bias": round(overall_bias, 4),
            "overall_correlation": round(overall_corr, 4),
            "climatology_rmse": round(clim_rmse, 4),
            "murphy_skill_pct": round(murphy_skill * 100.0, 2),
            "depthwise_metrics": depthwise_metrics,
            "priority_zones": zone_results,
        },
        "auxiliary_physical_heads": aux_results,
        "physics_and_stability_diagnostics": physics_diagnostics,
        "structural_quality_ssim": structural_results,
        "ddim_trajectory_comparison": trajectory_comparison,
    }

    out_file = Path("reports/phase6_comprehensive_suite_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(full_suite_results, f, indent=2)
    print(f"\nComprehensive Suite Results successfully saved to: {out_file}")

    print("\n" + "=" * 80)
    print("EXTENDED BENCHMARK SUITE SUMMARY")
    print("=" * 80)
    print(f"Overall Multi-Seasonal RMSE:     {overall_rmse:.4f} °C (Climatology: {clim_rmse:.4f} °C)")
    print(f"Overall Absolute Correlation:    {overall_corr:.4f}")
    print(f"Static Stability Inversion Rate: {inversion_rate:.3f}% (Target: < 1.0%)")
    print(f"TCHP Estimation RMSE:            {mean_tchp_rmse:.3f} kJ/cm²")
    print(f"D20 Isotherm Depth RMSE:         {mean_d20_rmse:.2f} m")
    print(f"D26 Isotherm Depth RMSE:         {mean_d26_rmse:.2f} m")
    print(f"Auxiliary MLD Prediction RMSE:   {mld_rmse:.2f} m")
    print(f"Auxiliary BLT Prediction RMSE:   {blt_rmse:.2f} m")
    print(f"Spatial SSIM (Surface / Thermo): {ssim_surface:.4f} / {ssim_thermo:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
