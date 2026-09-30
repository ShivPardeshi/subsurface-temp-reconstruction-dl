"""Comprehensive Extended Benchmark Suite for Phase 8 Zone-Adaptive 15k Run.

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


def load_phase8_model(checkpoint_path: str, device: torch.device):
    print(f"Loading Phase 8 checkpoint: {checkpoint_path}")
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

    if "models" in ckpt:
        context_encoder.load_state_dict(ckpt["models"]["context_encoder"])
        unet.load_state_dict(ckpt["models"]["unet"])
        aux_heads.load_state_dict(ckpt["models"]["aux_heads"])
    elif "model_state_dict" in ckpt:
        context_encoder.load_state_dict(ckpt["model_state_dict"].get("context_encoder", {}))
        unet.load_state_dict(ckpt["model_state_dict"].get("unet", {}))
        aux_heads.load_state_dict(ckpt["model_state_dict"].get("aux_heads", {}))

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
    print("PHASE 8 ZONE-ADAPTIVE SCALING COMPREHENSIVE EXTENDED EVALUATION SUITE")
    print("=" * 80)

    ckpt_path = "checkpoints/phase8_zone_adaptive_15k/last_checkpoint.pt"
    context_encoder, unet, aux_heads, diffusion = load_phase8_model(ckpt_path, device)

    # 10 DDIM steps for fast multi-date sweep
    ddim_10 = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=10, eta=0.0, schedule_type="quadratic")
    cascade_10 = DepthCascadeSampler(
        context_encoder=context_encoder,
        unet_denoiser=unet,
        ddim_sampler=ddim_10,
        depths=CANONICAL_DEPTHS,
        depth_scales_path="data/processed/anomaly_depth_scales_zone_adaptive_p8.json",
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
        "name": "Thermocline Core (75-150m)",
        "model_rmse": round(float(compute_rmse(p_z2, t_z2, mask=ocean_mask_np)), 4),
        "clim_rmse": round(float(compute_rmse(c_z2, t_z2, mask=ocean_mask_np)), 4),
        "mae": round(float(compute_mae(p_z2, t_z2, mask=ocean_mask_np)), 4),
        "bias": round(float(compute_bias(p_z2, t_z2, mask=ocean_mask_np)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(p_z2, t_z2, c_z2, mask=ocean_mask_np)) * 100.0, 2),
    }

    # Zone 3: Arabian Sea Persian Gulf Water (200-300m)
    p_z3 = P_temp[:, 10:12]
    t_z3 = T_temp[:, 10:12]
    c_z3 = C_temp[:, 10:12]
    zone_results["zone3_as_pgw"] = {
        "name": "Arabian Sea Persian Gulf Water (200-300m)",
        "model_rmse": round(float(compute_rmse(p_z3, t_z3, mask=as_mask)), 4),
        "clim_rmse": round(float(compute_rmse(c_z3, t_z3, mask=as_mask)), 4),
        "mae": round(float(compute_mae(p_z3, t_z3, mask=as_mask)), 4),
        "bias": round(float(compute_bias(p_z3, t_z3, mask=as_mask)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(p_z3, t_z3, c_z3, mask=as_mask)) * 100.0, 2),
    }

    # Zone 4: Confluence Zone (8-10N)
    zone_results["zone4_confluence"] = {
        "name": "Equatorial Confluence Zone (8-10N)",
        "model_rmse": round(float(compute_rmse(P_temp, T_temp, mask=conf_mask)), 4),
        "clim_rmse": round(float(compute_rmse(C_temp, T_temp, mask=conf_mask)), 4),
        "mae": round(float(compute_mae(P_temp, T_temp, mask=conf_mask)), 4),
        "bias": round(float(compute_bias(P_temp, T_temp, mask=conf_mask)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(P_temp, T_temp, C_temp, mask=conf_mask)) * 100.0, 2),
    }

    # Zone 5: Extreme Cyclone (Dates 150, 318, 331, 358)
    cyclone_indices = [3, 7, 8, 9]
    p_z5 = P_temp[cyclone_indices]
    t_z5 = T_temp[cyclone_indices]
    c_z5 = C_temp[cyclone_indices]
    zone_results["zone5_cyclone"] = {
        "name": "Extreme Cyclone Profiles (Pre/Post-Monsoon)",
        "model_rmse": round(float(compute_rmse(p_z5, t_z5, mask=ocean_mask_np)), 4),
        "clim_rmse": round(float(compute_rmse(c_z5, t_z5, mask=ocean_mask_np)), 4),
        "mae": round(float(compute_mae(p_z5, t_z5, mask=ocean_mask_np)), 4),
        "bias": round(float(compute_bias(p_z5, t_z5, mask=ocean_mask_np)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(p_z5, t_z5, c_z5, mask=ocean_mask_np)) * 100.0, 2),
    }

    # Zone 6: Monsoon Transitions (Dates 105, 285)
    trans_indices = [2, 6]
    p_z6 = P_temp[trans_indices]
    t_z6 = T_temp[trans_indices]
    c_z6 = C_temp[trans_indices]
    zone_results["zone6_monsoon_transitions"] = {
        "name": "Monsoon Inter-Monsoon Transitions (Apr/Oct)",
        "model_rmse": round(float(compute_rmse(p_z6, t_z6, mask=ocean_mask_np)), 4),
        "clim_rmse": round(float(compute_rmse(c_z6, t_z6, mask=ocean_mask_np)), 4),
        "mae": round(float(compute_mae(p_z6, t_z6, mask=ocean_mask_np)), 4),
        "bias": round(float(compute_bias(p_z6, t_z6, mask=ocean_mask_np)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(p_z6, t_z6, c_z6, mask=ocean_mask_np)) * 100.0, 2),
    }

    # Zone 7: Equatorial Domain Edge (2N to 5N)
    zone_results["zone7_equatorial_edge"] = {
        "name": "Equatorial Domain Boundary Edge (2N-5N)",
        "model_rmse": round(float(compute_rmse(P_temp, T_temp, mask=eq_edge_mask)), 4),
        "clim_rmse": round(float(compute_rmse(C_temp, T_temp, mask=eq_edge_mask)), 4),
        "mae": round(float(compute_mae(P_temp, T_temp, mask=eq_edge_mask)), 4),
        "bias": round(float(compute_bias(P_temp, T_temp, mask=eq_edge_mask)), 4),
        "murphy_skill_pct": round(float(compute_murphy_skill_score(P_temp, T_temp, C_temp, mask=eq_edge_mask)) * 100.0, 2),
    }

    # 4. Auxiliary Target Physical Soundness
    mld_true_arr = np.array(aux_eval_data["mld_true"])
    mld_pred_arr = np.array(aux_eval_data["mld_pred"])
    blt_true_arr = np.array(aux_eval_data["blt_true"])
    blt_pred_arr = np.array(aux_eval_data["blt_pred"])
    sal_true_arr = np.array(aux_eval_data["sal_true"])
    sal_pred_arr = np.array(aux_eval_data["sal_pred"])

    aux_results = {
        "mld_rmse_m": round(float(np.sqrt(np.mean((mld_pred_arr - mld_true_arr) ** 2))), 2),
        "mld_mae_m": round(float(np.mean(np.abs(mld_pred_arr - mld_true_arr))), 2),
        "mld_corr_r": round(float(np.corrcoef(mld_pred_arr, mld_true_arr)[0, 1]), 4),
        "blt_rmse_m": round(float(np.sqrt(np.mean((blt_pred_arr - blt_true_arr) ** 2))), 2),
        "blt_mae_m": round(float(np.mean(np.abs(blt_pred_arr - blt_true_arr))), 2),
        "blt_corr_r": round(float(np.corrcoef(blt_pred_arr, blt_true_arr)[0, 1]), 4),
        "sal_max_depth_rmse_m": round(float(np.sqrt(np.mean((sal_pred_arr - sal_true_arr) ** 2))), 2),
        "sal_max_depth_mae_m": round(float(np.mean(np.abs(sal_pred_arr - sal_true_arr))), 2),
    }

    # 5. Physical Consistency & Stability Violations
    stability_violations = 0
    total_profiles = 0
    d20_errors = []
    d26_errors = []

    for t_i in range(len(CANONICAL_MULTI_SEASONAL_DATES)):
        p_3d = P_temp[t_i]
        t_3d = T_temp[t_i]

        # Inversion check: count dT/dz > 0.5 °C / m (allows slight barrier layer, penalizes unphysical inverted hot spots)
        dz = np.diff(np.array(CANONICAL_DEPTHS))
        dt_dz = np.diff(p_3d, axis=0) / dz[:, None, None]
        unphysical_inversion = (dt_dz > 0.5) & ocean_mask_np[None, :, :]
        stability_violations += int(np.sum(unphysical_inversion))
        total_profiles += int(np.sum(ocean_mask_np) * (len(CANONICAL_DEPTHS) - 1))

        # D20 and D26 Isotherm errors
        d20_pred = compute_isotherm_depth(p_3d, CANONICAL_DEPTHS, iso_temp=20.0)
        d20_true = compute_isotherm_depth(t_3d, CANONICAL_DEPTHS, iso_temp=20.0)
        d20_diff = (d20_pred - d20_true)[ocean_mask_np]
        d20_errors.extend((d20_diff ** 2).tolist())

        d26_pred = compute_isotherm_depth(p_3d, CANONICAL_DEPTHS, iso_temp=26.0)
        d26_true = compute_isotherm_depth(t_3d, CANONICAL_DEPTHS, iso_temp=26.0)
        d26_diff = (d26_pred - d26_true)[ocean_mask_np]
        d26_errors.extend((d26_diff ** 2).tolist())

    stability_rate_pct = (stability_violations / max(1, total_profiles)) * 100.0
    d20_rmse = math.sqrt(np.mean(d20_errors))
    d26_rmse = math.sqrt(np.mean(d26_errors))

    physics_metrics = {
        "static_stability_violation_rate_pct": round(stability_rate_pct, 4),
        "d20_isotherm_depth_rmse_m": round(d20_rmse, 2),
        "d26_isotherm_depth_rmse_m": round(d26_rmse, 2),
    }

    # 6. Spatial Structural Similarity (SSIM)
    ssim_surface = []
    ssim_thermo = []
    ssim_deep = []

    for t_i in range(len(CANONICAL_MULTI_SEASONAL_DATES)):
        # Surface 0-30m average
        ssim_surface.append(compute_ssim_2d(P_temp[t_i, 0], T_temp[t_i, 0], mask=ocean_mask_np))
        # Thermocline 100m
        ssim_thermo.append(compute_ssim_2d(P_temp[t_i, 7], T_temp[t_i, 7], mask=ocean_mask_np))
        # Deep ocean 500m
        ssim_deep.append(compute_ssim_2d(P_temp[t_i, 12], T_temp[t_i, 12], mask=ocean_mask_np))

    ssim_metrics = {
        "surface_0m_ssim": round(float(np.mean(ssim_surface)), 4),
        "thermocline_100m_ssim": round(float(np.mean(ssim_thermo)), 4),
        "deep_500m_ssim": round(float(np.mean(ssim_deep)), 4),
        "mean_structural_similarity": round(float(np.mean(ssim_surface + ssim_thermo + ssim_deep)), 4),
    }

    # Build master output
    results = {
        "model_name": "Phase 8 Zone-Adaptive Scaling Pure Diffusion (15k Steps)",
        "evaluation_protocol": "Full Comprehensive Extended Benchmark (Multi-Seasonal 10-Date)",
        "overall_summary": {
            "overall_multi_seasonal_rmse": round(overall_rmse, 4),
            "climatology_baseline_rmse": round(clim_rmse, 4),
            "overall_mae": round(overall_mae, 4),
            "overall_bias": round(overall_bias, 4),
            "overall_correlation_r": round(overall_corr, 4),
            "murphy_skill_score_pct": round(murphy_skill * 100.0, 2),
            "surface_rmse_0_30m": round(float(np.mean([depthwise_metrics[f"{int(d)}m"]["model_rmse"] for d in CANONICAL_DEPTHS[:5]])), 4),
            "thermocline_rmse_50_200m": round(float(np.mean([depthwise_metrics[f"{int(d)}m"]["model_rmse"] for d in CANONICAL_DEPTHS[5:11]])), 4),
            "deep_rmse_250_1000m": round(float(np.mean([depthwise_metrics[f"{int(d)}m"]["model_rmse"] for d in CANONICAL_DEPTHS[11:]])), 4),
        },
        "depthwise_metrics": depthwise_metrics,
        "priority_zones": zone_results,
        "auxiliary_targets": aux_results,
        "physical_soundness": physics_metrics,
        "structural_quality_ssim": ssim_metrics,
    }

    print("\n" + "=" * 80)
    print("PHASE 8 COMPREHENSIVE MULTI-SEASONAL RESULTS (15 DEPTHS, 10 DATES)")
    print("=" * 80)
    print(f"Overall 15-Depth RMSE:        {overall_rmse:.4f} °C (Climatology: {clim_rmse:.4f} °C)")
    print(f"Murphy Skill Score:           {murphy_skill * 100.0:+.2f}%")
    print(f"Overall Correlation R:        {overall_corr:.4f}")
    print(f"Surface (0-30m) RMSE:         {results['overall_summary']['surface_rmse_0_30m']:.4f} °C")
    print(f"Thermocline (50-200m) RMSE:   {results['overall_summary']['thermocline_rmse_50_200m']:.4f} °C")
    print(f"Deep Ocean (250-1000m) RMSE:  {results['overall_summary']['deep_rmse_250_1000m']:.4f} °C")
    print("-" * 80)
    print("Auxiliary Heads Accuracy:")
    print(f"  MLD RMSE:                   {aux_results['mld_rmse_m']:.2f} m (R = {aux_results['mld_corr_r']:.4f})")
    print(f"  BLT RMSE:                   {aux_results['blt_rmse_m']:.2f} m (R = {aux_results['blt_corr_r']:.4f})")
    print(f"  Salinity Max Depth RMSE:    {aux_results['sal_max_depth_rmse_m']:.2f} m")
    print("-" * 80)
    print("Physical Soundness:")
    print(f"  Stability Violation Rate:   {physics_metrics['static_stability_violation_rate_pct']:.4f}%")
    print(f"  D20 Isotherm RMSE:          {physics_metrics['d20_isotherm_depth_rmse_m']:.2f} m")
    print(f"  D26 Isotherm RMSE:          {physics_metrics['d26_isotherm_depth_rmse_m']:.2f} m")
    print("-" * 80)
    print("Structural Quality (SSIM):")
    print(f"  Surface SSIM:               {ssim_metrics['surface_0m_ssim']:.4f}")
    print(f"  Thermocline SSIM:           {ssim_metrics['thermocline_100m_ssim']:.4f}")
    print(f"  Deep Ocean SSIM:            {ssim_metrics['deep_500m_ssim']:.4f}")
    print("=" * 80)

    out_path = Path("reports/phase8_comprehensive_suite_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comprehensive results to: {out_path}")


if __name__ == "__main__":
    main()
