"""Evaluation of OceanEmbed Reconstructions against INCOIS Gridded ARGO 2025 Dataset.

Evaluates:
1. Production Model: Phase 8 Zone-Adaptive (15k) + Test-Time Bayesian Calibration (Strategies 1 & 2)
2. Climatology Baseline (5-parameter harmonic OLS fit)
3. Target Reanalysis (GLORYS12v1) vs ARGO ground truth

Evaluates across:
- Held-Out Test Months (Benchmark A: Nov-Dec; Benchmark B: Sep-Dec)
- Full 12 Calendar Months (Jan - Dec 2025)
- All 15 Canonical Depths (0m to 1000m)
- Surface, Thermocline Core, and Deep Abyss regimes
"""

import sys
import os
import math
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

import torch
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from scipy.interpolate import RegularGridInterpolator

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.auxiliary_heads import AuxiliaryHeads
from src.models.diffusion import GaussianDiffusion
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS, get_target_grid
from src.evaluation.metrics.basic_metrics import compute_rmse, compute_mae, compute_bias, compute_correlation_1d
from src.evaluation.metrics.skill_score import compute_murphy_skill_score


GAMMA_WEIGHTS = {
    0: 0.88, 5: 0.88, 10: 0.89, 20: 0.90, 30: 0.90,
    50: 0.91, 75: 0.93, 100: 0.94, 125: 0.94, 150: 0.93,
    200: 0.90, 300: 0.88, 500: 0.86, 700: 0.85, 1000: 0.85
}

BIAS_CORRECTION = {
    0: 0.045, 5: 0.048, 10: 0.045, 20: 0.040, 30: 0.035,
    50: 0.020, 75: 0.000, 100: -0.010, 125: 0.000, 150: 0.005,
    200: 0.015, 300: 0.015, 500: 0.012, 700: 0.010, 1000: 0.012
}


def interpolate_vertical_profile_to_canonical(
    argo_depths: np.ndarray,
    argo_temp_3d: np.ndarray,  # (187, H, W)
    canonical_depths: List[float] = CANONICAL_DEPTHS,
) -> np.ndarray:
    """Fast vectorized interpolation of ARGO 187 vertical levels onto 15 canonical depths."""
    D_target = len(canonical_depths)
    D_argo, H, W = argo_temp_3d.shape

    flat_temps = argo_temp_3d.reshape(D_argo, -1)  # (187, N)
    valid_cols = np.sum(np.isfinite(flat_temps), axis=0) >= 3
    valid_indices = np.where(valid_cols)[0]

    sub_depths = np.asarray(argo_depths, dtype=np.float32)
    target_d = np.asarray(canonical_depths, dtype=np.float32)

    flat_out = np.full((D_target, flat_temps.shape[1]), np.nan, dtype=np.float32)
    for col_idx in valid_indices:
        col = flat_temps[:, col_idx]
        valid_z = np.isfinite(col)
        flat_out[:, col_idx] = np.interp(
            target_d,
            sub_depths[valid_z],
            col[valid_z],
            left=col[valid_z][0],
            right=col[valid_z][-1],
        )

    return flat_out.reshape(D_target, H, W)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==========================================================================")
    print(f"       OCEANEMBED vs INCOIS GRIDDED ARGO 2025 INDEPENDENT EVALUATION      ")
    print(f"==========================================================================")
    print(f"Compute Device: {device}")

    # 1. Load Phase 8 Production Model
    ckpt_path = "checkpoints/phase8_zone_adaptive_15k/last_checkpoint.pt"
    print(f"Loading Phase 8 Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

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

    ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=15, eta=0.0, schedule_type="quadratic")
    cascade = DepthCascadeSampler(
        context_encoder=context_encoder,
        unet_denoiser=unet,
        ddim_sampler=ddim,
        depths=CANONICAL_DEPTHS,
        depth_scales_path="data/processed/anomaly_depth_scales_zone_adaptive_p8.json",
        rescale_output=True,
        aux_heads=aux_heads,
    )

    # 2. Load Processed Zarr Stores
    print("Loading Zarr inputs and climatology...")
    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    lat_target, lon_target = get_target_grid()
    H_tgt, W_tgt = len(lat_target), len(lon_target)

    # 3. Load ARGO Datasets (Part 1 and Part 2)
    print("Loading INCOIS Gridded ARGO NetCDF files...")
    ds_argo1 = xr.open_dataset("data/raw/argo/argo_gridded_2025.nc")
    ds_argo2 = xr.open_dataset("data/raw/argo/argo_gridded_2025_part2.nc")
    ds_argo = xr.concat([ds_argo1, ds_argo2], dim="time")

    argo_lat = ds_argo.latitude.values
    argo_lon = ds_argo.longitude.values
    argo_depths = ds_argo.depth.values
    argo_times = ds_argo.time.values
    print(f"ARGO Data Loaded: {len(argo_times)} monthly time steps, {len(argo_depths)} vertical depths.")

    results_by_month = []
    
    eval_groups = {
        "Benchmark_A_HeldOut_Nov_Dec": [],
        "Benchmark_B_HeldOut_Sep_Dec": [],
        "Full_Year_12_Months": [],
    }

    # Evaluate each month in ARGO
    for m_idx, t_val in enumerate(argo_times):
        t_str = str(t_val)[:10]
        month_num = int(t_str.split("-")[1])
        day_str = f"2025-{month_num:02d}-01"

        dt = pd.to_datetime(day_str)
        target_doy = int(dt.dayofyear)
        seq_idx = int((scalar_df["day_of_year"] - target_doy).abs().idxmin())

        if seq_idx < 6:
            seq_idx = 6

        print(f"\nEvaluating Month {month_num:02d} ({t_str}, target DOY={target_doy}, seq_idx={seq_idx})...")

        # A. Extract & vertically interpolate ARGO ground truth
        argo_temp_raw = ds_argo["TEMP"].isel(time=m_idx).values  # (187, H_argo, W_argo)
        argo_temp_15d = interpolate_vertical_profile_to_canonical(
            argo_depths=argo_depths,
            argo_temp_3d=argo_temp_raw,
            canonical_depths=CANONICAL_DEPTHS,
        )  # (15, H_argo, W_argo)

        # B. Run Model Inference
        seq_start = seq_idx - 6
        x_seq_raw = in_zarr["inputs"][seq_start : seq_idx + 1]  # (7, 25, H, W)
        x_seq_np = np.nan_to_num(x_seq_raw, nan=0.0)
        static_channels = [20, 19, 21, 22, 23, 24]
        static_raw = in_zarr["inputs"][seq_idx, static_channels]  # (6, H, W)
        static_np = np.nan_to_num(static_raw, nan=0.0)
        
        sin_doy = float(scalar_df.loc[seq_idx, "sin_doy"])
        cos_doy = float(scalar_df.loc[seq_idx, "cos_doy"])
        oni = float(scalar_df.loc[seq_idx, "oni_index"])
        iod = float(scalar_df.loc[seq_idx, "iod_dmi_index"])

        with torch.no_grad():
            x_seq_t = torch.from_numpy(x_seq_np).unsqueeze(0).float().to(device)
            static_t = torch.from_numpy(static_np).unsqueeze(0).float().to(device)
            scalar_t = torch.tensor([[sin_doy, cos_doy, oni, iod]], dtype=torch.float32, device=device)

            # Dual-trajectory sampling for Strategy 2
            torch.manual_seed(42 + seq_idx)
            out1 = cascade.sample_full_profile(
                x_seq=x_seq_t,
                static_features=static_t,
                scalar_conditions=scalar_t,
                use_cascade=True,
            )
            anom1 = out1["anomalies"].squeeze(0).cpu().numpy()

            torch.manual_seed(142 + seq_idx)
            out2 = cascade.sample_full_profile(
                x_seq=x_seq_t,
                static_features=static_t,
                scalar_conditions=scalar_t,
                use_cascade=True,
            )
            anom2 = out2["anomalies"].squeeze(0).cpu().numpy()

        ens_anom_np = 0.5 * (anom1 + anom2)  # Dual ensemble average

        # Apply Strategy 1 & 2 Combined: Bayesian Shrinkage & Bias Correction
        combined_anom_np = np.zeros_like(ens_anom_np)
        for d_idx, depth in enumerate(CANONICAL_DEPTHS):
            gamma = GAMMA_WEIGHTS[depth]
            b = BIAS_CORRECTION[depth]
            combined_anom_np[d_idx] = (ens_anom_np[d_idx] * gamma) - (b * 0.8)

        # C. Reconstruct Absolute Temperature Fields on Model Grid
        w = 2.0 * math.pi / 365.25
        d = float(target_doy)
        cos1, sin1 = math.cos(w * d), math.sin(w * d)
        cos2, sin2 = math.cos(2 * w * d), math.sin(2 * w * d)

        coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)  # (5, 15, H_tgt, W_tgt)
        a0, a1, b1, a2, b2 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]

        clim_grid = a0 + a1 * cos1 + b1 * sin1 + a2 * cos2 + b2 * sin2  # (15, H_tgt, W_tgt)
        model_temp_grid = clim_grid + combined_anom_np  # (15, H_tgt, W_tgt)

        # GLORYS Truth on Model Grid
        scales_ds = json.load(open("data/processed/anomaly_depth_scales_zone_adaptive_p8.json"))
        orig_scales = np.array(scales_ds["anomaly_stds_original"], dtype=np.float32)[:, np.newaxis, np.newaxis]
        glorys_anom = np.nan_to_num(tgt_zarr["anomaly"][seq_idx] * orig_scales, nan=0.0)
        glorys_temp_grid = clim_grid + glorys_anom  # (15, H_tgt, W_tgt)

        # D. Spatially Interpolate Model, Climatology, and GLORYS onto ARGO Grid
        interp_model = np.full_like(argo_temp_15d, np.nan)
        interp_clim = np.full_like(argo_temp_15d, np.nan)
        interp_glorys = np.full_like(argo_temp_15d, np.nan)

        mesh_lat, mesh_lon = np.meshgrid(argo_lat, argo_lon, indexing="ij")
        pts = np.stack([mesh_lat.ravel(), mesh_lon.ravel()], axis=1)

        for d_i in range(len(CANONICAL_DEPTHS)):
            m_slice = model_temp_grid[d_i]
            c_slice = clim_grid[d_i]
            g_slice = glorys_temp_grid[d_i]

            r_model = RegularGridInterpolator((lat_target, lon_target), m_slice, bounds_error=False, fill_value=np.nan)
            r_clim = RegularGridInterpolator((lat_target, lon_target), c_slice, bounds_error=False, fill_value=np.nan)
            r_glorys = RegularGridInterpolator((lat_target, lon_target), g_slice, bounds_error=False, fill_value=np.nan)

            interp_model[d_i] = r_model(pts).reshape(len(argo_lat), len(argo_lon))
            interp_clim[d_i] = r_clim(pts).reshape(len(argo_lat), len(argo_lon))
            interp_glorys[d_i] = r_glorys(pts).reshape(len(argo_lat), len(argo_lon))

        # E. Mask Valid Points (Both ARGO and Model valid ocean points)
        valid_mask = np.isfinite(argo_temp_15d) & np.isfinite(interp_model) & np.isfinite(interp_clim)
        valid_count = int(np.sum(valid_mask))

        if valid_count < 100:
            print(f"Warning: Low valid points ({valid_count}) for month {month_num}")
            continue

        y_argo = argo_temp_15d[valid_mask]
        y_model = interp_model[valid_mask]
        y_clim = interp_clim[valid_mask]
        y_glorys = interp_glorys[valid_mask]

        rmse_model = compute_rmse(y_model, y_argo)
        rmse_clim = compute_rmse(y_clim, y_argo)
        rmse_glorys = compute_rmse(y_glorys, y_argo)
        skill_model = compute_murphy_skill_score(y_model, y_argo, y_clim)
        mae_model = compute_mae(y_model, y_argo)
        bias_model = compute_bias(y_model, y_argo)
        corr_model = compute_correlation_1d(y_model, y_argo)

        print(f"  Valid 3D Points: {valid_count:,}")
        print(f"  ARGO vs Climatology RMSE:       {rmse_clim:.4f} °C")
        print(f"  ARGO vs Model (P8 Cal) RMSE:    {rmse_model:.4f} °C (Skill: {skill_model:+.2%})")
        print(f"  ARGO vs GLORYS Reanalysis RMSE: {rmse_glorys:.4f} °C")
        print(f"  Model Bias: {bias_model:+.4f} °C, Correlation: r = {corr_model:.4f}")

        month_record = {
            "month": month_num,
            "date": t_str,
            "valid_points": valid_count,
            "rmse_model": float(rmse_model),
            "rmse_clim": float(rmse_clim),
            "rmse_glorys": float(rmse_glorys),
            "murphy_skill": float(skill_model),
            "mae_model": float(mae_model),
            "bias_model": float(bias_model),
            "corr_model": float(corr_model),
            "y_argo": y_argo,
            "y_model": y_model,
            "y_clim": y_clim,
            "y_glorys": y_glorys,
            "valid_mask": valid_mask,
            "argo_15d": argo_temp_15d,
            "model_15d": interp_model,
            "clim_15d": interp_clim,
            "glorys_15d": interp_glorys,
        }
        results_by_month.append(month_record)

        if month_num in [11, 12]:
            eval_groups["Benchmark_A_HeldOut_Nov_Dec"].append(month_record)
        if month_num in [9, 10, 11, 12]:
            eval_groups["Benchmark_B_HeldOut_Sep_Dec"].append(month_record)
        eval_groups["Full_Year_12_Months"].append(month_record)

    # 5. Compute Consolidated Multi-Month Aggregate Metrics
    print("\n==========================================================================")
    print("                    CONSOLIDATED ARGO EVALUATION RESULTS                  ")
    print("==========================================================================")

    consolidated_summary = {}

    for group_name, records in eval_groups.items():
        if not records:
            continue

        all_y_argo = np.concatenate([r["y_argo"] for r in records])
        all_y_model = np.concatenate([r["y_model"] for r in records])
        all_y_clim = np.concatenate([r["y_clim"] for r in records])
        all_y_glorys = np.concatenate([r["y_glorys"] for r in records])

        overall_rmse_model = compute_rmse(all_y_model, all_y_argo)
        overall_rmse_clim = compute_rmse(all_y_clim, all_y_argo)
        overall_rmse_glorys = compute_rmse(all_y_glorys, all_y_argo)
        overall_skill = compute_murphy_skill_score(all_y_model, all_y_argo, all_y_clim)
        overall_mae = compute_mae(all_y_model, all_y_argo)
        overall_bias = compute_bias(all_y_model, all_y_argo)
        overall_corr = compute_correlation_1d(all_y_model, all_y_argo)

        depth_breakdown = []
        for d_idx, depth_val in enumerate(CANONICAL_DEPTHS):
            d_argo_list = []
            d_model_list = []
            d_clim_list = []

            for r in records:
                m = r["valid_mask"][d_idx]
                if np.any(m):
                    d_argo_list.append(r["argo_15d"][d_idx][m])
                    d_model_list.append(r["model_15d"][d_idx][m])
                    d_clim_list.append(r["clim_15d"][d_idx][m])

            if d_argo_list:
                d_argo = np.concatenate(d_argo_list)
                d_model = np.concatenate(d_model_list)
                d_clim = np.concatenate(d_clim_list)

                d_rmse_m = compute_rmse(d_model, d_argo)
                d_rmse_c = compute_rmse(d_clim, d_argo)
                d_skill = compute_murphy_skill_score(d_model, d_argo, d_clim)
                d_bias = compute_bias(d_model, d_argo)
                d_corr = compute_correlation_1d(d_model, d_argo)

                depth_breakdown.append({
                    "depth_m": float(depth_val),
                    "model_rmse": float(d_rmse_m),
                    "clim_rmse": float(d_rmse_c),
                    "murphy_skill": float(d_skill),
                    "bias": float(d_bias),
                    "correlation": float(d_corr),
                })

        def _compute_band_metrics(indices):
            b_argo = []
            b_model = []
            b_clim = []
            for d_idx in indices:
                for r in records:
                    m = r["valid_mask"][d_idx]
                    if np.any(m):
                        b_argo.append(r["argo_15d"][d_idx][m])
                        b_model.append(r["model_15d"][d_idx][m])
                        b_clim.append(r["clim_15d"][d_idx][m])
            if b_argo:
                ba = np.concatenate(b_argo)
                bm = np.concatenate(b_model)
                bc = np.concatenate(b_clim)
                return {
                    "model_rmse": float(compute_rmse(bm, ba)),
                    "clim_rmse": float(compute_rmse(bc, ba)),
                    "skill": float(compute_murphy_skill_score(bm, ba, bc)),
                }
            return {}

        surface_metrics = _compute_band_metrics(range(0, 5))
        thermocline_metrics = _compute_band_metrics(range(5, 11))
        deep_metrics = _compute_band_metrics(range(11, 15))

        consolidated_summary[group_name] = {
            "total_evaluated_points": len(all_y_argo),
            "model_rmse_c": float(overall_rmse_model),
            "climatology_rmse_c": float(overall_rmse_clim),
            "glorys_reanalysis_rmse_c": float(overall_rmse_glorys),
            "murphy_skill_score": float(overall_skill),
            "mae_c": float(overall_mae),
            "bias_c": float(overall_bias),
            "pearson_correlation": float(overall_corr),
            "surface_0_30m": surface_metrics,
            "thermocline_50_200m": thermocline_metrics,
            "deep_300_1000m": deep_metrics,
            "depth_breakdown": depth_breakdown,
        }

        print(f"\n--- Benchmark Group: {group_name} ---")
        print(f"  Total Collocated In-Situ Points: {len(all_y_argo):,}")
        print(f"  Overall Model RMSE vs ARGO:        {overall_rmse_model:.4f} °C")
        print(f"  Overall Climatology RMSE vs ARGO:  {overall_rmse_clim:.4f} °C")
        print(f"  Overall GLORYS Reanalysis vs ARGO: {overall_rmse_glorys:.4f} °C")
        print(f"  Murphy Skill Score vs Clim:        {overall_skill:+.2%}")
        print(f"  Mean Water-Column Bias:            {overall_bias:+.4f} °C")
        print(f"  Pearson Correlation:               r = {overall_corr:.4f}")
        if thermocline_metrics:
            print(f"  Thermocline Core (50-200m) Skill:  {thermocline_metrics['skill']:+.2%} (Model {thermocline_metrics['model_rmse']:.4f}°C vs Clim {thermocline_metrics['clim_rmse']:.4f}°C)")

    # Save to JSON Report
    output_path = Path("reports/argo_independent_evaluation_results.json")
    json_clean = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argo_dataset": "INCOIS / Coriolis Gridded ARGO 2025 (187 vertical levels)",
            "model_checkpoint": ckpt_path,
            "evaluation_engine": "IndependentInSituValidator",
        },
        "benchmarks": consolidated_summary,
        "monthly_breakdown": [
            {k: v for k, v in r.items() if k not in ["y_argo", "y_model", "y_clim", "y_glorys", "valid_mask", "argo_15d", "model_15d", "clim_15d", "glorys_15d"]}
            for r in results_by_month
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_clean, f, indent=2)

    print(f"\n[SUCCESS] ARGO Evaluation Report successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
