"""Comprehensive Genuine Evaluation Runner for OceanEmbed 20,000-Step Checkpoints.

Evaluates:
- Stage B 20k Baseline (Region ON, Cascade ON)
- Stage C 20k Ablation (Region OFF, Cascade ON)

Incorporates:
1. Real DDIM stochastic sampling with eta=0.3 (Fix A1).
2. Unpooled per-depth meridional heat flux consistency correlation.
3. Updated Priority Zone 2 (Thermocline Core: 20-200m, Part B).
4. Exact SHA-256 checksum and ISO timestamp for reproducibility.
5. Real held-out test set performance (Nov-Dec 2025).
"""

import sys
import os
import math
import time
import json
import hashlib
import datetime
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
from src.models.auxiliary_heads import AuxiliaryHeads
from src.models.diffusion import GaussianDiffusion
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS
from src.evaluation.metrics.basic_metrics import (
    compute_rmse,
    compute_mae,
    compute_bias,
    compute_all_basic_metrics,
)
from src.evaluation.metrics.skill_score import compute_murphy_skill_score
from src.evaluation.metrics.ssim_metric import compute_profile_ssim, compute_ssim_2d
from src.evaluation.metrics.heat_flux_consistency import evaluate_heat_flux_consistency
from src.evaluation.metrics.calibration import evaluate_ensemble_calibration
from src.evaluation.slicing.priority_zones import get_priority_zones
from src.evaluation.slicing.zone_evaluator import evaluate_metric_by_zone
from src.evaluation.auxiliary_head_eval import evaluate_auxiliary_predictions


def get_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_model(checkpoint_path: str, device: torch.device):
    """Load model components from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    context_encoder = ContextEncoder(in_channels=25, hidden_dims=[32, 64, 64]).to(device)
    unet = UNetDenoiser(in_channels=72, stage_channels=[32, 64, 128, 256], cond_in_dim=8).to(device)
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
    return context_encoder, unet, aux_heads, diffusion, ckpt


def evaluate_20k_model(
    stage_name: str,
    checkpoint_path: str,
    in_zarr: Any,
    tgt_zarr: Any,
    aux_zarr: Any,
    scalar_df: pd.DataFrame,
    clim_coeffs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: np.ndarray,
    eval_target_days: List[int],
    use_cascade: bool = True,
    use_region: bool = True,
    eta: float = 0.3,
    device: torch.device = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cpu")

    ckpt_hash = get_sha256(checkpoint_path)
    eval_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"\n================================================================================")
    print(f"EVALUATING: {stage_name}")
    print(f"Checkpoint: {checkpoint_path} (SHA-256: {ckpt_hash[:12]}...)")
    print(f"Timestamp: {eval_timestamp} | DDIM eta: {eta}")
    print(f"================================================================================")

    enc, unet, aux_heads, diff, ckpt_obj = load_model(checkpoint_path, device)
    ddim = DDIMSampler(diff, num_ddim_timesteps=10, eta=eta)
    cascade = DepthCascadeSampler(enc, unet, ddim, CANONICAL_DEPTHS)

    static_channels = [20, 19, 21, 22, 23, 24]
    H, W = len(lat), len(lon)

    pred_temps_list = []
    true_temps_list = []
    clim_temps_list = []
    timestamps = []

    pred_mld_list = []
    pred_blt_list = []
    pred_sal_depth_list = []
    pred_sal_str_list = []

    true_mld_list = []
    true_blt_list = []
    true_sal_depth_list = []
    true_sal_str_list = []

    last_v_input = None

    for t_day in eval_target_days:
        date_str = scalar_df.loc[t_day, "date"]
        doy = int(scalar_df.loc[t_day, "day_of_year"])
        sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
        cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
        oni = float(scalar_df.loc[t_day, "oni_index"])
        iod = float(scalar_df.loc[t_day, "iod_dmi_index"])
        timestamps.append(date_str)

        # 7-day sequence input: t_day - 6 to t_day
        seq_slice = in_zarr["inputs"][t_day - 6 : t_day + 1]  # (7, 25, H, W)
        seq_slice = np.nan_to_num(seq_slice, nan=0.0)
        x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)

        static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
        if not use_region:
            static_feats[:, 2:6] = 0.0

        # Scalar conditioning vector [sin_doy, cos_doy, oni, iod]
        scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], device=device, dtype=torch.float32)

        with torch.no_grad():
            out = cascade.sample_full_profile(
                x_seq=x_seq,
                static_features=static_feats,
                scalar_conditions=scalar_cond,
                use_cascade=use_cascade,
            )
            pred_anom = out["anomalies"][0].cpu().numpy()

            # Auxiliary heads
            u_cond = enc(x_seq)
            aux_out = aux_heads(u_cond)
            pred_mld_list.append(aux_out["mld"][0].item())
            pred_blt_list.append(aux_out["blt"][0].item())
            pred_sal_depth_list.append(aux_out["sal_max_depth"][0].item())
            pred_sal_str_list.append(aux_out["sal_max_strength"][0].item())

        # Climatology for date
        omega = 2.0 * math.pi / 365.25
        clim_t = (
            clim_coeffs[0]
            + clim_coeffs[1] * math.cos(omega * doy)
            + clim_coeffs[2] * math.sin(omega * doy)
            + clim_coeffs[3] * math.cos(2 * omega * doy)
            + clim_coeffs[4] * math.sin(2 * omega * doy)
        )

        true_anom = np.nan_to_num(tgt_zarr["anomaly"][t_day].copy(), nan=0.0)
        # extrapolate 5m (idx 1) to 0m (idx 0)
        true_anom[0] = true_anom[1]

        pred_temp = pred_anom + clim_t
        true_temp = true_anom + clim_t

        pred_temps_list.append(pred_temp)
        true_temps_list.append(true_temp)
        clim_temps_list.append(clim_t)

        # Auxiliary ground truth
        aux_day = np.nan_to_num(aux_zarr["auxiliary_targets"][t_day], nan=0.0)
        true_mld_list.append(float(np.mean(aux_day[0][ocean_mask])))
        true_blt_list.append(float(np.mean(aux_day[1][ocean_mask])))
        true_sal_depth_list.append(float(np.mean(aux_day[2][ocean_mask])))
        true_sal_str_list.append(float(np.mean(aux_day[3][ocean_mask])))

        # Save surface current V for heat flux
        last_v_input = seq_slice[-1, 6]  # channel 6 is current_v

    pred_temps = np.array(pred_temps_list)  # (N, 15, H, W)
    true_temps = np.array(true_temps_list)  # (N, 15, H, W)
    clim_temps = np.array(clim_temps_list)  # (N, 15, H, W)

    # 1. Global Metrics
    global_basic = compute_all_basic_metrics(pred_temps, true_temps, mask=ocean_mask)
    murphy_ss = compute_murphy_skill_score(pred_temps, true_temps, clim_temps, mask=ocean_mask)

    # 2. Depthwise breakdown
    depthwise_metrics = {}
    for d_idx, d in enumerate(CANONICAL_DEPTHS):
        p_d = pred_temps[:, d_idx]
        t_d = true_temps[:, d_idx]
        c_d = clim_temps[:, d_idx]

        rmse_d = compute_rmse(p_d, t_d, mask=ocean_mask)
        mae_d = compute_mae(p_d, t_d, mask=ocean_mask)
        bias_d = compute_bias(p_d, t_d, mask=ocean_mask)
        skill_d = compute_murphy_skill_score(p_d, t_d, c_d, mask=ocean_mask)

        # Per-depth spatial correlation
        d_corrs = []
        for n_i in range(len(eval_target_days)):
            p_slice = p_d[n_i]
            t_slice = t_d[n_i]
            valid = ocean_mask & np.isfinite(p_slice) & np.isfinite(t_slice)
            if np.sum(valid) > 10:
                pv = p_slice[valid]
                tv = t_slice[valid]
                if pv.std() > 1e-6 and tv.std() > 1e-6:
                    d_corrs.append(float(np.corrcoef(pv, tv)[0, 1]))
        corr_d = float(np.mean(d_corrs)) if d_corrs else 0.0

        ssim_d = float(compute_ssim_2d(np.mean(p_d, axis=0), np.mean(t_d, axis=0), mask=ocean_mask))

        depthwise_metrics[f"{d}m"] = {
            "rmse": float(rmse_d),
            "mae": float(mae_d),
            "bias": float(bias_d),
            "correlation": float(corr_d),
            "murphy_skill_score": float(skill_d),
            "ssim": float(ssim_d),
        }

    # 3. Priority Zones (using newly redefined 20-200m Thermocline Core)
    zones = get_priority_zones(lat=lat, lon=lon)
    zone_rmses = evaluate_metric_by_zone(
        metric_fn=compute_rmse,
        pred=pred_temps,
        target=true_temps,
        zones=zones,
        timestamps=timestamps,
        ocean_mask=ocean_mask,
    )
    zone_maes = evaluate_metric_by_zone(
        metric_fn=compute_mae,
        pred=pred_temps,
        target=true_temps,
        zones=zones,
        timestamps=timestamps,
        ocean_mask=ocean_mask,
    )
    zone_metrics = {}
    for z_slug, z_obj in zones.items():
        z_rmse = zone_rmses.get(z_slug, 0.0)
        z_mae = zone_maes.get(z_slug, 0.0)
        zone_metrics[z_slug] = {
            "zone_id": z_obj.zone_id,
            "name": z_obj.name,
            "depth_range": f"{min(z_obj.depth_values)}-{max(z_obj.depth_values)}m",
            "rmse": float(z_rmse) if isinstance(z_rmse, (int, float)) and not np.isnan(z_rmse) else 0.0,
            "mae": float(z_mae) if isinstance(z_mae, (int, float)) and not np.isnan(z_mae) else 0.0,
        }

    # 4. Heat Flux Consistency (unpooled per-depth correlation across all 15 depths)
    heat_flux_results = evaluate_heat_flux_consistency(last_v_input, pred_temps[-1], true_temps[-1], mask=ocean_mask)

    # 5. Real DDIM Stochastic Ensemble Calibration
    n_ens = 5
    ens_preds = []
    print(f"Generating N={n_ens} DDIM stochastic ensemble members (eta={eta})...")
    for ens_i in range(n_ens):
        torch.manual_seed(100 + ens_i)
        with torch.no_grad():
            out_ens = cascade.sample_full_profile(x_seq, static_feats, scalar_cond, use_cascade=use_cascade)
            ens_preds.append(out_ens["anomalies"][0].cpu().numpy() + clim_temps_list[-1])
    ens_arr = np.array(ens_preds)
    std_map = np.std(ens_arr, axis=0)
    raw_spread = float(np.mean(std_map[:, ocean_mask]))
    calibration_results = evaluate_ensemble_calibration(ens_arr, true_temps[-1], mask=ocean_mask)

    # 6. Auxiliary predictions
    aux_eval = evaluate_auxiliary_predictions(
        pred_aux={
            "mld": np.array(pred_mld_list),
            "blt": np.array(pred_blt_list),
            "sal_max_depth": np.array(pred_sal_depth_list),
            "sal_max_strength": np.array(pred_sal_str_list),
        },
        true_aux={
            "mld": np.array(true_mld_list),
            "blt": np.array(true_blt_list),
            "sal_max_depth": np.array(true_sal_depth_list),
            "sal_max_strength": np.array(true_sal_str_list),
        },
    )

    # 7. Held-out Test Set Slicing (eval dates in Nov-Dec: indices 318, 331, 358)
    test_date_indices = [i for i, d in enumerate(eval_target_days) if d >= 298]
    if test_date_indices:
        test_pred = pred_temps[test_date_indices]
        test_true = true_temps[test_date_indices]
        held_out_test_rmse = float(compute_rmse(test_pred, test_true, mask=ocean_mask))
    else:
        held_out_test_rmse = 0.0

    summary = {
        "stage_name": stage_name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": ckpt_hash,
        "evaluation_timestamp": eval_timestamp,
        "ddim_eta": eta,
        "overall_rmse": float(global_basic["rmse"]),
        "overall_mae": float(global_basic["mae"]),
        "overall_bias": float(global_basic["bias"]),
        "overall_correlation": float(global_basic["correlation"]),
        "murphy_skill_score": float(murphy_ss),
        "held_out_test_rmse": held_out_test_rmse,
        "raw_ensemble_spread_c": raw_spread,
        "calibration_ece": float(calibration_results["expected_calibration_error"]),
        "calibration_diagnostic": calibration_results["calibration_diagnostics"],
        "heat_flux_relative_rmse_pct": float(heat_flux_results["heat_flux_relative_rmse_pct"]),
        "heat_flux_correlation": float(heat_flux_results["heat_flux_correlation"]),
        "depthwise_metrics": depthwise_metrics,
        "zone_metrics": zone_metrics,
        "aux_results": aux_eval,
    }

    print(f"\n--- {stage_name} Results ---")
    print(f"Overall Multi-Seasonal RMSE: {summary['overall_rmse']:.4f}°C")
    print(f"Held-Out Test Set RMSE (Nov-Dec): {summary['held_out_test_rmse']:.4f}°C")
    print(f"Pearson Correlation (Unpooled): {summary['overall_correlation']:.4f}")
    print(f"Murphy Skill Score: {summary['murphy_skill_score']:.4f}")
    print(f"Raw DDIM Ensemble Spread: {summary['raw_ensemble_spread_c']:.4f}°C")
    print(f"Calibration ECE (Raw DDIM): {summary['calibration_ece']:.4f}")
    print(f"Heat Flux Relative RMSE: {summary['heat_flux_relative_rmse_pct']:.2f}% (Corr: {summary['heat_flux_correlation']:.4f})")
    print(f"Zone 2 (Thermocline Core 20-200m) RMSE: {zone_metrics['zone2_thermocline_core']['rmse']:.4f}°C")
    return summary


def main():
    print("=" * 80)
    print("OCEANEMBED 20,000-STEP COMPREHENSIVE PRODUCTION RE-EVALUATION")
    print("=" * 80)

    device = torch.device("cpu")
    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    aux_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_auxiliary_targets.zarr", mode="r")
    clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    lat = in_zarr["lat"][:]
    lon = in_zarr["lon"][:]
    ocean_mask = (in_zarr["inputs"][0, 20] > 0.5)
    clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)

    # 8 multi-seasonal dates across 2025:
    eval_target_days = [20, 145, 200, 268, 280, 318, 331, 358]

    # 1. Evaluate Stage B 20k Baseline
    res_b_20k = evaluate_20k_model(
        stage_name="Stage B (Baseline 20k Steps)",
        checkpoint_path="checkpoints/baseline_20k/best_checkpoint.pt",
        in_zarr=in_zarr,
        tgt_zarr=tgt_zarr,
        aux_zarr=aux_zarr,
        scalar_df=scalar_df,
        clim_coeffs=clim_coeffs,
        lat=lat,
        lon=lon,
        ocean_mask=ocean_mask,
        eval_target_days=eval_target_days,
        use_cascade=True,
        use_region=True,
        eta=0.3,
        device=device,
    )

    # 2. Evaluate Stage C 20k Ablation (No Region)
    res_c_20k = evaluate_20k_model(
        stage_name="Stage C (No Region 20k Steps)",
        checkpoint_path="checkpoints/ablation_no_region_20k/best_checkpoint.pt",
        in_zarr=in_zarr,
        tgt_zarr=tgt_zarr,
        aux_zarr=aux_zarr,
        scalar_df=scalar_df,
        clim_coeffs=clim_coeffs,
        lat=lat,
        lon=lon,
        ocean_mask=ocean_mask,
        eval_target_days=eval_target_days,
        use_cascade=True,
        use_region=False,
        eta=0.3,
        device=device,
    )

    # Save to logs
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    out_json = logs_dir / "evaluation_results_20k.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"stage_b_20k": res_b_20k, "stage_c_20k": res_c_20k}, f, indent=2)
    print(f"\nSaved consolidated 20k evaluation results to {out_json}")


if __name__ == "__main__":
    main()
