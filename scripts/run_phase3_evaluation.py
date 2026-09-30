"""Phase 3 Comprehensive Multi-Seasonal and Continuous Test Evaluation.

Evaluates Phase 3 retrained checkpoint (normalized ConvLSTM inputs, skill loss, warm restart)
against Climatology baseline and Stage B reference benchmark.

Evaluates:
1. 10 Multi-Seasonal Dates spanning all 4 seasons and transitions:
   [15, 60, 105, 150, 195, 240, 285, 318, 331, 358]
2. Continuous Nov-Dec Held-Out Test Window (Days 298 to 358, 61 days)
3. Depthwise breakdown across all 15 canonical depths (0m to 1000m)
4. All 7 Priority Zones (including Zone 2 20-200m Thermocline Core)
5. Auxiliary head diagnostics (MLD, BLT, D_max, S_max)
6. Physical meridional heat-flux consistency
7. Stochastic DDIM calibration (ECE) and deterministic DDIM (eta=0.0)
"""

import sys
import os
import math
import time
import json
import hashlib
import datetime
import argparse
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
from src.evaluation.metrics.ssim_metric import compute_ssim_2d
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


def load_model(
    checkpoint_path: str,
    device: torch.device,
    norm_stats_path: str = "data/processed/channel_normalization_stats_ocean_only.json",
):
    """Load model components from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Using normalization stats: {norm_stats_path}")
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


def evaluate_checkpoint(
    checkpoint_path: str,
    eta: float = 0.0,
    eval_continuous: bool = True,
    device_str: str = "auto",
    norm_stats_path: str = "data/processed/channel_normalization_stats_ocean_only.json",
    output_path: str = "logs/phase3_evaluation_results.json",
) -> Dict[str, Any]:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    ckpt_hash = get_sha256(checkpoint_path)
    eval_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"================================================================================")
    print(f"PHASE 3 EVALUATION: {checkpoint_path}")
    print(f"SHA-256: {ckpt_hash}")
    print(f"Device: {device} | DDIM eta: {eta}")
    print(f"Timestamp: {eval_timestamp}")
    print(f"================================================================================")

    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    aux_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_auxiliary_targets.zarr", mode="r")
    clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    lat = in_zarr["lat"][:]
    lon = in_zarr["lon"][:]
    ocean_mask = (in_zarr["inputs"][0, 20] > 0.5)
    clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)

    enc, unet, aux_heads, diff, ckpt_obj = load_model(checkpoint_path, device, norm_stats_path=norm_stats_path)
    ddim = DDIMSampler(diff, num_ddim_timesteps=10, eta=eta)
    cascade = DepthCascadeSampler(enc, unet, ddim, CANONICAL_DEPTHS)

    static_channels = [20, 19, 21, 22, 23, 24]
    multi_seasonal_days = [15, 60, 105, 150, 195, 240, 285, 318, 331, 358]

    # Run inference for target days
    def run_days(days_list: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], List[str], Any]:
        p_list, t_list, c_list = [], [], []
        timestamps = []
        p_mld, p_blt, p_sal_d, p_sal_s = [], [], [], []
        t_mld, t_blt, t_sal_d, t_sal_s = [], [], [], []
        last_v = None

        omega = 2.0 * math.pi / 365.25

        for t_day in days_list:
            date_str = scalar_df.loc[t_day, "date"]
            doy = int(scalar_df.loc[t_day, "day_of_year"])
            sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
            cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
            oni = float(scalar_df.loc[t_day, "oni_index"])
            iod = float(scalar_df.loc[t_day, "iod_dmi_index"])
            timestamps.append(date_str)

            seq_slice = in_zarr["inputs"][t_day - 6 : t_day + 1]
            seq_slice = np.nan_to_num(seq_slice, nan=0.0)
            x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)
            static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
            scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], device=device, dtype=torch.float32)

            with torch.no_grad():
                out = cascade.sample_full_profile(
                    x_seq=x_seq,
                    static_features=static_feats,
                    scalar_conditions=scalar_cond,
                    use_cascade=True,
                )
                pred_anom = out["anomalies"][0].cpu().numpy()

                u_cond = enc(x_seq)
                aux_out = aux_heads(u_cond)
                p_mld.append(aux_out["mld"][0].item())
                p_blt.append(aux_out["blt"][0].item())
                p_sal_d.append(aux_out["sal_max_depth"][0].item())
                p_sal_s.append(aux_out["sal_max_strength"][0].item())

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

            p_list.append(pred_temp)
            t_list.append(true_temp)
            c_list.append(clim_t)

            aux_day = np.nan_to_num(aux_zarr["auxiliary_targets"][t_day], nan=0.0)
            t_mld.append(float(np.mean(aux_day[0][ocean_mask])))
            t_blt.append(float(np.mean(aux_day[1][ocean_mask])))
            t_sal_d.append(float(np.mean(aux_day[2][ocean_mask])))
            t_sal_s.append(float(np.mean(aux_day[3][ocean_mask])))

            last_v = seq_slice[-1, 6]

        aux_data = {
            "pred": {"mld": np.array(p_mld), "blt": np.array(p_blt), "sal_max_depth": np.array(p_sal_d), "sal_max_strength": np.array(p_sal_s)},
            "true": {"mld": np.array(t_mld), "blt": np.array(t_blt), "sal_max_depth": np.array(t_sal_d), "sal_max_strength": np.array(t_sal_s)},
        }
        return np.array(p_list), np.array(t_list), np.array(c_list), aux_data, timestamps, last_v

    print(f"\n1. Running Multi-Seasonal Evaluation across {len(multi_seasonal_days)} dates...")
    t0 = time.time()
    p_ms, t_ms, c_ms, aux_ms, ts_ms, last_v = run_days(multi_seasonal_days)
    ms_elapsed = time.time() - t0
    print(f"Multi-seasonal inference completed in {ms_elapsed:.2f}s ({ms_elapsed/len(multi_seasonal_days):.3f}s/date)")

    # Global basic metrics
    ms_basic = compute_all_basic_metrics(p_ms, t_ms, mask=ocean_mask)
    clim_basic = compute_all_basic_metrics(c_ms, t_ms, mask=ocean_mask)
    ms_murphy = compute_murphy_skill_score(p_ms, t_ms, c_ms, mask=ocean_mask)

    print(f"-> Multi-Seasonal RMSE: {ms_basic['rmse']:.4f}°C (Climatology: {clim_basic['rmse']:.4f}°C)")
    print(f"-> Multi-Seasonal MAE:  {ms_basic['mae']:.4f}°C (Climatology: {clim_basic['mae']:.4f}°C)")
    print(f"-> Multi-Seasonal Bias: {ms_basic['bias']:+.4f}°C")
    print(f"-> Correlation (r):     {ms_basic['correlation']:.4f}")
    print(f"-> Murphy Skill Score:  {ms_murphy:.4f}")

    # Depthwise Breakdown
    depthwise_metrics = {}
    for d_idx, d in enumerate(CANONICAL_DEPTHS):
        p_d = p_ms[:, d_idx]
        t_d = t_ms[:, d_idx]
        c_d = c_ms[:, d_idx]

        rmse_d = compute_rmse(p_d, t_d, mask=ocean_mask)
        mae_d = compute_mae(p_d, t_d, mask=ocean_mask)
        bias_d = compute_bias(p_d, t_d, mask=ocean_mask)
        skill_d = compute_murphy_skill_score(p_d, t_d, c_d, mask=ocean_mask)
        clim_rmse_d = compute_rmse(c_d, t_d, mask=ocean_mask)

        d_corrs = []
        for n_i in range(len(multi_seasonal_days)):
            pv = p_d[n_i][ocean_mask]
            tv = t_d[n_i][ocean_mask]
            if pv.std() > 1e-6 and tv.std() > 1e-6:
                d_corrs.append(float(np.corrcoef(pv, tv)[0, 1]))
        corr_d = float(np.mean(d_corrs)) if d_corrs else 0.0
        ssim_d = float(compute_ssim_2d(np.mean(p_d, axis=0), np.mean(t_d, axis=0), mask=ocean_mask))

        depthwise_metrics[f"{d}m"] = {
            "depth_m": d,
            "rmse": float(rmse_d),
            "mae": float(mae_d),
            "bias": float(bias_d),
            "clim_rmse": float(clim_rmse_d),
            "correlation": float(corr_d),
            "murphy_skill_score": float(skill_d),
            "ssim": float(ssim_d),
        }

    # Priority Zones
    zones = get_priority_zones(lat=lat, lon=lon)
    zone_rmses = evaluate_metric_by_zone(metric_fn=compute_rmse, pred=p_ms, target=t_ms, zones=zones, timestamps=ts_ms, ocean_mask=ocean_mask)
    zone_maes = evaluate_metric_by_zone(metric_fn=compute_mae, pred=p_ms, target=t_ms, zones=zones, timestamps=ts_ms, ocean_mask=ocean_mask)
    zone_clim_rmses = evaluate_metric_by_zone(metric_fn=compute_rmse, pred=c_ms, target=t_ms, zones=zones, timestamps=ts_ms, ocean_mask=ocean_mask)

    zone_metrics = {}
    for z_slug, z_obj in zones.items():
        z_rmse = float(zone_rmses.get(z_slug, 0.0))
        z_mae = float(zone_maes.get(z_slug, 0.0))
        z_c_rmse = float(zone_clim_rmses.get(z_slug, 0.0))
        z_skill = 1.0 - (z_rmse**2) / (z_c_rmse**2) if z_c_rmse > 1e-6 else 0.0
        zone_metrics[z_slug] = {
            "zone_id": z_obj.zone_id,
            "name": z_obj.name,
            "depth_range": f"{min(z_obj.depth_values)}-{max(z_obj.depth_values)}m",
            "rmse": z_rmse,
            "mae": z_mae,
            "clim_rmse": z_c_rmse,
            "skill_score": float(z_skill),
        }

    # Heat Flux Consistency
    heat_flux_results = evaluate_heat_flux_consistency(last_v, p_ms[-1], t_ms[-1], mask=ocean_mask)

    # Auxiliary head evaluation
    aux_eval = evaluate_auxiliary_predictions(aux_ms["pred"], aux_ms["true"])

    # Held-out Test Set (Nov-Dec: 318, 331, 358)
    spot_test_indices = [i for i, d in enumerate(multi_seasonal_days) if d >= 298]
    spot_test_rmse = float(compute_rmse(p_ms[spot_test_indices], t_ms[spot_test_indices], mask=ocean_mask))
    spot_test_skill = float(compute_murphy_skill_score(p_ms[spot_test_indices], t_ms[spot_test_indices], c_ms[spot_test_indices], mask=ocean_mask))

    # Continuous Nov-Dec Evaluation (Phase 4.2)
    continuous_results = None
    if eval_continuous:
        cont_days = list(range(298, 359))  # 61 consecutive days Nov 1 to Dec 31
        print(f"\n2. Running Continuous Held-Out Test Evaluation across all {len(cont_days)} days (Nov 1 - Dec 31)...")
        t0_c = time.time()
        p_c, t_c, c_c, _, _, _ = run_days(cont_days)
        cont_elapsed = time.time() - t0_c
        print(f"Continuous evaluation completed in {cont_elapsed:.2f}s ({cont_elapsed/len(cont_days):.3f}s/date)")

        c_basic = compute_all_basic_metrics(p_c, t_c, mask=ocean_mask)
        c_clim = compute_all_basic_metrics(c_c, t_c, mask=ocean_mask)
        c_skill = compute_murphy_skill_score(p_c, t_c, c_c, mask=ocean_mask)

        # Early (Nov 1 - Nov 30, Days 298-327) vs Late (Dec 1 - Dec 31, Days 328-358)
        early_p, early_t, early_c = p_c[:30], t_c[:30], c_c[:30]
        late_p, late_t, late_c = p_c[30:], t_c[30:], c_c[30:]

        early_rmse = float(compute_rmse(early_p, early_t, mask=ocean_mask))
        early_clim_rmse = float(compute_rmse(early_c, early_t, mask=ocean_mask))
        early_skill = float(compute_murphy_skill_score(early_p, early_t, early_c, mask=ocean_mask))

        late_rmse = float(compute_rmse(late_p, late_t, mask=ocean_mask))
        late_clim_rmse = float(compute_rmse(late_c, late_t, mask=ocean_mask))
        late_skill = float(compute_murphy_skill_score(late_p, late_t, late_c, mask=ocean_mask))

        continuous_results = {
            "num_days": len(cont_days),
            "start_day": 298,
            "end_day": 358,
            "start_date": scalar_df.loc[298, "date"],
            "end_date": scalar_df.loc[358, "date"],
            "continuous_rmse": float(c_basic["rmse"]),
            "continuous_mae": float(c_basic["mae"]),
            "continuous_bias": float(c_basic["bias"]),
            "continuous_clim_rmse": float(c_clim["rmse"]),
            "continuous_murphy_skill": float(c_skill),
            "continuous_correlation": float(c_basic["correlation"]),
            "early_window_nov": {
                "days": "298-327 (Nov 1 - Nov 30)",
                "rmse": early_rmse,
                "clim_rmse": early_clim_rmse,
                "murphy_skill": early_skill,
            },
            "late_window_dec": {
                "days": "328-358 (Dec 1 - Dec 31)",
                "rmse": late_rmse,
                "clim_rmse": late_clim_rmse,
                "murphy_skill": late_skill,
            },
        }
        print(f"-> Continuous Test RMSE: {c_basic['rmse']:.4f}°C (Climatology: {c_clim['rmse']:.4f}°C)")
        print(f"-> Continuous Test Skill: {c_skill:.4f}")
        print(f"   * Early Window (Nov): RMSE {early_rmse:.4f}°C vs Clim {early_clim_rmse:.4f}°C (Skill: {early_skill:.4f})")
        print(f"   * Late Window  (Dec): RMSE {late_rmse:.4f}°C vs Clim {late_clim_rmse:.4f}°C (Skill: {late_skill:.4f})")

    # Calibration check (N=5 stochastic members with eta=0.3)
    calibration_summary = {}
    if eta > 0.0:
        print(f"\n3. Evaluating Stochastic DDIM Calibration (eta={eta}, N=5)...")
        n_ens = 5
        seq_slice = in_zarr["inputs"][358 - 6 : 358 + 1]
        x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)
        static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
        doy = int(scalar_df.loc[358, "day_of_year"])
        scalar_cond = torch.tensor([[float(scalar_df.loc[358, "sin_doy"]), float(scalar_df.loc[358, "cos_doy"]), float(scalar_df.loc[358, "oni_index"]), float(scalar_df.loc[358, "iod_dmi_index"])]], device=device, dtype=torch.float32)

        ens_preds = []
        for ens_i in range(n_ens):
            torch.manual_seed(100 + ens_i)
            with torch.no_grad():
                out_ens = cascade.sample_full_profile(x_seq, static_feats, scalar_cond, use_cascade=True)
                ens_preds.append(out_ens["anomalies"][0].cpu().numpy() + c_ms[-1])
        ens_arr = np.array(ens_preds)
        std_map = np.std(ens_arr, axis=0)
        raw_spread = float(np.mean(std_map[:, ocean_mask]))
        cal_res = evaluate_ensemble_calibration(ens_arr, t_ms[-1], mask=ocean_mask)
        calibration_summary = {
            "eta": eta,
            "raw_ensemble_spread_c": raw_spread,
            "calibration_ece": float(cal_res["expected_calibration_error"]),
            "calibration_diagnostic": cal_res["calibration_diagnostics"],
        }
        print(f"-> Calibration ECE: {calibration_summary['calibration_ece']:.4f}, Spread: {raw_spread:.4f}°C")

    # Consolidate results
    results = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": ckpt_hash,
        "evaluation_timestamp": eval_timestamp,
        "device": str(device),
        "ddim_eta": eta,
        "multi_seasonal_dates": multi_seasonal_days,
        "overall_rmse": float(ms_basic["rmse"]),
        "overall_mae": float(ms_basic["mae"]),
        "overall_bias": float(ms_basic["bias"]),
        "overall_correlation": float(ms_basic["correlation"]),
        "murphy_skill_score": float(ms_murphy),
        "climatology_overall_rmse": float(clim_basic["rmse"]),
        "spot_held_out_test_rmse": spot_test_rmse,
        "spot_held_out_test_skill": spot_test_skill,
        "continuous_test_results": continuous_results,
        "depthwise_metrics": depthwise_metrics,
        "zone_metrics": zone_metrics,
        "aux_results": aux_eval,
        "heat_flux": {
            "relative_rmse_pct": float(heat_flux_results["heat_flux_relative_rmse_pct"]),
            "correlation": float(heat_flux_results["heat_flux_correlation"]),
        },
        "calibration": calibration_summary,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Phase 3 evaluation results to {out_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase3_retrain_normalized/best_checkpoint.pt")
    parser.add_argument("--norm-stats", type=str, default="data/processed/channel_normalization_stats_ocean_only.json")
    parser.add_argument("--output", type=str, default="logs/phase3_evaluation_results.json")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--skip-continuous", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        eta=args.eta,
        eval_continuous=not args.skip_continuous,
        device_str=args.device,
        norm_stats_path=args.norm_stats,
        output_path=args.output,
    )
