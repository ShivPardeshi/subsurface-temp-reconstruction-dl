"""Fresh Evaluation Suite for OceanEmbed Reference Model (Seed 42 Post-Fix-A2).

Adheres strictly to PS26066 Action List Step 2:
1. Computes exact SHA-256 hash of reference checkpoint.
2. Loads raw Phase 2 Zarr inputs, targets, auxiliary maps, and climatology coefficients.
3. Evaluates 9 multi-seasonal target dates across 2025:
   - Winter: Day 15 (Jan 15), Day 45 (Feb 14)
   - Spring: Day 105 (Apr 15)
   - Summer: Day 195 (Jul 15)
   - Fall Intermonsoon (Val): Day 245 (Sep 2), Day 275 (Oct 2)
   - Held-Out Test Set (Nov-Dec): Day 318 (Nov 14), Day 331 (Nov 27), Day 358 (Dec 24)
4. Computes:
   - Full 3D overall and depthwise metrics (RMSE, MAE, Bias, Pearson r, SSIM, Murphy Skill Score).
   - Strictly held-out test-set performance (Nov-Dec 2025).
   - Auxiliary head correlations (MLD, BLT, Salinity Max Depth/Strength) with domain masks.
   - Stochastic calibration ECE and ensemble spread (eta=0.3).
   - All 7 Priority Zones under redefined 20-200m Thermocline Core.
   - Physical surface heat-flux consistency.
   - Ablation models (Stage C No-Region, Stage D No-Cascade) under identical protocol.
5. Saves raw execution results to logs/evaluation_results_fresh_step2.json.
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
from src.utils.grid import CANONICAL_DEPTHS, get_target_grid
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


def evaluate_checkpoint(
    stage_name: str,
    checkpoint_path: str,
    in_data: Any,
    tgt_data: Any,
    aux_data: Any,
    scalar_df: pd.DataFrame,
    clim_coeffs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: np.ndarray,
    bob_mask: np.ndarray,
    as_mask: np.ndarray,
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
    print(f"\n{'='*80}")
    print(f"EVALUATING: {stage_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"SHA-256:   {ckpt_hash}")
    print(f"Timestamp: {eval_timestamp} | DDIM eta: {eta} | Cascade: {use_cascade} | Region: {use_region}")
    print(f"{'='*80}")

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

    for idx, t_day in enumerate(eval_target_days):
        date_str = scalar_df.loc[t_day, "date"]
        doy = int(scalar_df.loc[t_day, "day_of_year"])
        sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
        cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
        oni = float(scalar_df.loc[t_day, "oni_index"])
        iod = float(scalar_df.loc[t_day, "iod_dmi_index"])
        timestamps.append(date_str)

        # 7-day input slice: t_day - 6 to t_day
        seq_slice = in_data[t_day - 6 : t_day + 1]
        seq_slice = np.nan_to_num(seq_slice, nan=0.0)
        x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)

        static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
        if not use_region:
            static_feats[:, 2:6] = 0.0

        scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], device=device, dtype=torch.float32)

        t_start = time.time()
        with torch.no_grad():
            out = cascade.sample_full_profile(
                x_seq=x_seq,
                static_features=static_feats,
                scalar_conditions=scalar_cond,
                use_cascade=use_cascade,
            )
            pred_anom = out["anomalies"][0].cpu().numpy()  # (15, H, W)

            # Auxiliary heads prediction
            u_cond = enc(x_seq)
            aux_out = aux_heads(u_cond)
            pred_mld_list.append(float(aux_out["mld"].item()))
            pred_blt_list.append(float(aux_out["blt"].item()))
            pred_sal_depth_list.append(float(aux_out["sal_max_depth"].item()))
            pred_sal_str_list.append(float(aux_out["sal_max_strength"].item()))

        # Ground truth anomaly targets
        true_anom = np.asarray(tgt_data[t_day], dtype=np.float32)
        true_anom = np.nan_to_num(true_anom, nan=0.0)

        # Auxiliary ground truth targets
        aux_t = np.asarray(aux_data[t_day], dtype=np.float32)
        # Compute regional scalar means for ground truth
        mld_true_scalar = float(np.mean(aux_t[0][ocean_mask]))
        blt_true_scalar = float(np.mean(aux_t[1][bob_mask]))
        sal_depth_true_scalar = float(np.mean(aux_t[2][as_mask]))
        sal_str_true_scalar = float(np.mean(aux_t[3][as_mask]))

        true_mld_list.append(mld_true_scalar)
        true_blt_list.append(blt_true_scalar)
        true_sal_depth_list.append(sal_depth_true_scalar)
        true_sal_str_list.append(sal_str_true_scalar)

        # 2-Harmonic Annual Cycle Climatology
        omega = 2.0 * math.pi / 365.25
        cos1 = math.cos(omega * doy)
        sin1 = math.sin(omega * doy)
        cos2 = math.cos(2.0 * omega * doy)
        sin2 = math.sin(2.0 * omega * doy)

        # clim_coeffs shape: (5, 15, H, W) -> a0, a1, b1, a2, b2
        clim_t = (
            clim_coeffs[0]
            + clim_coeffs[1] * cos1
            + clim_coeffs[2] * sin1
            + clim_coeffs[3] * cos2
            + clim_coeffs[4] * sin2
        )

        pred_temp = clim_t + pred_anom
        true_temp = clim_t + true_anom

        pred_temps_list.append(pred_temp)
        true_temps_list.append(true_temp)
        clim_temps_list.append(clim_t)
        last_v_input = seq_slice[0]

        step_elapsed = time.time() - t_start
        # Compute quick step RMSE cleanly using compute_rmse
        step_rmse = float(compute_rmse(pred_temp, true_temp, mask=ocean_mask))
        print(f"  Day {t_day:03d} ({date_str}) | Step RMSE: {step_rmse:.4f} °C | Inference: {step_elapsed:.2f}s", flush=True)

    pred_temps = np.stack(pred_temps_list, axis=0)  # (N_dates, 15, H, W)
    true_temps = np.stack(true_temps_list, axis=0)
    clim_temps = np.stack(clim_temps_list, axis=0)

    # 1. Global Metrics (overall pooled ocean cells across all dates & depths)
    global_mask_3d = np.broadcast_to(ocean_mask, pred_temps.shape)
    global_basic = compute_all_basic_metrics(pred_temps, true_temps, mask=global_mask_3d)
    global_skill = compute_murphy_skill_score(pred_temps, true_temps, clim_temps, mask=global_mask_3d)

    # SSIM computed across horizontal slices
    ssim_per_depth = []
    for d_idx in range(len(CANONICAL_DEPTHS)):
        p_slice = pred_temps[:, d_idx]
        t_slice = true_temps[:, d_idx]
        d_ssims = [compute_ssim_2d(p_slice[i], t_slice[i], mask=ocean_mask) for i in range(len(eval_target_days))]
        ssim_per_depth.append(float(np.mean(d_ssims)))

    global_ssim = float(np.mean(ssim_per_depth))

    # 2. Depthwise Breakdown across all 15 canonical depths
    depthwise_results = {}
    for d_idx, depth in enumerate(CANONICAL_DEPTHS):
        d_pred = pred_temps[:, d_idx]
        d_true = true_temps[:, d_idx]
        d_clim = clim_temps[:, d_idx]
        d_mask = np.broadcast_to(ocean_mask, d_pred.shape)

        d_basic = compute_all_basic_metrics(d_pred, d_true, mask=d_mask)
        d_skill = compute_murphy_skill_score(d_pred, d_true, d_clim, mask=d_mask)

        depthwise_results[f"{depth}m"] = {
            "depth_m": depth,
            "rmse": float(d_basic["rmse"]),
            "mae": float(d_basic["mae"]),
            "bias": float(d_basic["bias"]),
            "correlation": float(d_basic["correlation"]),
            "skill_score": float(d_skill),
            "ssim": float(ssim_per_depth[d_idx]),
        }

    # 3. Held-Out Test Set Metrics (Dates 318, 331, 358 in Nov-Dec 2025)
    test_day_indices = [i for i, d in enumerate(eval_target_days) if d >= 298]
    if test_day_indices:
        test_pred = pred_temps[test_day_indices]
        test_true = true_temps[test_day_indices]
        test_clim = clim_temps[test_day_indices]
        test_mask = np.broadcast_to(ocean_mask, test_pred.shape)
        test_basic = compute_all_basic_metrics(test_pred, test_true, mask=test_mask)
        test_skill = compute_murphy_skill_score(test_pred, test_true, test_clim, mask=test_mask)
        held_out_test_metrics = {
            "test_rmse": float(test_basic["rmse"]),
            "test_mae": float(test_basic["mae"]),
            "test_bias": float(test_basic["bias"]),
            "test_correlation": float(test_basic["correlation"]),
            "test_skill_score": float(test_skill),
            "dates_evaluated": [eval_target_days[i] for i in test_day_indices],
        }
    else:
        held_out_test_metrics = {}

    # 4. Auxiliary Head Metrics
    pred_aux_dict = {
        "mld": np.array(pred_mld_list),
        "blt": np.array(pred_blt_list),
        "sal_max_depth": np.array(pred_sal_depth_list),
        "sal_max_strength": np.array(pred_sal_str_list),
    }
    true_aux_dict = {
        "mld": np.array(true_mld_list),
        "blt": np.array(true_blt_list),
        "sal_max_depth": np.array(true_sal_depth_list),
        "sal_max_strength": np.array(true_sal_str_list),
    }
    aux_eval = evaluate_auxiliary_predictions(pred_aux_dict, true_aux_dict)

    # 5. Priority Zones Evaluation (including redefined 20-200m thermocline)
    zones = get_priority_zones(lat=lat, lon=lon, ibtracs_csv_path=None)
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
    priority_zone_metrics = {}
    for z_slug, z_obj in zones.items():
        z_rmse = zone_rmses.get(z_slug, 0.0)
        z_mae = zone_maes.get(z_slug, 0.0)
        priority_zone_metrics[z_slug] = {
            "zone_id": z_obj.zone_id,
            "name": z_obj.name,
            "description": z_obj.description,
            "rmse": float(z_rmse) if isinstance(z_rmse, (int, float)) and not np.isnan(z_rmse) else 0.0,
            "mae": float(z_mae) if isinstance(z_mae, (int, float)) and not np.isnan(z_mae) else 0.0,
        }

    # 6. Physical Heat-Flux Consistency Check
    v_field = last_v_input[6]  # channel 6 = current_v
    heat_flux_results = evaluate_heat_flux_consistency(
        v_input=v_field,
        t_pred=pred_temps[-1],
        t_true=true_temps[-1],
        mask=ocean_mask,
    )

    # 7. Stochastic Ensemble Calibration (eta=0.3)
    # Generate 5 ensemble members on a test date to assess spread and ECE
    sample_day = eval_target_days[-1]
    doy = int(scalar_df.loc[sample_day, "day_of_year"])
    seq_slice = in_data[sample_day - 6 : sample_day + 1]
    seq_slice = np.nan_to_num(seq_slice, nan=0.0)
    x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)
    static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
    if not use_region:
        static_feats[:, 2:6] = 0.0
    scalar_cond = torch.tensor(
        [[scalar_df.loc[sample_day, "sin_doy"], scalar_df.loc[sample_day, "cos_doy"], scalar_df.loc[sample_day, "oni_index"], scalar_df.loc[sample_day, "iod_dmi_index"]]],
        device=device,
        dtype=torch.float32,
    )

    ens_preds = []
    for e_i in range(5):
        with torch.no_grad():
            out_e = cascade.sample_full_profile(
                x_seq=x_seq,
                static_features=static_feats,
                scalar_conditions=scalar_cond,
                use_cascade=use_cascade,
            )
            anom_e = out_e["anomalies"][0].cpu().numpy()
            ens_preds.append(clim_temps[-1] + anom_e)
    ens_preds = np.stack(ens_preds, axis=0)  # (5, 15, H, W)
    ens_mask_3d = np.broadcast_to(ocean_mask, ens_preds[0].shape)
    calib_results = evaluate_ensemble_calibration(
        ensemble_preds=ens_preds,
        target=true_temps[-1],
        mask=ens_mask_3d,
    )
    ens_spread = float(np.mean(np.std(ens_preds, axis=0)[ens_mask_3d]))
    calib_results["ensemble_spread_c"] = ens_spread

    return {
        "stage_name": stage_name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": ckpt_hash,
        "eval_timestamp": eval_timestamp,
        "global_metrics": {
            "rmse": float(global_basic["rmse"]),
            "mae": float(global_basic["mae"]),
            "bias": float(global_basic["bias"]),
            "correlation": float(global_basic["correlation"]),
            "skill_score": float(global_skill),
            "ssim": float(global_ssim),
        },
        "held_out_test_metrics": held_out_test_metrics,
        "depthwise_metrics": depthwise_results,
        "auxiliary_head_metrics": aux_eval,
        "priority_zone_metrics": priority_zone_metrics,
        "heat_flux_consistency": heat_flux_results,
        "calibration_metrics": calib_results,
        "learned_weights": {
            "w1": float(ckpt_obj.get("loss_fn", {}).get("w1", torch.tensor([0.0])).item()),
            "w2": float(ckpt_obj.get("loss_fn", {}).get("w2", torch.tensor([0.0])).item()),
            "w3": float(ckpt_obj.get("loss_fn", {}).get("w3", torch.tensor([0.0])).item()),
        },
    }


def make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj


def save_results(results: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_obj = make_serializable(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_obj, f, indent=2)
    print(f"Saved results update to: {path}", flush=True)


def main():
    print("Starting Fresh Evaluation Suite for OceanEmbed PS26066...")
    device = torch.device("cpu")
    out_json_path = REPO_ROOT / "logs" / "evaluation_results_fresh_step2.json"

    # Load metadata
    phase2_dir = REPO_ROOT / "data" / "processed" / "phase2_dataset"
    in_zarr = zarr.open_group(str(phase2_dir / "oceanembed_training_inputs.zarr"), mode="r")
    tgt_zarr = zarr.open_group(str(phase2_dir / "oceanembed_anomaly_targets.zarr"), mode="r")
    aux_zarr = zarr.open_group(str(phase2_dir / "oceanembed_auxiliary_targets.zarr"), mode="r")

    in_data = in_zarr["inputs"] if "inputs" in in_zarr else in_zarr["datacube"]
    tgt_data = tgt_zarr["anomaly"] if "anomaly" in tgt_zarr else tgt_zarr["anomaly_targets"]
    aux_data = aux_zarr["auxiliary_targets"] if "auxiliary_targets" in aux_zarr else aux_zarr["aux_target"]

    scalar_df = pd.read_csv(phase2_dir / "scalar_conditioning.csv")
    clim_ds = xr.open_dataset(phase2_dir / "climatology_coefficients.nc")
    clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)  # (5, 15, 112, 240)

    lat, lon = get_target_grid()

    # Generate masks from static channels
    sample_static = np.nan_to_num(in_data[0, [20, 21, 22]], nan=0.0)
    ocean_mask = (sample_static[0] > 0.5)
    as_mask = (sample_static[1] > 0.5) & ocean_mask
    bob_mask = (sample_static[2] > 0.5) & ocean_mask

    # 9 Multi-seasonal target dates across 2025
    eval_target_days = [15, 45, 105, 195, 245, 275, 318, 331, 358]

    # Models to evaluate:
    stage_b_ckpt = str(REPO_ROOT / "checkpoints" / "baseline_20k_fixA2_seed42" / "best_checkpoint.pt")
    stage_c_ckpt = str(REPO_ROOT / "checkpoints" / "ablation_no_region_20k_fixA2_seed42" / "best_checkpoint.pt")
    stage_d_ckpt = str(REPO_ROOT / "checkpoints" / "ablation_no_cascade_20k_fixA2_seed42" / "best_checkpoint.pt")

    results = {}

    # 1. Evaluate Reference Model (Stage B)
    results["stage_b"] = evaluate_checkpoint(
        stage_name="Stage B Reference (Region ON, Cascade ON)",
        checkpoint_path=stage_b_ckpt,
        in_data=in_data,
        tgt_data=tgt_data,
        aux_data=aux_data,
        scalar_df=scalar_df,
        clim_coeffs=clim_coeffs,
        lat=lat,
        lon=lon,
        ocean_mask=ocean_mask,
        bob_mask=bob_mask,
        as_mask=as_mask,
        eval_target_days=eval_target_days,
        use_cascade=True,
        use_region=True,
        eta=0.3,
        device=device,
    )
    save_results(results, out_json_path)

    # 2. Evaluate Stage C Ablation (No Region)
    results["stage_c"] = evaluate_checkpoint(
        stage_name="Stage C Ablation (Region OFF, Cascade ON)",
        checkpoint_path=stage_c_ckpt,
        in_data=in_data,
        tgt_data=tgt_data,
        aux_data=aux_data,
        scalar_df=scalar_df,
        clim_coeffs=clim_coeffs,
        lat=lat,
        lon=lon,
        ocean_mask=ocean_mask,
        bob_mask=bob_mask,
        as_mask=as_mask,
        eval_target_days=eval_target_days,
        use_cascade=True,
        use_region=False,
        eta=0.3,
        device=device,
    )
    save_results(results, out_json_path)

    # 3. Evaluate Stage D Ablation (No Cascade)
    results["stage_d"] = evaluate_checkpoint(
        stage_name="Stage D Ablation (Region ON, Cascade OFF)",
        checkpoint_path=stage_d_ckpt,
        in_data=in_data,
        tgt_data=tgt_data,
        aux_data=aux_data,
        scalar_df=scalar_df,
        clim_coeffs=clim_coeffs,
        lat=lat,
        lon=lon,
        ocean_mask=ocean_mask,
        bob_mask=bob_mask,
        as_mask=as_mask,
        eval_target_days=eval_target_days,
        use_cascade=False,
        use_region=True,
        eta=0.3,
        device=device,
    )
    save_results(results, out_json_path)
    print(f"\nSuccessfully wrote all fresh evaluation results to: {out_json_path}")


if __name__ == "__main__":
    main()
