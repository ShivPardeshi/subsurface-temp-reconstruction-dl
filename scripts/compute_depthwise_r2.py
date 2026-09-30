"""Compute Depthwise R^2 values for OceanEmbed Reference Model.

Computes:
1. Statistical R^2 (Coefficient of Determination vs spatial mean):
   R^2_stat = 1 - sum((T_true - T_pred)^2) / sum((T_true - mean(T_true))^2)
2. Squared Pearson Correlation (r^2):
   r^2 = (corr(T_pred, T_true))^2
3. Climatology Skill R^2 (Murphy Skill Score vs annual cycle climatology):
   R^2_clim = 1 - sum((T_true - T_pred)^2) / sum((T_true - T_clim)^2)

Evaluated across all 15 canonical depths for:
checkpoints/baseline_20k_fixA2_seed42/best_checkpoint.pt
"""

import sys
import math
import time
from pathlib import Path

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
from src.evaluation.metrics.basic_metrics import compute_rmse, compute_mae, compute_correlation, compute_r2, _apply_mask


def main():
    print("Computing Depthwise R^2 values for OceanEmbed Reference Model...")
    device = torch.device("cpu")

    ckpt_path = REPO_ROOT / "checkpoints" / "baseline_20k_fixA2_seed42" / "best_checkpoint.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    enc = ContextEncoder(in_channels=25, hidden_dims=[32, 64, 64]).to(device)
    unet = UNetDenoiser(in_channels=72, stage_channels=[32, 64, 128, 256], cond_in_dim=8).to(device)
    diff = GaussianDiffusion(timesteps=1000, schedule_type="cosine").to(device)

    enc.load_state_dict(ckpt["models"]["context_encoder"])
    unet.load_state_dict(ckpt["models"]["unet"])
    enc.eval()
    unet.eval()

    ddim = DDIMSampler(diff, num_ddim_timesteps=10, eta=0.3)
    cascade = DepthCascadeSampler(enc, unet, ddim, CANONICAL_DEPTHS)

    # Load dataset
    phase2_dir = REPO_ROOT / "data" / "processed" / "phase2_dataset"
    in_zarr = zarr.open_group(str(phase2_dir / "oceanembed_training_inputs.zarr"), mode="r")
    tgt_zarr = zarr.open_group(str(phase2_dir / "oceanembed_anomaly_targets.zarr"), mode="r")

    in_data = in_zarr["inputs"] if "inputs" in in_zarr else in_zarr["datacube"]
    tgt_data = tgt_zarr["anomaly"] if "anomaly" in tgt_zarr else tgt_zarr["anomaly_targets"]

    scalar_df = pd.read_csv(phase2_dir / "scalar_conditioning.csv")
    clim_ds = xr.open_dataset(phase2_dir / "climatology_coefficients.nc")
    clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)

    lat, lon = get_target_grid()
    sample_static = np.nan_to_num(in_data[0, [20, 21, 22]], nan=0.0)
    ocean_mask = (sample_static[0] > 0.5)

    eval_target_days = [15, 45, 105, 195, 245, 275, 318, 331, 358]
    static_channels = [20, 19, 21, 22, 23, 24]

    pred_temps_list = []
    true_temps_list = []
    clim_temps_list = []

    for t_day in eval_target_days:
        date_str = scalar_df.loc[t_day, "date"]
        doy = int(scalar_df.loc[t_day, "day_of_year"])
        sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
        cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
        oni = float(scalar_df.loc[t_day, "oni_index"])
        iod = float(scalar_df.loc[t_day, "iod_dmi_index"])

        seq_slice = in_data[t_day - 6 : t_day + 1]
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

        true_anom = np.asarray(tgt_data[t_day], dtype=np.float32)
        true_anom = np.nan_to_num(true_anom, nan=0.0)

        # Climatology
        omega = 2.0 * math.pi / 365.25
        cos1 = math.cos(omega * doy)
        sin1 = math.sin(omega * doy)
        cos2 = math.cos(2.0 * omega * doy)
        sin2 = math.sin(2.0 * omega * doy)

        clim_t = (
            clim_coeffs[0]
            + clim_coeffs[1] * cos1
            + clim_coeffs[2] * sin1
            + clim_coeffs[3] * cos2
            + clim_coeffs[4] * sin2
        )

        pred_temps_list.append(clim_t + pred_anom)
        true_temps_list.append(clim_t + true_anom)
        clim_temps_list.append(clim_t)
        print(f"Processed day {t_day:03d} ({date_str})", flush=True)

    pred_temps = np.stack(pred_temps_list, axis=0)  # (9, 15, H, W)
    true_temps = np.stack(true_temps_list, axis=0)
    clim_temps = np.stack(clim_temps_list, axis=0)

    print("\n" + "=" * 88)
    print(f"{'Depth':>8} | {'RMSE (°C)':>10} | {'Pearson r':>10} | {'r^2 (Corr^2)':>12} | {'R^2 (Statistical)':>18} | {'R^2 (vs Climatology)':>20}")
    print("-" * 88)

    results_table = []
    for d_idx, depth in enumerate(CANONICAL_DEPTHS):
        p_d = pred_temps[:, d_idx]
        t_d = true_temps[:, d_idx]
        c_d = clim_temps[:, d_idx]

        mask_d = np.broadcast_to(ocean_mask, p_d.shape)

        p_flat = _apply_mask(p_d, mask_d)
        t_flat = _apply_mask(t_d, mask_d)
        c_flat = _apply_mask(c_d, mask_d)

        valid = np.isfinite(p_flat) & np.isfinite(t_flat) & np.isfinite(c_flat)
        p_val = p_flat[valid]
        t_val = t_flat[valid]
        c_val = c_flat[valid]

        rmse = float(np.sqrt(np.mean((t_val - p_val) ** 2)))
        r = float(np.corrcoef(p_val, t_val)[0, 1])
        r_squared = r ** 2

        # Statistical R^2 = 1 - SS_res / SS_tot (vs mean)
        ss_res = np.sum((t_val - p_val) ** 2)
        ss_tot_mean = np.sum((t_val - np.mean(t_val)) ** 2)
        r2_stat = float(1.0 - (ss_res / ss_tot_mean)) if ss_tot_mean > 0 else 0.0

        # Skill R^2 = 1 - SS_res / SS_tot_clim (vs Climatology)
        ss_tot_clim = np.sum((t_val - c_val) ** 2)
        r2_clim = float(1.0 - (ss_res / ss_tot_clim)) if ss_tot_clim > 0 else 0.0

        print(f"{depth:>6}m | {rmse:10.4f} | {r:10.4f} | {r_squared:12.4f} | {r2_stat:18.4f} | {r2_clim:20.4f}")
        results_table.append({
            "depth_m": depth,
            "rmse": rmse,
            "pearson_r": r,
            "r_squared": r_squared,
            "r2_statistical": r2_stat,
            "r2_climatology": r2_clim,
        })

    # Overall 3D pooled
    global_mask = np.broadcast_to(ocean_mask, pred_temps.shape)
    p_all = _apply_mask(pred_temps, global_mask)
    t_all = _apply_mask(true_temps, global_mask)
    c_all = _apply_mask(clim_temps, global_mask)
    valid_all = np.isfinite(p_all) & np.isfinite(t_all) & np.isfinite(c_all)
    p_a = p_all[valid_all]
    t_a = t_all[valid_all]
    c_a = c_all[valid_all]

    tot_rmse = float(np.sqrt(np.mean((t_a - p_a) ** 2)))
    tot_r = float(np.corrcoef(p_a, t_a)[0, 1])
    tot_r2_corr = tot_r ** 2
    tot_r2_stat = float(1.0 - (np.sum((t_a - p_a) ** 2) / np.sum((t_a - np.mean(t_a)) ** 2)))
    tot_r2_clim = float(1.0 - (np.sum((t_a - p_a) ** 2) / np.sum((t_a - c_a) ** 2)))

    print("-" * 88)
    print(f"{'Overall':>8} | {tot_rmse:10.4f} | {tot_r:10.4f} | {tot_r2_corr:12.4f} | {tot_r2_stat:18.4f} | {tot_r2_clim:20.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
