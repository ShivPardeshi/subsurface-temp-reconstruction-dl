"""Diagnostic script to check why evaluation returned NaN."""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_genuine_evaluation import load_model, compute_climatology_for_doy
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS
from src.evaluation.metrics.basic_metrics import compute_rmse, compute_all_basic_metrics

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    lat = in_zarr["lat"][:]
    lon = in_zarr["lon"][:]
    ocean_mask = (in_zarr["inputs"][0, 20] > 0.5)
    clim_coeffs = clim_ds["coefficients"].values

    ckpt_path = "checkpoints/baseline/best_checkpoint.pt"
    context_encoder, unet, aux_heads, diffusion = load_model(ckpt_path, device)

    ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=10, eta=0.0)
    cascade = DepthCascadeSampler(
        context_encoder=context_encoder,
        unet_denoiser=unet,
        ddim_sampler=ddim,
        depths=CANONICAL_DEPTHS,
    )

    t_day = 20
    doy = int(scalar_df.loc[t_day, "day_of_year"])
    sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
    cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
    oni = float(scalar_df.loc[t_day, "oni_index"])
    iod = float(scalar_df.loc[t_day, "iod_dmi_index"])

    seq_slice = in_zarr["inputs"][t_day - 6 : t_day + 1]
    seq_slice = np.nan_to_num(seq_slice, nan=0.0)
    x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)
    static_channels = [20, 19, 21, 22, 23, 24]
    static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
    scalar_cond = torch.tensor([[oni, iod, sin_doy, cos_doy]], device=device, dtype=torch.float32)

    with torch.no_grad():
        u_cond = context_encoder(x_seq)
        print(f"u_cond: shape={u_cond.shape}, nans={torch.isnan(u_cond).sum().item()}, min={u_cond.min().item():.3f}, max={u_cond.max().item():.3f}")
        spatial_cond = torch.cat([u_cond, static_feats], dim=1)
        print(f"spatial_cond: shape={spatial_cond.shape}, nans={torch.isnan(spatial_cond).sum().item()}")

        out = cascade.sample_full_profile(
            x_seq=x_seq,
            static_features=static_feats,
            scalar_conditions=scalar_cond,
            use_cascade=True,
        )
    pred_anom = out["anomalies"][0].cpu().numpy()
    print(f"pred_anom: shape={pred_anom.shape}, nans={np.isnan(pred_anom).sum()}, min={np.nanmin(pred_anom):.3f}, max={np.nanmax(pred_anom):.3f}")

    true_anom = tgt_zarr["anomaly"][t_day]
    print(f"true_anom: shape={true_anom.shape}, nans={np.isnan(true_anom).sum()}, min={np.nanmin(true_anom):.3f}, max={np.nanmax(true_anom):.3f}")

    clim_day = compute_climatology_for_doy(clim_coeffs, doy)
    print(f"clim_day: shape={clim_day.shape}, nans={np.isnan(clim_day).sum()}, min={np.nanmin(clim_day):.3f}, max={np.nanmax(clim_day):.3f}")

    pred_temp = pred_anom + clim_day
    true_temp = true_anom + clim_day
    print(f"pred_temp: shape={pred_temp.shape}, nans={np.isnan(pred_temp).sum()}")
    print(f"true_temp: shape={true_temp.shape}, nans={np.isnan(true_temp).sum()}")

    # Check metrics with ocean_mask
    print("Testing compute_all_basic_metrics...")
    bm = compute_all_basic_metrics(pred_temp, true_temp, clim=clim_day, mask=ocean_mask)
    print(f"bm with mask=ocean_mask: {bm}")

if __name__ == "__main__":
    main()
