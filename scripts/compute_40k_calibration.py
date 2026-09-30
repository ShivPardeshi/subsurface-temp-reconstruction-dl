"""Compute real ECE and ensemble spread for Step 40,000 checkpoint at eta=0.0 and eta=0.3.
"""

import sys
import math
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
from src.utils.grid import CANONICAL_DEPTHS
from src.evaluation.metrics.calibration import evaluate_ensemble_calibration


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running calibration on: {device}")

    ckpt_path = "checkpoints/phase3_retrain_normalized/step40000_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    context_encoder = ContextEncoder(
        in_channels=25,
        hidden_dims=[32, 64, 64],
        normalize_inputs=True,
        norm_stats_path="data/processed/channel_normalization_stats.json",
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

    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    lat = in_zarr["lat"][:]
    lon = in_zarr["lon"][:]
    ocean_mask = (in_zarr["inputs"][0, 20] > 0.5)
    clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)

    static_channels = [20, 19, 21, 22, 23, 24]
    t_day = 358  # Test date Dec 25, 2025

    doy = int(scalar_df.loc[t_day, "day_of_year"])
    omega = 2.0 * math.pi / 365.25
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

    seq_slice = in_zarr["inputs"][t_day - 6 : t_day + 1]
    seq_slice = np.nan_to_num(seq_slice, nan=0.0)
    x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)
    static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
    scalar_cond = torch.tensor(
        [[float(scalar_df.loc[t_day, "sin_doy"]), float(scalar_df.loc[t_day, "cos_doy"]), float(scalar_df.loc[t_day, "oni_index"]), float(scalar_df.loc[t_day, "iod_dmi_index"])]],
        device=device,
        dtype=torch.float32,
    )

    for eta in [0.0, 0.3]:
        print(f"\n--- Testing DDIM eta={eta} (N=5) ---")
        ddim = DDIMSampler(diffusion, num_ddim_timesteps=10, eta=eta)
        cascade = DepthCascadeSampler(context_encoder, unet, ddim, CANONICAL_DEPTHS)

        n_ens = 5
        ens_preds = []
        for ens_i in range(n_ens):
            torch.manual_seed(100 + ens_i)
            with torch.no_grad():
                out_ens = cascade.sample_full_profile(x_seq, static_feats, scalar_cond, use_cascade=True)
                p_anom = out_ens["anomalies"][0].cpu().numpy()
                p_temp = p_anom + clim_t
                ens_preds.append(p_temp)

        ens_arr = np.array(ens_preds)
        print(f"ens_arr shape: {ens_arr.shape}, any NaN in ens: {np.isnan(ens_arr).any()}, any NaN in true: {np.isnan(true_temp).any()}")

        std_map = np.std(ens_arr, axis=0)
        spread_ocean = float(np.mean(std_map[:, ocean_mask]))
        print(f"Physical Ensemble Spread (Ocean): {spread_ocean:.4f} °C")

        # Evaluate calibration
        cal_res = evaluate_ensemble_calibration(ens_arr, true_temp, mask=ocean_mask)
        print(f"Expected Calibration Error (ECE): {cal_res['expected_calibration_error']:.4f}")
        print(f"Calibration Diagnostic: {cal_res['calibration_diagnostics']}")
        print(f"Coverage by Level: {cal_res.get('coverage_by_level', {})}")


if __name__ == "__main__":
    main()
