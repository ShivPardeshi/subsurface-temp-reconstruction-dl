"""Evaluate Post-Hoc Uncertainty Calibration for Clean Scratch 40k Model."""

import sys
import os
import math
import json
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.diffusion import GaussianDiffusion
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS
from src.evaluation.metrics.calibration import evaluate_ensemble_calibration
from src.evaluation.calibration_scaling import PostHocCalibrator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = "checkpoints/phase3_retrain_scratch_40k_full/best_checkpoint.pt"
    norm_stats_path = "data/processed/channel_normalization_stats_ocean_only.json"

    ckpt = torch.load(ckpt_path, map_location=device)
    
    enc = ContextEncoder(
        in_channels=25,
        hidden_dims=[32, 64, 64],
        normalize_inputs=True,
        norm_stats_path=norm_stats_path,
    ).to(device)
    enc.load_state_dict(ckpt["models"]["context_encoder"])
    enc.eval()

    unet = UNetDenoiser(
        in_channels=72,
        stage_channels=[32, 64, 128, 256],
        cond_in_dim=8,
    ).to(device)
    unet.load_state_dict(ckpt["models"]["unet"])
    unet.eval()

    diff = GaussianDiffusion(timesteps=1000, schedule_type="cosine").to(device)

    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    ocean_mask_np = (in_zarr["inputs"][0, 20] > 0.5)
    clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)
    static_channels = [20, 19, 21, 22, 23, 24]
    omega = 2.0 * math.pi / 365.25

    target_dates = [60, 150, 240, 318, 358]
    num_ens = 5

    all_ens_preds = []  # (N_ens, N_dates, D, H, W)
    all_targets = []    # (N_dates, D, H, W)

    with torch.no_grad():
        for t_day in target_dates:
            doy = int(scalar_df.loc[t_day, "day_of_year"])
            sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
            cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
            oni = float(scalar_df.loc[t_day, "oni_index"])
            iod = float(scalar_df.loc[t_day, "iod_dmi_index"])

            seq_slice = in_zarr["inputs"][t_day - 6 : t_day + 1]
            seq_slice = np.nan_to_num(seq_slice, nan=0.0)
            x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)
            static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
            scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], device=device, dtype=torch.float32)

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
            all_targets.append(true_temp)

            date_ens = []
            for ens_i in range(num_ens):
                torch.manual_seed(42 + ens_i * 100)
                ddim = DDIMSampler(diffusion=diff, num_ddim_timesteps=10, eta=1.0)
                cascade = DepthCascadeSampler(
                    context_encoder=enc,
                    unet_denoiser=unet,
                    ddim_sampler=ddim,
                    depths=CANONICAL_DEPTHS,
                )
                out = cascade.sample_full_profile(
                    x_seq=x_seq,
                    static_features=static_feats,
                    scalar_conditions=scalar_cond,
                    use_cascade=True,
                )
                pred_anom = out["anomalies"][0].cpu().numpy()
                pred_temp = pred_anom + clim_t
                date_ens.append(pred_temp)
            all_ens_preds.append(date_ens)
            print(f"  Processed Day {t_day} across {num_ens} ensemble runs.")

    all_ens_preds = np.array(all_ens_preds)  # (N_dates, N_ens, D, H, W)
    all_ens_preds = np.swapaxes(all_ens_preds, 0, 1)  # (N_ens, N_dates, D, H, W)
    all_targets = np.array(all_targets)  # (N_dates, D, H, W)

    print(f"Generated Ensemble Shape: {all_ens_preds.shape}")
    print(f"Target Shape: {all_targets.shape}")

    # 1. Evaluate Raw Calibration
    raw_eval = evaluate_ensemble_calibration(
        ensemble_preds=all_ens_preds,
        target=all_targets,
        confidence_levels=[0.50, 0.68, 0.80, 0.90, 0.95],
        mask=ocean_mask_np,
    )
    print("\n--- RAW UNCALIBRATED ENSEMBLE ---")
    print(f"Expected Calibration Error (ECE): {raw_eval['expected_calibration_error']:.4f}")
    print(f"Coverage by Level: {raw_eval['coverage_by_level']}")
    print(f"Diagnostics: {raw_eval['calibration_diagnostics']}")

    # 2. Fit Post-Hoc Calibrator
    calibrator = PostHocCalibrator(canonical_depths=CANONICAL_DEPTHS)
    fit_info = calibrator.fit(
        ensemble_preds=all_ens_preds,
        targets=all_targets,
        mask=ocean_mask_np,
    )
    print("\n--- POST-HOC CALIBRATION FIT ---")
    print(f"Depth-wise scaling factors s(z): {np.round(calibrator.s_factors, 3).tolist()}")
    print(f"Residual variance sigma_res(z): {np.round(calibrator.sigma_res, 3).tolist()}")

    # 3. Apply Calibrator
    calibrated_ens = calibrator.calibrate_ensemble(all_ens_preds)

    # 4. Evaluate Calibrated Calibration
    cal_eval = evaluate_ensemble_calibration(
        ensemble_preds=calibrated_ens,
        target=all_targets,
        confidence_levels=[0.50, 0.68, 0.80, 0.90, 0.95],
        mask=ocean_mask_np,
    )
    print("\n--- POST-HOC CALIBRATED ENSEMBLE ---")
    print(f"Calibrated ECE: {cal_eval['expected_calibration_error']:.4f}")
    print(f"Calibrated Coverage by Level: {cal_eval['coverage_by_level']}")
    print(f"Calibrated Diagnostics: {cal_eval['calibration_diagnostics']}")

    out_file = Path("reports/calibration_scaling_scratch_40k_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "raw_calibration": raw_eval,
            "fitted_calibration_parameters": fit_info,
            "calibrated_calibration": cal_eval,
        }, f, indent=2)
    print(f"\n[SAVED] Results saved to {out_file}")


if __name__ == "__main__":
    main()
