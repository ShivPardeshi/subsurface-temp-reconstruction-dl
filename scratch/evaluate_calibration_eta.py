import torch
import zarr
import math
import numpy as np
import pandas as pd
import xarray as xr
import sys
sys.path.insert(0, ".")
from scripts.run_genuine_evaluation import load_model
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS
from src.evaluation.metrics.calibration import evaluate_ensemble_calibration

device = torch.device("cpu")
in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

ocean_mask = (in_zarr["inputs"][0, 20] > 0.5)
clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)
static_channels = [20, 19, 21, 22, 23, 24]

# Evaluate on test day 358 (held-out winter date)
t_day = 358
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

omega = 2.0 * math.pi / 365.25
clim_t = (
    clim_coeffs[0]
    + clim_coeffs[1] * math.cos(omega * doy)
    + clim_coeffs[2] * math.sin(omega * doy)
    + clim_coeffs[3] * math.cos(2 * omega * doy)
    + clim_coeffs[4] * math.sin(2 * omega * doy)
)
true_anom = np.nan_to_num(tgt_zarr["anomaly"][t_day].copy(), nan=0.0)
true_temp = true_anom + clim_t

enc, unet, aux, diff = load_model("checkpoints/baseline_20k/best_checkpoint.pt", device)

for eta_val in [0.0, 0.3, 0.5]:
    print(f"\n--- Testing eta = {eta_val} (N=5 ensemble) ---")
    ddim = DDIMSampler(diff, num_ddim_timesteps=10, eta=eta_val)
    cascade = DepthCascadeSampler(enc, unet, ddim, CANONICAL_DEPTHS)
    ens = []
    for i in range(5):
        torch.manual_seed(1000 + i)
        with torch.no_grad():
            out = cascade.sample_full_profile(x_seq, static_feats, scalar_cond, use_cascade=True)
            pred_anom = out["anomalies"][0].cpu().numpy()
            pred_temp = pred_anom + clim_t
            ens.append(pred_temp)
    ens_arr = np.array(ens) # (5, 15, 112, 240)
    std_map = np.std(ens_arr, axis=0) # (15, 112, 240)
    mean_spread = float(np.mean(std_map[:, ocean_mask]))
    print(f"Mean Ensemble Spread (std): {mean_spread:.4f} °C")
    
    cal = evaluate_ensemble_calibration(ens_arr, true_temp, mask=ocean_mask)
    print(f"Expected Calibration Error (ECE): {cal['expected_calibration_error']:.4f}")
    print(f"Diagnostic: {cal['calibration_diagnostics']}")
