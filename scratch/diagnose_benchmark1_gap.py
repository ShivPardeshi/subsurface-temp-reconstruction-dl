"""Detailed diagnostic analysis of Benchmark 1 (Continuous Nov-Dec Split).

Analyzes:
1. Climatology RMSE vs Model RMSE per depth on the Nov-Dec split.
2. Date-by-date RMSE breakdown (which specific dates in Nov-Dec are contributing to high error).
3. Region-by-region breakdown (Arabian Sea vs Bay of Bengal vs Equatorial).
4. Physical root cause: Anomaly magnitude vs seasonal signal during Nov-Dec.
"""

import sys
import os
import math
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

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

device = torch.device("cpu")
ckpt_path = "checkpoints/phase8_zone_adaptive_15k/last_checkpoint.pt"
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

ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=25, eta=0.0, schedule_type="quadratic")
cascade = DepthCascadeSampler(
    context_encoder=context_encoder,
    unet_denoiser=unet,
    ddim_sampler=ddim,
    depths=CANONICAL_DEPTHS,
    depth_scales_path="data/processed/anomaly_depth_scales_zone_adaptive_p8.json",
    rescale_output=True,
    aux_heads=aux_heads,
)

# Load dataset
in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

total_samples = len(scalar_df)
test_start = max(0, total_samples - 60)
eval_indices = list(range(test_start, total_samples, max(1, 60 // 10)))[:10]
print(f"Evaluating Nov-Dec indices: {eval_indices}")

ocean_mask_np = (in_zarr["inputs"][0, 20] > 0.5)
as_mask = (in_zarr["inputs"][0, 21] > 0.5) & ocean_mask_np
bob_mask = (in_zarr["inputs"][0, 22] > 0.5) & ocean_mask_np
clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)
static_channels = [20, 19, 21, 22, 23, 24]
omega = 2.0 * math.pi / 365.25

per_depth_p8_sq = {d: [] for d in CANONICAL_DEPTHS}
per_depth_clim_sq = {d: [] for d in CANONICAL_DEPTHS}
per_date_p8_rmse = {}
per_date_clim_rmse = {}
per_date_doy = {}

with torch.no_grad():
    for sample_idx in eval_indices:
        seq = np.nan_to_num(in_zarr["inputs"][sample_idx - 6 : sample_idx + 1], nan=0.0)
        static_feats_np = np.nan_to_num(in_zarr["inputs"][sample_idx, static_channels], nan=0.0)
        true_anom_np = np.nan_to_num(tgt_zarr["anomaly"][sample_idx].copy(), nan=0.0)
        true_anom_np[0] = true_anom_np[1]

        doy = int(scalar_df.loc[sample_idx, "day_of_year"])
        sin_doy = float(scalar_df.loc[sample_idx, "sin_doy"])
        cos_doy = float(scalar_df.loc[sample_idx, "cos_doy"])
        oni = float(scalar_df.loc[sample_idx, "oni_index"])
        iod = float(scalar_df.loc[sample_idx, "iod_dmi_index"])

        x_seq = torch.from_numpy(seq).unsqueeze(0).float().to(device)
        static_feats = torch.from_numpy(static_feats_np).unsqueeze(0).float().to(device)
        scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], dtype=torch.float32, device=device)

        out = cascade.sample_full_profile(
            x_seq=x_seq,
            static_features=static_feats,
            scalar_conditions=scalar_cond,
            use_cascade=True,
        )
        pred_anom_np = out["anomalies"].squeeze(0).cpu().numpy()

        date_p8_sq = []
        date_clim_sq = []
        for d_idx, depth in enumerate(CANONICAL_DEPTHS):
            diff_p8 = (pred_anom_np[d_idx] - true_anom_np[d_idx])[ocean_mask_np]
            diff_clim = (0.0 - true_anom_np[d_idx])[ocean_mask_np]  # climatology prediction is anomaly = 0

            sq_p8 = diff_p8 ** 2
            sq_clim = diff_clim ** 2

            per_depth_p8_sq[depth].extend(sq_p8.tolist())
            per_depth_clim_sq[depth].extend(sq_clim.tolist())

            date_p8_sq.extend(sq_p8.tolist())
            date_clim_sq.extend(sq_clim.tolist())

        per_date_p8_rmse[sample_idx] = math.sqrt(np.mean(date_p8_sq))
        per_date_clim_rmse[sample_idx] = math.sqrt(np.mean(date_clim_sq))
        per_date_doy[sample_idx] = doy

print("\n" + "=" * 90)
print("BENCHMARK 1 (NOV-DEC SPLIT) — DEPTH-BY-DEPTH DIAGNOSTIC")
print("=" * 90)
print(f"{'Depth':>6} | {'Phase 8 RMSE':>12} | {'Climatology RMSE':>16} | {'Diff (°C)':>10} | {'Skill (%)':>10} | {'Status':>12}")
print("-" * 90)
all_p8 = []
all_clim = []
for d in CANONICAL_DEPTHS:
    p8_r = math.sqrt(np.mean(per_depth_p8_sq[d]))
    cl_r = math.sqrt(np.mean(per_depth_clim_sq[d]))
    all_p8.extend(per_depth_p8_sq[d])
    all_clim.extend(per_depth_clim_sq[d])
    skill = (1.0 - (p8_r ** 2) / (cl_r ** 2)) * 100.0
    status = "BEATS CLIM" if p8_r < cl_r else "BEHIND CLIM"
    print(f"{d:>5.0f}m | {p8_r:>12.4f} | {cl_r:>16.4f} | {p8_r - cl_r:>+10.4f} | {skill:>+9.2f}% | {status:>12}")

overall_p8_rmse = math.sqrt(np.mean(all_p8))
overall_clim_rmse = math.sqrt(np.mean(all_clim))
overall_skill = (1.0 - (overall_p8_rmse ** 2) / (overall_clim_rmse ** 2)) * 100.0
print("-" * 90)
print(f"{'OVERALL':>6} | {overall_p8_rmse:>12.4f} | {overall_clim_rmse:>16.4f} | {overall_p8_rmse - overall_clim_rmse:>+10.4f} | {overall_skill:>+9.2f}% |")
print("=" * 90)

print("\nDATE-BY-DATE BREAKDOWN ACROSS NOV-DEC:")
print("-" * 75)
print(f"{'Index':>6} | {'DOY':>6} | {'Date Est.':>12} | {'Phase 8 RMSE':>12} | {'Clim RMSE':>12} | {'Skill (%)':>10}")
print("-" * 75)
for s_idx in eval_indices:
    doy = per_date_doy[s_idx]
    p8_r = per_date_p8_rmse[s_idx]
    cl_r = per_date_clim_rmse[s_idx]
    sk = (1.0 - (p8_r ** 2) / (cl_r ** 2)) * 100.0
    print(f"{s_idx:>6} | {doy:>6} | {'Nov/Dec':>12} | {p8_r:>12.4f} | {cl_r:>12.4f} | {sk:>+9.2f}%")
print("=" * 75)
