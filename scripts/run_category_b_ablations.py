"""Category B: Core Ablation Re-Evaluation Script under Fixed Ocean-Only Pipeline.

Evaluates:
1. Full Architecture (Clean Scratch 40k with Region Conditioning + Depth Cascade)
2. Ablation 1: No Region Conditioning (Channels 21-24 zeroed out)
3. Ablation 2: No Depth Cascade (Single-shot diffusion without sequential vertical conditioning)
4. Ablation 3: No Region & No Cascade (Both removed)
"""

import sys
import os
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
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
    compute_correlation,
)
from src.evaluation.metrics.skill_score import compute_murphy_skill_score
from src.evaluation.slicing.priority_zones import get_priority_zones
from src.evaluation.slicing.zone_evaluator import evaluate_metric_by_zone


CANONICAL_MULTI_SEASONAL_DATES = [15, 60, 105, 150, 195, 240, 285, 318, 331, 358]


def load_model(checkpoint_path: str, device: torch.device, norm_stats_path: str):
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

    context_encoder.load_state_dict(ckpt["models"]["context_encoder"], strict=False)
    unet.load_state_dict(ckpt["models"]["unet"])
    aux_heads.load_state_dict(ckpt["models"]["aux_heads"])

    context_encoder.eval()
    unet.eval()
    aux_heads.eval()
    return context_encoder, unet, aux_heads, diffusion


def evaluate_configuration(
    context_encoder: nn.Module,
    unet: nn.Module,
    diffusion: GaussianDiffusion,
    device: torch.device,
    use_region: bool,
    use_cascade: bool,
    target_dates: List[int],
) -> Dict[str, Any]:
    in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
    tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
    clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
    scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

    lat = in_zarr["lat"][:]
    lon = in_zarr["lon"][:]
    ocean_mask_np = (in_zarr["inputs"][0, 20] > 0.5)
    clim_coeffs = np.nan_to_num(clim_ds["coefficients"].values, nan=0.0)
    static_channels = [20, 19, 21, 22, 23, 24]
    omega = 2.0 * math.pi / 365.25

    ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=10, eta=0.0)
    cascade = DepthCascadeSampler(
        context_encoder=context_encoder,
        unet_denoiser=unet,
        ddim_sampler=ddim,
        depths=CANONICAL_DEPTHS,
    )

    all_preds, all_targets, all_clim, timestamps = [], [], [], []

    with torch.no_grad():
        for t_day in target_dates:
            doy = int(scalar_df.loc[t_day, "day_of_year"])
            sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
            cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
            oni = float(scalar_df.loc[t_day, "oni_index"])
            iod = float(scalar_df.loc[t_day, "iod_dmi_index"])
            date_str = str(scalar_df.loc[t_day, "date"]) if "date" in scalar_df.columns else f"2025-day-{t_day:03d}"
            timestamps.append(date_str)

            seq_slice = in_zarr["inputs"][t_day - 6 : t_day + 1].copy()
            seq_slice = np.nan_to_num(seq_slice, nan=0.0)

            # If no region conditioning, zero out channels 21-24 in sequence
            if not use_region:
                seq_slice[:, 21:25] = 0.0

            x_seq = torch.from_numpy(seq_slice[None, ...]).float().to(device)
            static_feats = torch.from_numpy(seq_slice[0:1, static_channels]).float().to(device)
            if not use_region:
                static_feats[:, 2:6] = 0.0

            scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], device=device, dtype=torch.float32)

            out = cascade.sample_full_profile(
                x_seq=x_seq,
                static_features=static_feats,
                scalar_conditions=scalar_cond,
                use_cascade=use_cascade,
            )
            pred_anom = out["anomalies"][0].cpu().numpy()

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

            all_preds.append(pred_temp)
            all_targets.append(true_temp)
            all_clim.append(clim_t)

    preds_arr = np.stack(all_preds, axis=0)
    targets_arr = np.stack(all_targets, axis=0)
    clim_arr = np.stack(all_clim, axis=0)

    overall_rmse = float(compute_rmse(preds_arr, targets_arr, mask=ocean_mask_np))
    overall_mae = float(compute_mae(preds_arr, targets_arr, mask=ocean_mask_np))
    overall_bias = float(compute_bias(preds_arr, targets_arr, mask=ocean_mask_np))
    overall_corr = float(compute_correlation(preds_arr, targets_arr, mask=ocean_mask_np))
    clim_rmse = float(compute_rmse(clim_arr, targets_arr, mask=ocean_mask_np))
    murphy_skill = float(compute_murphy_skill_score(preds_arr, targets_arr, clim_arr, mask=ocean_mask_np))

    shallow_idx = [0, 1, 2, 3, 4]
    thermo_idx = [6, 7, 8, 9]
    deep_idx = [10, 11, 12, 13, 14]

    shallow_rmse = float(compute_rmse(preds_arr[:, shallow_idx], targets_arr[:, shallow_idx], mask=ocean_mask_np))
    thermo_rmse = float(compute_rmse(preds_arr[:, thermo_idx], targets_arr[:, thermo_idx], mask=ocean_mask_np))
    deep_rmse = float(compute_rmse(preds_arr[:, deep_idx], targets_arr[:, deep_idx], mask=ocean_mask_np))

    # Priority zones
    zones = get_priority_zones(lat, lon, toy_mode=False)
    zone_res = {}
    for zk, zv in zones.items():
        z_mask = ocean_mask_np & zv.spatial_mask
        if np.sum(z_mask) > 0:
            z_rmse = float(compute_rmse(preds_arr[:, zv.depth_indices], targets_arr[:, zv.depth_indices], mask=z_mask))
            z_clim = float(compute_rmse(clim_arr[:, zv.depth_indices], targets_arr[:, zv.depth_indices], mask=z_mask))
            z_skill = float(compute_murphy_skill_score(preds_arr[:, zv.depth_indices], targets_arr[:, zv.depth_indices], clim_arr[:, zv.depth_indices], mask=z_mask))
            zone_res[zk] = {
                "name": zv.name,
                "model_rmse": z_rmse,
                "clim_rmse": z_clim,
                "murphy_skill": z_skill,
            }

    return {
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "overall_bias": overall_bias,
        "overall_correlation": overall_corr,
        "clim_rmse": clim_rmse,
        "murphy_skill_score": murphy_skill,
        "shallow_rmse": shallow_rmse,
        "thermo_rmse": thermo_rmse,
        "deep_rmse": deep_rmse,
        "priority_zones": zone_res,
    }


def main():
    device = torch.device("cpu")
    print("=" * 80)
    print("CATEGORY B: RE-RUNNING CORE ABLATIONS UNDER FIXED OCEAN-ONLY PIPELINE")
    print("=" * 80)

    ckpt_path = "checkpoints/phase3_retrain_scratch_40k_full/best_checkpoint.pt"
    norm_stats = "data/processed/channel_normalization_stats_ocean_only.json"

    enc, unet, aux, diff = load_model(ckpt_path, device, norm_stats)

    configs = [
        ("Full Architecture (Region + Cascade)", True, True),
        ("Ablation: No Region Conditioning", False, True),
        ("Ablation: No Depth Cascade", True, False),
        ("Ablation: No Region & No Cascade", False, False),
    ]

    results = {}
    for name, use_region, use_cascade in configs:
        print(f"\nEvaluating: {name} (use_region={use_region}, use_cascade={use_cascade})...")
        res = evaluate_configuration(enc, unet, diff, device, use_region, use_cascade, CANONICAL_MULTI_SEASONAL_DATES)
        results[name] = res
        print(f"  Overall RMSE: {res['overall_rmse']:.4f} °C (Skill: {res['murphy_skill_score']:.4f})")
        print(f"  Shallow RMSE: {res['shallow_rmse']:.4f} °C | Thermo RMSE: {res['thermo_rmse']:.4f} °C | Deep RMSE: {res['deep_rmse']:.4f} °C")

    # Save results
    out_file = REPO_ROOT / "reports" / "clean_scratch_40k_ablation_benchmarks.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[COMPLETE] Clean ablation benchmarks written to: {out_file}")


if __name__ == "__main__":
    main()
