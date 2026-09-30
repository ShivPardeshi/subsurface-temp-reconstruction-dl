"""Benchmark Evaluation for Phase 7 Dampened Scaling 25k Checkpoint on GCP/Local."""

import sys
import os
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

import torch
import numpy as np
import pandas as pd
import zarr

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.auxiliary_heads import AuxiliaryHeads
from src.models.diffusion import GaussianDiffusion
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS

CLIMATOLOGY_BASELINE_RMSE = 0.6422

def load_phase7_model(checkpoint_path: str, device: torch.device):
    print(f"Loading Phase 7 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
    return context_encoder, unet, aux_heads, diffusion


def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Phase 7 model on device: {device}")

    ckpt_path = "checkpoints/phase7_dampened_scaling_thermocline_25k/last_checkpoint.pt"
    context_encoder, unet, aux_heads, diffusion = load_phase7_model(ckpt_path, device)

    scales_path = "data/processed/anomaly_depth_scales_dampened_p05.json"

    from src.training.dataset import OceanEmbedDataset
    dataset = OceanEmbedDataset(
        inputs_zarr_path="data/processed/phase2_dataset/oceanembed_training_inputs.zarr",
        anomaly_targets_zarr_path="data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr",
        aux_targets_zarr_path="data/processed/phase2_dataset/oceanembed_auxiliary_targets.zarr",
        scalar_csv_path="data/processed/phase2_dataset/scalar_conditioning.csv",
        depth_scales_path=scales_path,
        sequence_length=7,
        augment=False,
        standardize_targets=False,
    )

    total_samples = len(dataset)
    test_start = max(0, total_samples - 60)
    eval_indices = list(range(test_start, total_samples, max(1, 60 // 10)))[:10]
    print(f"Evaluating across {len(eval_indices)} held-out test samples: {eval_indices}")

    ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=25, eta=0.0, schedule_type="quadratic")
    cascade = DepthCascadeSampler(
        context_encoder=context_encoder,
        unet_denoiser=unet,
        ddim_sampler=ddim,
        depths=CANONICAL_DEPTHS,
        depth_scales_path=scales_path,
        rescale_output=True,
        aux_heads=aux_heads,
    )

    per_depth_errors = {d: [] for d in CANONICAL_DEPTHS}
    zone_errors = {"arabian_sea": [], "bay_of_bengal": [], "confluence": [], "open_ocean": []}
    all_squared_diffs = []

    t0 = time.time()
    with torch.no_grad():
        for i, sample_idx in enumerate(eval_indices):
            print(f"  Evaluating sample {i+1}/{len(eval_indices)} (index {sample_idx})...")
            sample = dataset[sample_idx]
            x_seq = sample["x_seq"].unsqueeze(0).to(device).float()
            static_feats = sample["static_features"].unsqueeze(0).to(device).float()
            scalar_cond = sample["scalar_cond"].unsqueeze(0).to(device).float()
            true_anomalies = sample["anomaly_target"].numpy()

            out = cascade.sample_full_profile(
                x_seq=x_seq,
                static_features=static_feats,
                scalar_conditions=scalar_cond,
                use_cascade=True,
            )
            pred_anomalies = out["anomalies"].squeeze(0).cpu().numpy()

            static_np = sample["static_features"].numpy()
            ocean_mask = (static_np[0] > 0.5)
            as_mask = (static_np[2] > 0.5) & ocean_mask
            bob_mask = (static_np[3] > 0.5) & ocean_mask
            conf_mask = (static_np[4] > 0.5) & ocean_mask
            open_mask = (static_np[5] > 0.5) & ocean_mask

            for d_idx, depth in enumerate(CANONICAL_DEPTHS):
                diff = (pred_anomalies[d_idx] - true_anomalies[d_idx])
                sq_diff = (diff ** 2)[ocean_mask]
                if len(sq_diff) > 0:
                    mse_d = np.mean(sq_diff)
                    per_depth_errors[depth].append(mse_d)
                    all_squared_diffs.extend(sq_diff.tolist())

                if len(diff[as_mask]) > 0:
                    zone_errors["arabian_sea"].append(np.mean((diff ** 2)[as_mask]))
                if len(diff[bob_mask]) > 0:
                    zone_errors["bay_of_bengal"].append(np.mean((diff ** 2)[bob_mask]))
                if len(diff[conf_mask]) > 0:
                    zone_errors["confluence"].append(np.mean((diff ** 2)[conf_mask]))
                if len(diff[open_mask]) > 0:
                    zone_errors["open_ocean"].append(np.mean((diff ** 2)[open_mask]))

    eval_dur = time.time() - t0
    overall_rmse = math.sqrt(np.mean(all_squared_diffs))
    depth_rmses = {d: math.sqrt(np.mean(errs)) for d, errs in per_depth_errors.items()}

    surf_rmses = [depth_rmses[d] for d in CANONICAL_DEPTHS[:5]]
    thermo_rmses = [depth_rmses[d] for d in CANONICAL_DEPTHS[5:11]]
    deep_rmses = [depth_rmses[d] for d in CANONICAL_DEPTHS[11:]]

    surface_rmse = float(np.mean(surf_rmses))
    thermo_rmse = float(np.mean(thermo_rmses))
    deep_rmse = float(np.mean(deep_rmses))

    murphy_skill = (1.0 - (overall_rmse ** 2) / (CLIMATOLOGY_BASELINE_RMSE ** 2)) * 100.0
    zone_rmses = {z: math.sqrt(np.mean(errs)) for z, errs in zone_errors.items() if len(errs) > 0}

    results = {
        "model_name": "Phase 7 Dampened Scaling Pure Diffusion (25k Steps)",
        "overall_rmse": round(overall_rmse, 4),
        "climatology_baseline_rmse": CLIMATOLOGY_BASELINE_RMSE,
        "murphy_skill_score_pct": round(murphy_skill, 2),
        "surface_rmse_0_30m": round(surface_rmse, 4),
        "thermocline_rmse_50_200m": round(thermo_rmse, 4),
        "deep_rmse_250_1000m": round(deep_rmse, 4),
        "per_depth_rmse": {str(k): round(v, 4) for k, v in depth_rmses.items()},
        "zone_rmse": {k: round(v, 4) for k, v in zone_rmses.items()},
        "eval_duration_sec": round(eval_dur, 1),
    }

    print("\n" + "=" * 80)
    print("PHASE 7 DAMPENED SCALING BENCHMARK RESULTS (15 DEPTHS)")
    print("=" * 80)
    print(f"Overall 15-Depth RMSE:        {overall_rmse:.4f} °C (Climatology Baseline: {CLIMATOLOGY_BASELINE_RMSE:.4f} °C)")
    print(f"Murphy Skill Score:           {murphy_skill:+.2f}%")
    print(f"Surface (0-30m) RMSE:         {surface_rmse:.4f} °C")
    print(f"Thermocline (50-200m) RMSE:   {thermo_rmse:.4f} °C")
    print(f"Deep Ocean (250-1000m) RMSE:  {deep_rmse:.4f} °C")
    print("-" * 80)
    print("Per-Depth RMSE Breakdown:")
    for d in CANONICAL_DEPTHS:
        print(f"  Depth {d:4.0f}m: {depth_rmses[d]:.4f} °C")
    print("-" * 80)
    print("Regional RMSE Breakdown:")
    for z, val in zone_rmses.items():
        print(f"  {z.replace('_', ' ').title():20s}: {val:.4f} °C")
    print("=" * 80)

    out_path = Path("reports/phase7_dampened_scaling_25k_benchmark_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    evaluate()
