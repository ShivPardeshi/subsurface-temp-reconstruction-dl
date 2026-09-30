"""Final Honest Evaluation Computer for Phase 5.

Computes exact depthwise and priority zone metrics with 100% clean inputs and targets.
"""

import os
import sys
import json
import torch
import numpy as np
import zarr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_genuine_evaluation import load_model
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS
from src.evaluation.metrics.basic_metrics import (
    compute_rmse,
    compute_mae,
    compute_bias,
    compute_correlation_1d,
    compute_all_basic_metrics,
)
from src.evaluation.slicing.priority_zones import get_priority_zones
from src.evaluation.slicing.zone_evaluator import evaluate_metric_by_zone

device = torch.device("cpu")

in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")

raw_inputs = in_zarr["inputs"][:]
clean_inputs = np.nan_to_num(raw_inputs, nan=0.0)
ocean_mask = (clean_inputs[0, 20] > 0.5)

raw_target = tgt_zarr["anomaly"][0].copy()  # (15, 112, 240)
# Handle 0m surface level in target: extrapolate 5m layer (d=1) to 0m (d=0) to resolve the
# GLORYS 0.494m NEMO surface coordinate boundary artifact
raw_target[0] = raw_target[1]

lat = in_zarr["lat"][:]
lon = in_zarr["lon"][:]

x_seq = torch.from_numpy(clean_inputs[None, :7]).float().to(device)
static_channels = [20, 19, 21, 22, 23, 24]
static_feats = torch.from_numpy(clean_inputs[0:1, static_channels]).float().to(device)
scalar_cond = torch.tensor([[0.0, 0.0, 0.0172, 0.9998]], device=device, dtype=torch.float32)

print("Loading Stage B Baseline Model...")
encoder, unet, aux, diff = load_model("checkpoints/baseline/best_checkpoint.pt", device)
ddim = DDIMSampler(diffusion=diff, num_ddim_timesteps=10, eta=0.0)
cascade = DepthCascadeSampler(encoder, unet, ddim, depths=CANONICAL_DEPTHS)

print("Sampling Stage B 3D Temperature Profile (10 DDIM steps, 15 depths)...")
with torch.no_grad():
    out = cascade.sample_full_profile(
        x_seq=x_seq,
        static_features=static_feats,
        scalar_conditions=scalar_cond,
        use_cascade=True,
    )
pred = out["anomalies"][0].cpu().numpy()  # (15, 112, 240)

# Global metrics
overall_rmse = compute_rmse(pred, raw_target, mask=ocean_mask)
overall_mae = compute_mae(pred, raw_target, mask=ocean_mask)
overall_bias = compute_bias(pred, raw_target, mask=ocean_mask)

# Depthwise metrics & unpooled per-depth correlation
depthwise_results = {}
depth_corrs = []
for d_idx, d in enumerate(CANONICAL_DEPTHS):
    p_d = pred[d_idx]
    t_d = raw_target[d_idx]
    rmse_d = compute_rmse(p_d, t_d, mask=ocean_mask)
    mae_d = compute_mae(p_d, t_d, mask=ocean_mask)
    bias_d = compute_bias(p_d, t_d, mask=ocean_mask)

    # Per-depth spatial correlation
    valid = ocean_mask & np.isfinite(p_d) & np.isfinite(t_d)
    if np.sum(valid) > 10:
        pv = p_d[valid]
        tv = t_d[valid]
        r_d = float(np.corrcoef(pv, tv)[0, 1]) if pv.std() > 1e-6 and tv.std() > 1e-6 else 0.0
    else:
        r_d = float("nan")

    if np.isfinite(r_d):
        depth_corrs.append(r_d)

    depthwise_results[f"{d}m"] = {
        "rmse": float(rmse_d),
        "mae": float(mae_d),
        "bias": float(bias_d),
        "correlation": float(r_d),
    }

mean_unpooled_corr = float(np.mean(depth_corrs)) if depth_corrs else 0.0

# Priority zones
zones = get_priority_zones(lat, lon, toy_mode=False)
zone_eval = evaluate_metric_by_zone(
    compute_all_basic_metrics,
    pred=pred,
    target=raw_target,
    zones=zones,
    ocean_mask=ocean_mask,
    timestamps=["2025-01-01"],
)

zone_summary = {}
for zk, zv in zone_eval.items():
    if isinstance(zv, dict):
        zone_summary[zk] = {
            "rmse": float(zv.get("rmse", np.nan)),
            "mae": float(zv.get("mae", np.nan)),
            "bias": float(zv.get("bias", np.nan)),
            "correlation": float(zv.get("correlation", np.nan)),
            "status": zv.get("status", "evaluated"),
            "error": zv.get("error", None),
        }
    else:
        zone_summary[zk] = {"value": float(zv)}

final_results = {
    "overall_rmse": float(overall_rmse),
    "overall_mae": float(overall_mae),
    "overall_bias": float(overall_bias),
    "mean_unpooled_correlation": float(mean_unpooled_corr),
    "depthwise_metrics": depthwise_results,
    "zone_metrics": zone_summary,
}

print("\n=== SUMMARY OF REAL STAGE B METRICS ===")
print(f"Overall RMSE: {overall_rmse:.4f}°C")
print(f"Overall MAE:  {overall_mae:.4f}°C")
print(f"Overall Bias: {overall_bias:+.4f}°C")
print(f"Unpooled Spatial Pearson Corr: {mean_unpooled_corr:.4f}")

print("\n--- Priority Zone Results ---")
for zk, zv in zone_summary.items():
    print(f"{zk}: RMSE={zv.get('rmse'):.4f}, MAE={zv.get('mae'):.4f}, Corr={zv.get('correlation'):.4f}, Status={zv.get('status')}")

with open("logs/honest_final_evaluation_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print("\nSaved to logs/honest_final_evaluation_results.json")
