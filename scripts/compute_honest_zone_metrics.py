"""Compute Honest Priority Zone Metrics for Stage B, Stage C, and Stage D."""

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
    compute_all_basic_metrics,
    compute_rmse,
    compute_mae,
    compute_correlation,
)
from src.evaluation.slicing.priority_zones import get_priority_zones
from src.evaluation.slicing.zone_evaluator import evaluate_metric_by_zone

device = torch.device("cpu")

in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")

inputs = in_zarr["inputs"][:]
true_anom = tgt_zarr["anomaly"][0]  # (15, 112, 240)
lat = in_zarr["lat"][:]
lon = in_zarr["lon"][:]
ocean_mask = (inputs[0, 20] > 0.5)

x_seq = torch.from_numpy(inputs[None, :7]).float().to(device)
static_channels = [20, 19, 21, 22, 23, 24]
static_feats = torch.from_numpy(inputs[0:1, static_channels]).float().to(device)
scalar_cond = torch.tensor([[0.0, 0.0, 0.0172, 0.9998]], device=device, dtype=torch.float32)

zones = get_priority_zones(lat, lon, toy_mode=False)
timestamps = ["2025-01-01"]

stages = [
    ("Stage B (Baseline)", "checkpoints/baseline/best_checkpoint.pt", True, True),
    ("Stage C (No Region)", "checkpoints/ablation_no_region/best_checkpoint.pt", True, False),
    ("Stage D (No Cascade)", "checkpoints/ablation_no_cascade/best_checkpoint.pt", False, True),
]

all_stage_zone_results = {}

for name, ckpt_path, use_cascade, use_region in stages:
    print(f"\nComputing zones for {name}...")
    encoder, unet, aux, diff = load_model(ckpt_path, device)
    ddim = DDIMSampler(diffusion=diff, num_ddim_timesteps=5, eta=0.0)
    cascade = DepthCascadeSampler(encoder, unet, ddim, depths=CANONICAL_DEPTHS)

    s_feats = static_feats.clone()
    if not use_region:
        s_feats[:, 2:6] = 0.0

    with torch.no_grad():
        out = cascade.sample_full_profile(
            x_seq=x_seq,
            static_features=s_feats,
            scalar_conditions=scalar_cond,
            use_cascade=use_cascade,
        )
    pred_anom = out["anomalies"][0].cpu().numpy()

    # Fill depth 0 surface NaN in true_anom using 5m layer (physical proxy for surface layer)
    # for metric computation if needed, or evaluate on valid depths
    zone_res = evaluate_metric_by_zone(
        compute_all_basic_metrics,
        pred=pred_anom,
        target=true_anom,
        zones=zones,
        ocean_mask=ocean_mask,
        timestamps=timestamps,
    )

    stage_summary = {}
    for zk, zv in zone_res.items():
        if isinstance(zv, dict):
            stage_summary[zk] = {
                "rmse": float(zv.get("rmse", np.nan)),
                "mae": float(zv.get("mae", np.nan)),
                "correlation": float(zv.get("correlation", np.nan)),
                "status": zv.get("status", "evaluated"),
                "error": zv.get("error", None),
            }
        else:
            stage_summary[zk] = {"value": float(zv)}
    all_stage_zone_results[name] = stage_summary
    print(f"Results for {name}:")
    for zk, zv in stage_summary.items():
        print(f"  {zk}: RMSE={zv.get('rmse'):.4f}, MAE={zv.get('mae'):.4f}, Status={zv.get('status')}")

with open("logs/genuine_zone_metrics_bcd.json", "w") as f:
    json.dump(all_stage_zone_results, f, indent=2)

print("\nSaved genuine zone metrics to logs/genuine_zone_metrics_bcd.json")
