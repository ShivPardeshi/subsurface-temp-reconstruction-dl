"""Rigorous Re-Evaluation of ALL Models on Genuinely Held-Out Test Data.

Evaluates every model checkpoint against:
1. Benchmark A: Pure Held-Out Test Set (Nov-Dec, Days 298-358, 0% Train Leakage)
2. Benchmark B: Extended Out-of-Sample Set (Sep-Dec, Days 237-358, 0% Train Leakage)
3. Diagnostic Comparison: Contaminated 10-Date Benchmark (Days 15-358) to quantify exact leakage distortion.

Models Evaluated:
- Baseline 1: 2-Harmonic Climatology
- Baseline 2: Ridge Regression (Trained on Days 6-236)
- Model 3: Historical Hybrid (75% Ridge + 25% Phase 4 Diffusion)
- Model 4: Phase 4 Baseline Diffusion (baseline_20k_fixA2_seed42)
- Model 5: Phase 5 Rectified Pure Diffusion (25k steps)
- Model 6: Phase 6 Thermocline Breakthrough (25k steps)
- Model 7: Phase 7 Dampened Scaling (25k steps)
- Model 8: Phase 8 Zone-Adaptive Scaling Raw (15k steps)
- Model 9: Phase 8 Zone-Adaptive Calibrated Ensemble (Strategy 1+2)
"""

import sys
import os
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 8)
print(f"Running rigorous benchmark audit on {device}...")

# 1. Dataset Loading
in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

lat = in_zarr["lat"][:]
lon = in_zarr["lon"][:]
ocean_mask_np = (in_zarr["inputs"][0, 20] > 0.5)
as_mask = (in_zarr["inputs"][0, 21] > 0.5) & ocean_mask_np
bob_mask = (in_zarr["inputs"][0, 22] > 0.5) & ocean_mask_np
static_channels = [20, 19, 21, 22, 23, 24]

# Define the Benchmarks
# Benchmark A: Pure Held-Out Test Set (Nov-Dec, Days 298-358)
BENCHMARK_A_INDICES = [305, 311, 317, 323, 329, 335, 341, 347, 353, 359]

# Benchmark B: Extended Out-of-Sample Set (Sep-Dec, Days 237-358)
BENCHMARK_B_INDICES = [238, 251, 264, 277, 290, 303, 316, 330, 344, 357]

# Diagnostic: Old Contaminated 10-Date Benchmark (Jan-Dec)
CONTAMINATED_INDICES = [14, 59, 104, 149, 194, 239, 284, 317, 330, 357]

print(f"Benchmark A (Nov-Dec Held-Out Test): {BENCHMARK_A_INDICES}")
print(f"Benchmark B (Sep-Dec Out-of-Sample): {BENCHMARK_B_INDICES}")
print(f"Contaminated 10-Date Diagnostic:     {CONTAMINATED_INDICES}")

# 2. Train Ridge Weights strictly on training slice (Days 6 to 236)
def get_ridge_model():
    feature_channels = [0, 1, 2, 3, 4, 11, 14, 15, 19, 21, 22, 23, 24]
    train_days = list(range(6, 237))
    n_ocean = int(np.sum(ocean_mask_np))
    lat_mesh, lon_mesh = np.meshgrid(lat, lon, indexing="ij")
    lat_pts = lat_mesh[ocean_mask_np][None, :]
    lon_pts = lon_mesh[ocean_mask_np][None, :]

    X_list, Y_list = [], []
    for t_day in train_days:
        seq = in_zarr["inputs"][t_day - 6 : t_day + 1]
        anom = tgt_zarr["anomaly"][t_day].copy()
        anom[0] = anom[1]

        sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
        cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
        oni = float(scalar_df.loc[t_day, "oni_index"])
        iod = float(scalar_df.loc[t_day, "iod_dmi_index"])

        curr_feat = seq[-1, feature_channels][:, ocean_mask_np]
        mean_feat = np.mean(seq[:, feature_channels], axis=0)[:, ocean_mask_np]
        diff_feat = (seq[-1, feature_channels] - seq[0, feature_channels])[:, ocean_mask_np]
        scalars = np.array([[sin_doy], [cos_doy], [oni], [iod]]) * np.ones((4, n_ocean))

        x_day = np.vstack([curr_feat, mean_feat, diff_feat, lat_pts, lon_pts, scalars]).T
        y_day = anom[:, ocean_mask_np].T
        X_list.append(x_day)
        Y_list.append(y_day)

    X_train = np.nan_to_num(np.vstack(X_list), nan=0.0)
    Y_train = np.nan_to_num(np.vstack(Y_list), nan=0.0)

    mean_X = np.mean(X_train, axis=0, keepdims=True)
    std_X = np.std(X_train, axis=0, keepdims=True) + 1e-6
    X_train_norm = (X_train - mean_X) / std_X
    X_train_b = np.hstack([X_train_norm, np.ones((X_train_norm.shape[0], 1))])

    alpha = 100.0
    XtX = X_train_b.T @ X_train_b + alpha * np.eye(X_train_b.shape[1])
    XtY = X_train_b.T @ Y_train
    W = np.linalg.solve(XtX, XtY)
    return W, mean_X, std_X, feature_channels

print("Fitting Ridge regression model on training split (Days 6-236)...")
W_ridge, mean_X, std_X, ridge_feat_ch = get_ridge_model()

def predict_ridge(sample_idx: int) -> np.ndarray:
    seq = in_zarr["inputs"][sample_idx - 6 : sample_idx + 1]
    n_ocean = int(np.sum(ocean_mask_np))
    lat_mesh, lon_mesh = np.meshgrid(lat, lon, indexing="ij")
    lat_pts = lat_mesh[ocean_mask_np][None, :]
    lon_pts = lon_mesh[ocean_mask_np][None, :]

    sin_doy = float(scalar_df.loc[sample_idx, "sin_doy"])
    cos_doy = float(scalar_df.loc[sample_idx, "cos_doy"])
    oni = float(scalar_df.loc[sample_idx, "oni_index"])
    iod = float(scalar_df.loc[sample_idx, "iod_dmi_index"])

    curr_feat = seq[-1, ridge_feat_ch][:, ocean_mask_np]
    mean_feat = np.mean(seq[:, ridge_feat_ch], axis=0)[:, ocean_mask_np]
    diff_feat = (seq[-1, ridge_feat_ch] - seq[0, ridge_feat_ch])[:, ocean_mask_np]
    scalars = np.array([[sin_doy], [cos_doy], [oni], [iod]]) * np.ones((4, n_ocean))

    x_day = np.vstack([curr_feat, mean_feat, diff_feat, lat_pts, lon_pts, scalars]).T
    x_norm = (x_day - mean_X) / std_X
    x_b = np.hstack([x_norm, np.ones((x_norm.shape[0], 1))])
    y_pred_pts = (x_b @ W_ridge).T # (15, n_ocean)

    pred_full = np.zeros((15, 112, 240), dtype=np.float32)
    pred_full[:, ocean_mask_np] = y_pred_pts
    return pred_full

# 3. Model Loader Helper
def load_diffusion_cascade(ckpt_path: str, scale_path: str, unet_in_channels: int = 74, cond_in_dim: int = 14):
    norm_stats = "data/processed/channel_normalization_stats_ocean_only.json"
    context_encoder = ContextEncoder(
        in_channels=25,
        hidden_dims=[32, 64, 64],
        normalize_inputs=True,
        norm_stats_path=norm_stats,
    ).to(device)

    unet = UNetDenoiser(
        in_channels=unet_in_channels,
        stage_channels=[32, 64, 128, 256],
        cond_in_dim=cond_in_dim,
    ).to(device)

    aux_heads = AuxiliaryHeads(in_features=64).to(device)
    diffusion = GaussianDiffusion(timesteps=1000, schedule_type="cosine").to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    context_encoder.load_state_dict(ckpt["models"]["context_encoder"], strict=False)
    unet.load_state_dict(ckpt["models"]["unet"], strict=False)
    if "aux_heads" in ckpt["models"]:
        aux_heads.load_state_dict(ckpt["models"]["aux_heads"], strict=False)

    context_encoder.eval()
    unet.eval()
    aux_heads.eval()

    ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=25, eta=0.0, schedule_type="quadratic")
    cascade = DepthCascadeSampler(
        context_encoder=context_encoder,
        unet_denoiser=unet,
        ddim_sampler=ddim,
        depths=CANONICAL_DEPTHS,
        depth_scales_path=scale_path,
        rescale_output=True,
        aux_heads=aux_heads,
    )
    return cascade

# Load Models
print("Loading Model Checkpoints...")
# Phase 8
p8_cascade = load_diffusion_cascade(
    "checkpoints/phase8_zone_adaptive_15k/last_checkpoint.pt",
    "data/processed/anomaly_depth_scales_zone_adaptive_p8.json",
    unet_in_channels=74, cond_in_dim=14
)

# Phase 7
p7_cascade = load_diffusion_cascade(
    "checkpoints/phase7_dampened_scaling_thermocline_25k/last_checkpoint.pt",
    "data/processed/anomaly_depth_scales_dampened_p7.json",
    unet_in_channels=74, cond_in_dim=14
)

# Phase 6
p6_cascade = load_diffusion_cascade(
    "checkpoints/phase6_thermocline_breakthrough_25k/last_checkpoint.pt",
    "data/processed/anomaly_depth_scales_phase6.json",
    unet_in_channels=74, cond_in_dim=14
)

# Phase 4/5 Baseline
p4_cascade = load_diffusion_cascade(
    "checkpoints/baseline_20k_fixA2_seed42/best_checkpoint.pt",
    "data/processed/anomaly_depth_scales.json",
    unet_in_channels=72, cond_in_dim=8
)

GAMMA_WEIGHTS = {
    0: 0.88, 5: 0.88, 10: 0.89, 20: 0.90, 30: 0.90,
    50: 0.91, 75: 0.93, 100: 0.94, 125: 0.94, 150: 0.93,
    200: 0.90, 300: 0.88, 500: 0.86, 700: 0.85, 1000: 0.85
}
BIAS_CORRECTION = {
    0: 0.045, 5: 0.048, 10: 0.045, 20: 0.040, 30: 0.035,
    50: 0.020, 75: 0.000, 100: -0.010, 125: 0.000, 150: 0.005,
    200: 0.015, 300: 0.015, 500: 0.012, 700: 0.010, 1000: 0.012
}

def evaluate_on_indices(indices: List[int], benchmark_label: str) -> Dict[str, Any]:
    print(f"\n--- Evaluating on {benchmark_label} (N={len(indices)}) ---")
    
    models = ["Climatology", "Ridge", "Historical Hybrid (75% R + 25% P4)", "Phase 4 Baseline", "Phase 6", "Phase 7", "Phase 8 Raw", "Phase 8 Calibrated (Opt)"]
    sq_errors = {m: {d: [] for d in CANONICAL_DEPTHS} for m in models}
    overall_sq = {m: [] for m in models}

    # Thermocline core (75-150m) accumulator
    thermo_sq = {m: [] for m in models}
    # Barrier layer (0-30m BoB) accumulator
    bob_bl_sq = {m: [] for m in models}

    with torch.no_grad():
        for count, s_idx in enumerate(indices):
            print(f"[{benchmark_label}] Sample {count+1}/{len(indices)} (Index {s_idx}, DOY {int(scalar_df.loc[s_idx, 'day_of_year'])})...", flush=True)
            seq = np.nan_to_num(in_zarr["inputs"][s_idx - 6 : s_idx + 1], nan=0.0)
            static_feats_np = np.nan_to_num(seq[-1, static_channels], nan=0.0)
            true_anom = np.nan_to_num(tgt_zarr["anomaly"][s_idx].copy(), nan=0.0)
            true_anom[0] = true_anom[1]

            sin_doy = float(scalar_df.loc[s_idx, "sin_doy"])
            cos_doy = float(scalar_df.loc[s_idx, "cos_doy"])
            oni = float(scalar_df.loc[s_idx, "oni_index"])
            iod = float(scalar_df.loc[s_idx, "iod_dmi_index"])

            x_seq = torch.from_numpy(seq).unsqueeze(0).float().to(device)
            static_feats = torch.from_numpy(static_feats_np).unsqueeze(0).float().to(device)
            scalar_cond = torch.tensor([[sin_doy, cos_doy, oni, iod]], dtype=torch.float32, device=device)

            # Predictions
            # 1. Climatology
            pred_clim = np.zeros_like(true_anom)

            # 2. Ridge
            pred_ridge = predict_ridge(s_idx)

            # 3. Phase 4 Baseline Diffusion
            torch.manual_seed(42 + s_idx)
            out_p4 = p4_cascade.sample_full_profile(x_seq=x_seq, static_features=static_feats, scalar_conditions=scalar_cond, use_cascade=True)
            pred_p4 = out_p4["anomalies"].squeeze(0).cpu().numpy()

            # 4. Historical Hybrid (75% Ridge + 25% P4 Diffusion)
            pred_hybrid = 0.75 * pred_ridge + 0.25 * pred_p4

            # 5. Phase 6 Diffusion
            torch.manual_seed(42 + s_idx)
            out_p6 = p6_cascade.sample_full_profile(x_seq=x_seq, static_features=static_feats, scalar_conditions=scalar_cond, use_cascade=True)
            pred_p6 = out_p6["anomalies"].squeeze(0).cpu().numpy()

            # 6. Phase 7 Diffusion
            torch.manual_seed(42 + s_idx)
            out_p7 = p7_cascade.sample_full_profile(x_seq=x_seq, static_features=static_feats, scalar_conditions=scalar_cond, use_cascade=True)
            pred_p7 = out_p7["anomalies"].squeeze(0).cpu().numpy()

            # 7. Phase 8 Raw Diffusion (Sample 1)
            torch.manual_seed(42 + s_idx)
            out_p8_1 = p8_cascade.sample_full_profile(x_seq=x_seq, static_features=static_feats, scalar_conditions=scalar_cond, use_cascade=True)
            pred_p8_1 = out_p8_1["anomalies"].squeeze(0).cpu().numpy()

            # Phase 8 Sample 2 (for ensemble)
            torch.manual_seed(1042 + s_idx)
            out_p8_2 = p8_cascade.sample_full_profile(x_seq=x_seq, static_features=static_feats, scalar_conditions=scalar_cond, use_cascade=True)
            pred_p8_2 = out_p8_2["anomalies"].squeeze(0).cpu().numpy()

            # 8. Phase 8 Calibrated Ensemble
            ens_p8 = 0.5 * (pred_p8_1 + pred_p8_2)
            pred_p8_cal = np.zeros_like(ens_p8)
            for d_idx, depth in enumerate(CANONICAL_DEPTHS):
                gamma = GAMMA_WEIGHTS[depth]
                b = BIAS_CORRECTION[depth]
                pred_p8_cal[d_idx] = (ens_p8[d_idx] * gamma) - (b * 0.8)

            preds_dict = {
                "Climatology": pred_clim,
                "Ridge": pred_ridge,
                "Historical Hybrid (75% R + 25% P4)": pred_hybrid,
                "Phase 4 Baseline": pred_p4,
                "Phase 6": pred_p6,
                "Phase 7": pred_p7,
                "Phase 8 Raw": pred_p8_1,
                "Phase 8 Calibrated (Opt)": pred_p8_cal,
            }

            for m in models:
                for d_idx, depth in enumerate(CANONICAL_DEPTHS):
                    diff_ocean = (preds_dict[m][d_idx] - true_anom[d_idx])[ocean_mask_np]
                    sq = (diff_ocean ** 2).tolist()
                    sq_errors[m][depth].extend(sq)
                    overall_sq[m].extend(sq)

                    if depth in [75, 100, 125, 150]:
                        thermo_sq[m].extend(sq)
                    if depth in [0, 5, 10, 20, 30]:
                        diff_bob = (preds_dict[m][d_idx] - true_anom[d_idx])[bob_mask]
                        bob_bl_sq[m].extend(((diff_bob) ** 2).tolist())

    res = {}
    for m in models:
        rmse_overall = math.sqrt(np.mean(overall_sq[m]))
        depth_rmses = {d: math.sqrt(np.mean(sq_errors[m][d])) for d in CANONICAL_DEPTHS}
        thermo_rmse = math.sqrt(np.mean(thermo_sq[m]))
        bob_rmse = math.sqrt(np.mean(bob_bl_sq[m]))
        
        clim_overall = math.sqrt(np.mean(overall_sq["Climatology"]))
        clim_thermo = math.sqrt(np.mean(thermo_sq["Climatology"]))
        clim_bob = math.sqrt(np.mean(bob_bl_sq["Climatology"]))

        skill_overall = (1.0 - (rmse_overall**2) / (clim_overall**2)) * 100.0
        skill_thermo = (1.0 - (thermo_rmse**2) / (clim_thermo**2)) * 100.0

        res[m] = {
            "overall_rmse": rmse_overall,
            "overall_skill_pct": skill_overall,
            "thermo_core_rmse": thermo_rmse,
            "thermo_skill_pct": skill_thermo,
            "bob_barrier_rmse": bob_rmse,
            "depth_rmses": depth_rmses,
        }
    return res

print("\n" + "=" * 90)
print("EXECUTION 1: BENCHMARK A (Pure Held-Out Test Set: Nov-Dec)")
print("=" * 90)
results_bench_a = evaluate_on_indices(BENCHMARK_A_INDICES, "Benchmark A (Nov-Dec Held-Out)")

print("\n" + "=" * 90)
print("EXECUTION 2: BENCHMARK B (Extended Out-of-Sample: Sep-Dec)")
print("=" * 90)
results_bench_b = evaluate_on_indices(BENCHMARK_B_INDICES, "Benchmark B (Sep-Dec Out-of-Sample)")

print("\n" + "=" * 90)
print("EXECUTION 3: DIAGNOSTIC (Contaminated 10-Date Benchmark)")
print("=" * 90)
results_contaminated = evaluate_on_indices(CONTAMINATED_INDICES, "Contaminated Diagnostic (Jan-Dec)")

final_audit = {
    "benchmark_a_nov_dec_held_out_test": results_bench_a,
    "benchmark_b_sep_dec_out_of_sample": results_bench_b,
    "diagnostic_contaminated_10_date": results_contaminated
}

out_file = "reports/rigorous_uncontaminated_model_re_evaluation.json"
with open(out_file, "w") as f:
    json.dump(final_audit, f, indent=2)

print(f"\nAll results saved to {out_file}")
