import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import math
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from src.utils.grid import CANONICAL_DEPTHS

# Load Zarr targets and Climatology
tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
clim_ds = xr.open_dataset("data/processed/phase2_dataset/climatology_coefficients.nc")
scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")
in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")

ocean_mask = (in_zarr["inputs"][0, 24] > 0.5) & (~np.isnan(in_zarr["inputs"][0, 0]))

scales_ds = json.load(open("data/processed/anomaly_depth_scales_zone_adaptive_p8.json"))
orig_scales = np.array(scales_ds["anomaly_stds_original"], dtype=np.float32)[:, np.newaxis, np.newaxis]
coeffs = clim_ds["coefficients"].values
omega = 2.0 * math.pi / 365.25

# Dates for Benchmark B (Sep-Dec): 237 to 358 (e.g. step of 10 or test dates)
# Let's inspect the exact dates used in Benchmark B
from scripts.diagnostics.run_phase8_comprehensive_suite import CANONICAL_MULTI_SEASONAL_DATES

# Benchmark B dates: days >= 237
benchmark_b_dates = [d for d in range(237, 359, 7)] # weekly sampling
print(f"Benchmark B evaluation across {len(benchmark_b_dates)} dates...")

# Let's compute true temperature variance and climatology variance per depth
all_true_temps = {d: [] for d in range(len(CANONICAL_DEPTHS))}
all_clim_temps = {d: [] for d in range(len(CANONICAL_DEPTHS))}
all_true_anoms = {d: [] for d in range(len(CANONICAL_DEPTHS))}

for t_day in benchmark_b_dates:
    doy = int(scalar_df.loc[t_day, "day_of_year"])
    clim_t = (
        coeffs[0]
        + coeffs[1] * math.cos(omega * doy)
        + coeffs[2] * math.sin(omega * doy)
        + coeffs[3] * math.cos(2 * omega * doy)
        + coeffs[4] * math.sin(2 * omega * doy)
    )
    true_anom = tgt_zarr["anomaly"][t_day] * orig_scales
    true_anom[0] = true_anom[1] # surface extrapolation
    true_temp = clim_t + true_anom

    for d_idx in range(len(CANONICAL_DEPTHS)):
        v_mask = np.isfinite(true_temp[d_idx]) & ocean_mask
        all_true_temps[d_idx].append(true_temp[d_idx][v_mask])
        all_clim_temps[d_idx].append(clim_t[d_idx][v_mask])
        all_true_anoms[d_idx].append(true_anom[d_idx][v_mask])

# Also load Benchmark B RMSE numbers from post_phase6_decontaminated_chronicle_and_model_audit.md
p8_cal_rmse = {
    0: 0.4700, 5: 0.4684, 10: 0.4727, 20: 0.5059, 30: 0.5569,
    50: 0.6885, 75: 0.9006, 100: 1.0996, 125: 1.1189, 150: 0.9376,
    200: 0.6265, 300: 0.3724, 500: 0.2242, 700: 0.2153, 1000: 0.2385
}

clim_rmse_b = {
    0: 0.4476, 5: 0.4476, 10: 0.4475, 20: 0.4826, 30: 0.5439,
    50: 0.6838, 75: 0.9127, 100: 1.1445, 125: 1.1806, 150: 0.9958,
    200: 0.6508, 300: 0.3527, 500: 0.2086, 700: 0.1957, 1000: 0.2219
}

print("\n=== DEPTH-BY-DEPTH R2 (COEFFICIENT OF DETERMINATION) ON BENCHMARK B (SEP-DEC) ===")
print("Depth (m) | Total Temp Var | Anom Var | Model RMSE | Climatology R² | OceanEmbed R² (Total) | Anomaly R²")
print("-" * 105)

results_table = []
for d_idx, depth in enumerate(CANONICAL_DEPTHS):
    y_true = np.concatenate(all_true_temps[d_idx])
    a_true = np.concatenate(all_true_anoms[d_idx])
    var_total = np.var(y_true)
    var_anom = np.var(a_true)
    
    mse_model = p8_cal_rmse[depth] ** 2
    mse_clim = clim_rmse_b[depth] ** 2
    
    # R2 on absolute temperature
    r2_model_total = 1.0 - (mse_model / var_total)
    r2_clim_total = 1.0 - (mse_clim / var_total)
    
    # R2 on anomaly (equivalent to anomaly skill)
    r2_model_anom = 1.0 - (mse_model / var_anom)
    
    print(f"{depth:7.0f}m  |    {var_total:6.4f}     |  {var_anom:6.4f}  |  {p8_cal_rmse[depth]:6.4f}°C |     {r2_clim_total:.4f}     |         {r2_model_total:.4f}         |  {r2_model_anom:+.4f}")

# Overall water column
all_y_true = np.concatenate([np.concatenate(all_true_temps[d]) for d in range(15)])
all_a_true = np.concatenate([np.concatenate(all_true_anoms[d]) for d in range(15)])
var_water_column = np.var(all_y_true)
var_water_anom = np.var(all_a_true)
overall_mse_model = 0.6603 ** 2
overall_mse_clim = 0.6735 ** 2
overall_r2_model = 1.0 - (overall_mse_model / var_water_column)
overall_r2_clim = 1.0 - (overall_mse_clim / var_water_column)
overall_r2_anom = 1.0 - (overall_mse_model / var_water_anom)

print("=" * 105)
print(f"OVERALL  |   {var_water_column:6.4f}     |  {var_water_anom:6.4f}  |  0.6603°C |     {overall_r2_clim:.4f}     |         {overall_r2_model:.4f}         |  {overall_r2_anom:+.4f}")
