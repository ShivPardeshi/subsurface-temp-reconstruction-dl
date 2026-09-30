"""Comprehensive Audit of Ocean-Masking Integrity across Inputs, Targets, and Auxiliaries.

Tasks:
1. Audit all 25 channels: Ocean-only vs All-grid mean & std, distortion ratio, and match against channel_normalization_stats_ocean_only.json.
2. Audit Categorical/Structural channels (20: land_ocean_mask, 21-24: region memberships).
3. Audit Anomaly Targets (oceanembed_anomaly_targets.zarr): land values and loss masking.
4. Audit Auxiliary Targets (oceanembed_auxiliary_targets.zarr): land values and loss masking.
"""

import sys
import json
import zarr
import numpy as np
import xarray as xr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

INPUTS_ZARR = REPO_ROOT / "data/processed/phase2_dataset/oceanembed_training_inputs.zarr"
TARGETS_ZARR = REPO_ROOT / "data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr"
AUX_ZARR = REPO_ROOT / "data/processed/phase2_dataset/oceanembed_auxiliary_targets.zarr"
OCEAN_STATS_JSON = REPO_ROOT / "data/processed/channel_normalization_stats_ocean_only.json"
ALL_STATS_JSON = REPO_ROOT / "data/processed/channel_normalization_stats.json"

CHANNEL_NAMES = [
    "sst", "sss", "ssh", "wind_u", "wind_v", "current_u", "current_v",
    "geostrophic_u", "geostrophic_v", "ageostrophic_u", "ageostrophic_v",
    "wind_stress_curl", "wind_mixing_energy", "precipitation",
    "latent_heat_flux", "e_minus_p_flux", "chlorophyll", "river_plume_field",
    "missingness_mask", "bathymetry_log", "land_ocean_mask",
    "region_arabian_sea", "region_bay_of_bengal", "region_confluence_zone",
    "region_open_ocean",
]

def run_audit():
    print("=" * 80)
    print("AUDIT PART 1: ALL 25 INPUT CHANNELS (OCEAN-ONLY VS ALL-GRID)")
    print("=" * 80)

    in_z = zarr.open(str(INPUTS_ZARR), mode="r")
    inputs = in_z["inputs"]  # (365, 25, 112, 240)
    ocean_mask = (inputs[0, 20, :, :] > 0.5)  # (112, 240)
    ocean_frac = np.mean(ocean_mask)
    print(f"Dataset shape: {inputs.shape}")
    print(f"Ocean cell fraction: {ocean_frac * 100:.2f}% ({np.count_nonzero(ocean_mask)} / {ocean_mask.size} cells)")

    with open(OCEAN_STATS_JSON, "r") as f:
        ocean_stats = json.load(f)
    with open(ALL_STATS_JSON, "r") as f:
        all_stats = json.load(f)

    # Compute stats on train partition (days 0-236)
    train_data = inputs[0:237]  # (237, 25, 112, 240)

    print(f"\n{'Ch':<3} {'Name':<22} {'Ocean Mean':<12} {'Ocean Std':<12} {'All-Grid Std':<14} {'Distortion Ratio':<18} {'JSON Match'}")
    print("-" * 95)

    results = []
    for c in range(25):
        name = CHANNEL_NAMES[c]
        ch_data = train_data[:, c, :, :]  # (237, 112, 240)
        
        # Ocean only
        ocean_vals = ch_data[:, ocean_mask]
        valid_ocean = np.isfinite(ocean_vals)
        ocean_mean = float(np.mean(ocean_vals[valid_ocean]))
        ocean_std = float(np.std(ocean_vals[valid_ocean]))

        # All grid (with land zero-fill as in old all-grid pipeline)
        all_vals = np.nan_to_num(ch_data.reshape(-1), nan=0.0)
        all_mean = float(np.mean(all_vals))
        all_std = float(np.std(all_vals))

        ratio = (all_std / ocean_std) if ocean_std > 1e-12 else 1.0

        json_mean = ocean_stats["means"][c]
        json_std = ocean_stats["stds"][c]
        mean_diff = abs(ocean_mean - json_mean)
        std_diff = abs(ocean_std - json_std)
        matches_json = (mean_diff < 1e-3 and std_diff < 1e-3)

        results.append({
            "channel": c,
            "name": name,
            "ocean_mean": ocean_mean,
            "ocean_std": ocean_std,
            "all_mean": all_mean,
            "all_std": all_std,
            "distortion_ratio": ratio,
            "matches_json": matches_json,
        })

        ratio_str = f"{ratio:.2f}x"
        if ratio > 1.5:
            ratio_str += " (LARGE)"
        elif ratio < 0.8:
            ratio_str += " (COMPRESSED)"
        else:
            ratio_str += " (NORMAL)"

        print(f"{c:<3} {name:<22} {ocean_mean:<12.4g} {ocean_std:<12.4g} {all_std:<14.4g} {ratio_str:<18} {'YES' if matches_json else 'DIFF'}")

    print("\n" + "=" * 80)
    print("AUDIT PART 2: CATEGORICAL & STRUCTURAL CHANNELS (20-24)")
    print("=" * 80)
    for c in range(20, 25):
        name = CHANNEL_NAMES[c]
        ch_data = train_data[0, c, :, :]
        ocean_vals = ch_data[ocean_mask]
        land_vals = ch_data[~ocean_mask]
        print(f"Ch {c} ({name}):")
        print(f"  Ocean min/max/mean: {ocean_vals.min():.4f} / {ocean_vals.max():.4f} / {ocean_vals.mean():.4f}")
        print(f"  Land  min/max/mean: {land_vals.min():.4f} / {land_vals.max():.4f} / {land_vals.mean():.4f}")

    print("\n" + "=" * 80)
    print("AUDIT PART 3: ANOMALY TARGETS (oceanembed_anomaly_targets.zarr)")
    print("=" * 80)
    tgt_z = zarr.open(str(TARGETS_ZARR), mode="r")
    tgt_key = "anomaly" if "anomaly" in tgt_z else "anomaly_targets"
    targets = tgt_z[tgt_key]  # (365, 15, 112, 240)
    print(f"Target key: '{tgt_key}', shape: {targets.shape}, dtype: {targets.dtype}")
    tgt_sample = targets[0]  # (15, 112, 240)
    for d in [0, 7, 14]:
        d_ocean = tgt_sample[d, ocean_mask]
        d_land = tgt_sample[d, ~ocean_mask]
        print(f"Depth {d} (0m, 100m, 1000m):")
        print(f"  Ocean min/max/mean/std: {np.nanmin(d_ocean):.4f} / {np.nanmax(d_ocean):.4f} / {np.nanmean(d_ocean):.4f} / {np.nanstd(d_ocean):.4f}")
        print(f"  Land  min/max/mean/std: {np.nanmin(d_land):.4f} / {np.nanmax(d_land):.4f} / {np.nanmean(d_land):.4f} / {np.nanstd(d_land):.4f}")
        print(f"  Are all land values zero or NaN? {np.all(np.isnan(d_land) | (d_land == 0.0))}")

    print("\n" + "=" * 80)
    print("AUDIT PART 4: AUXILIARY TARGETS (oceanembed_auxiliary_targets.zarr)")
    print("=" * 80)
    aux_z = zarr.open(str(AUX_ZARR), mode="r")
    aux_key = "auxiliary_targets" if "auxiliary_targets" in aux_z else "aux_target"
    aux_arr = aux_z[aux_key]
    print(f"Aux key: '{aux_key}', shape: {aux_arr.shape}, dtype: {aux_arr.dtype}")
    aux_names = ["mld", "blt", "sal_max_depth", "sal_max_strength"]
    aux_sample = aux_arr[0]  # (4, 112, 240)
    for i, name in enumerate(aux_names):
        o_v = aux_sample[i, ocean_mask]
        l_v = aux_sample[i, ~ocean_mask]
        print(f"Aux Var {i} ({name}):")
        print(f"  Ocean min/max/mean/std: {np.nanmin(o_v):.4f} / {np.nanmax(o_v):.4f} / {np.nanmean(o_v):.4f} / {np.nanstd(o_v):.4f}")
        print(f"  Land  min/max/mean/std: {np.nanmin(l_v):.4f} / {np.nanmax(l_v):.4f} / {np.nanmean(l_v):.4f} / {np.nanstd(l_v):.4f}")
        print(f"  Are land values all zero or NaN? {np.all(np.isnan(l_v) | (l_v == 0.0))}")

    # Save summary report to logs
    out_audit = REPO_ROOT / "logs/systematic_ocean_masking_audit.json"
    out_audit.parent.mkdir(exist_ok=True)
    with open(out_audit, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved systematic audit to {out_audit}")

if __name__ == "__main__":
    run_audit()
