"""Compute per-channel normalization statistics strictly on ocean cells.

Strict Anti-Leakage Protocol:
- Computes statistics ONLY across the training split (Days 0 to 236).
- Masks out zero-filled land cells using Channel 20 (land_ocean_mask > 0.5).
- Filters out non-physical coastal fill zeros for SST (>200K) and SSS (>10 PSU).
- Stores exact ocean-only means and standard deviations in JSON format.
"""

import json
from pathlib import Path
import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "phase2_dataset" / "oceanembed_training_inputs.zarr"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "channel_normalization_stats_ocean_only.json"


def compute_ocean_only_stats():
    print(f"Opening Zarr store: {DATA_PATH}")
    z = zarr.open_group(str(DATA_PATH), mode="r")
    inputs = z["inputs"]  # (T, C, H, W)
    channel_names = [str(c) for c in z["channel"][:]]
    num_channels = len(channel_names)

    # Training partition: Days 0 to 236 (anti-leakage guarantee)
    train_end = 237
    train_slice = np.nan_to_num(inputs[:train_end], nan=0.0)
    print(f"Loaded training slice: shape {train_slice.shape} (T=0..{train_end-1})")

    # Channel 20 is land_ocean_mask (1.0 = ocean, 0.0 = land)
    ocean_mask = (train_slice[0, 20] > 0.5)
    ocean_cell_count = int(np.sum(ocean_mask))
    total_cells = ocean_mask.size
    print(f"Ocean mask: {ocean_cell_count}/{total_cells} cells ({ocean_cell_count/total_cells*100:.2f}% ocean)")

    means = []
    stds = []
    summary_table = []

    for c_idx, name in enumerate(channel_names):
        # Extract ocean cells across all training time steps: (T, N_ocean)
        ocean_vals = train_slice[:, c_idx, ocean_mask].flatten()

        # Handle valid physical ranges for ocean variables with coastal fill zeros
        if name == "sst":
            # Real SST in Kelvin: ~285K to 305K (filter unassimilated coastal 0s)
            valid_vals = ocean_vals[ocean_vals > 200.0]
        elif name == "sss":
            # Real ocean salinity: 15 to 45 PSU (filter coastal 0s)
            valid_vals = ocean_vals[ocean_vals > 10.0]
        elif name == "bathymetry_log":
            # Log-bathymetry on ocean: >0.05
            valid_vals = ocean_vals[ocean_vals > 0.05]
        elif name == "land_ocean_mask":
            # Over ocean, mask is identically 1.0; use std=1.0 to prevent zero division
            valid_vals = ocean_vals
        elif name.startswith("region_"):
            # Geographic regional indicators: 0 or 1
            valid_vals = ocean_vals
        else:
            valid_vals = ocean_vals

        c_mean = float(np.mean(valid_vals))
        c_std = float(np.std(valid_vals))

        # Guard against zero or near-zero std
        if c_std < 1e-12:
            print(f"  [Notice] Channel {c_idx} ({name}) has near-zero ocean std ({c_std:.2e}), setting std=1.0")
            c_std = 1.0

        means.append(c_mean)
        stds.append(c_std)

        # Also compute old-style all-grid stats for comparison
        all_vals = train_slice[:, c_idx].flatten()
        summary_table.append({
            "idx": c_idx,
            "name": name,
            "ocean_mean": c_mean,
            "ocean_std": c_std,
            "all_mean": float(np.mean(all_vals)),
            "all_std": float(np.std(all_vals)),
            "ratio_std": float(np.std(all_vals) / c_std if c_std > 0 else 1.0),
        })

    # Save to JSON
    out_dict = {
        "description": "Per-channel normalization statistics computed strictly over ocean cells on training partition (Days 0-236)",
        "train_days": [0, train_end - 1],
        "ocean_cell_fraction": float(ocean_cell_count / total_cells),
        "channel_names": channel_names,
        "means": means,
        "stds": stds,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved ocean-only normalization stats to: {OUTPUT_PATH}")

    # Print comparative audit
    print("\n" + "=" * 90)
    print(f"{'Idx':<4} {'Channel Name':<24} {'Ocean Mean':<14} {'Ocean Std':<14} {'All-Grid Std':<14} {'Scale Distortion':<16}")
    print("=" * 90)
    for row in summary_table:
        distortion = f"{row['ratio_std']:.1f}x"
        print(f"{row['idx']:<4} {row['name']:<24} {row['ocean_mean']:<14.4e} {row['ocean_std']:<14.4e} {row['all_std']:<14.4e} {distortion:<16}")
    print("=" * 90)


if __name__ == "__main__":
    compute_ocean_only_stats()
