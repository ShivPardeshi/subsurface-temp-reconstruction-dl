"""
Automated sanity checks and validation for OceanEmbed Phase 1 data cube.
"""

import os
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from src.utils.io_utils import open_datacube_zarr
from src.utils.logging_config import setup_logger

logger = setup_logger("sanity_checks")

# Physical bounds dictionary: (min_val, max_val)
PHYSICAL_BOUNDS = {
    "sst": (270.0, 315.0),            # Kelvin (~ -3°C to 42°C)
    "sss": (15.0, 45.0),              # psu
    "ssh": (-3.0, 3.0),               # meters
    "wind_u": (-60.0, 60.0),          # m/s
    "wind_v": (-60.0, 60.0),          # m/s
    "current_u": (-5.0, 5.0),         # m/s
    "current_v": (-5.0, 5.0),         # m/s
    "precipitation": (0.0, 1000.0),   # mm/day
    "chlorophyll": (0.0, 100.0),      # mg/m^3
    "bathymetry_log": (0.0, 5.0),     # log10(1 + depth) -> 0 to ~10,000m
    "land_ocean_mask": (0.0, 1.0),
    "region_arabian_sea": (0.0, 1.0),
    "region_bay_of_bengal": (0.0, 1.0),
    "region_confluence_zone": (0.0, 1.0),
    "region_open_ocean": (0.0, 1.0),
}

def validate_datacube(
    zarr_path: str,
    output_plot_dir: str = None,
    expected_channels: int = 25
) -> Dict[str, Any]:
    """
    Validates the generated Zarr data cube against structural, numerical, and physical criteria.
    """
    logger.info(f"Running sanity checks on Zarr store: {zarr_path}")
    ds = open_datacube_zarr(zarr_path)
    cube = ds["cube"]

    report = {
        "status": "PASS",
        "checks": {},
        "warnings": [],
        "errors": []
    }

    # 1. Dimension and Shape check
    dims = cube.dims
    shape = cube.shape
    report["checks"]["dimensions"] = list(dims)
    report["checks"]["shape"] = list(shape)

    if dims != ("time", "channel", "lat", "lon"):
        msg = f"Unexpected dimensions: {dims}. Expected ('time', 'channel', 'lat', 'lon')"
        report["errors"].append(msg)
        report["status"] = "FAIL"

    if shape[1] != expected_channels:
        msg = f"Expected {expected_channels} channels, found {shape[1]}"
        report["errors"].append(msg)
        report["status"] = "FAIL"

    # 2. NaN and Inf check in ocean cells
    land_ocean_mask = cube.sel(channel="land_ocean_mask").values[0] # (H, W)
    ocean_indices = np.where(land_ocean_mask > 0.5)

    cube_vals = cube.values # (T, C, H, W)
    nan_count = np.isnan(cube_vals).sum()
    inf_count = np.isinf(cube_vals).sum()

    report["checks"]["total_nans"] = int(nan_count)
    report["checks"]["total_infs"] = int(inf_count)

    if nan_count > 0 or inf_count > 0:
        msg = f"Found {nan_count} NaNs and {inf_count} Infs in data cube."
        report["warnings"].append(msg)

    # 3. Channel Range Checks
    channel_names = list(ds["channel"].values)
    channel_stats = {}

    for ch_name in channel_names:
        ch_slice = cube.sel(channel=ch_name).values # (T, H, W)
        ocean_data = ch_slice[:, ocean_indices[0], ocean_indices[1]]

        c_min = float(np.nanmin(ocean_data)) if ocean_data.size > 0 else 0.0
        c_max = float(np.nanmax(ocean_data)) if ocean_data.size > 0 else 0.0
        c_mean = float(np.nanmean(ocean_data)) if ocean_data.size > 0 else 0.0

        channel_stats[ch_name] = {"min": c_min, "max": c_max, "mean": c_mean}

        if ch_name in PHYSICAL_BOUNDS:
            b_min, b_max = PHYSICAL_BOUNDS[ch_name]
            if c_min < b_min - 1e-3 or c_max > b_max + 1e-3:
                msg = f"Channel '{ch_name}' values [{c_min:.2f}, {c_max:.2f}] exceed plausible bounds [{b_min}, {b_max}]"
                report["warnings"].append(msg)

    report["checks"]["channel_stats"] = channel_stats

    # 4. Generate Verification Spot-Check Plot
    if output_plot_dir:
        os.makedirs(output_plot_dir, exist_ok=True)
        plot_file = os.path.join(output_plot_dir, "datacube_validation_overview.png")
        _generate_validation_plot(ds, plot_file)
        report["validation_plot"] = plot_file
        logger.info(f"Validation plot saved to: {plot_file}")

    if report["errors"]:
        report["status"] = "FAIL"
        logger.error(f"Sanity checks FAILED with errors: {report['errors']}")
    else:
        logger.info(f"Sanity checks PASSED ({len(report['warnings'])} warnings)")

    ds.close()
    return report

def _generate_validation_plot(ds: xr.Dataset, output_path: str):
    """Generates a 2x3 overview plot for key channels on day 0."""
    lat = ds["lat"].values
    lon = ds["lon"].values
    t_idx = 0
    date_str = str(ds["time"].values[t_idx])[:10]

    channels_to_plot = [
        ("sst", "Sea Surface Temp (K)", "coolwarm"),
        ("sss", "Sea Surface Salinity (psu)", "viridis"),
        ("bathymetry_log", "Log Bathymetry", "Blues_r"),
        ("land_ocean_mask", "Ocean/Land Mask", "gray"),
        ("region_arabian_sea", "Arabian Sea Membership", "plasma"),
        ("region_bay_of_bengal", "Bay of Bengal Membership", "plasma")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for idx, (ch_name, title, cmap) in enumerate(channels_to_plot):
        ax = axes[idx]
        if ch_name in ds["channel"].values:
            arr = ds["cube"].sel(channel=ch_name).isel(time=t_idx).values
            im = ax.pcolormesh(lon, lat, arr, cmap=cmap, shading="auto")
            fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        ax.set_title(f"{title}\n({date_str})", fontsize=10)
        ax.set_xlabel("Lon (°E)", fontsize=8)
        ax.set_ylabel("Lat (°N)", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate OceanEmbed Zarr Data Cube")
    parser.add_argument("--zarr", type=str, default="data/processed/toy_datacube.zarr", help="Path to Zarr store")
    parser.add_argument("--plot_dir", type=str, default="data/processed/validation_plots", help="Directory for plots")
    args = parser.parse_args()

    validate_datacube(args.zarr, output_plot_dir=args.plot_dir)
