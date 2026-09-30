"""
Fast Phase 2 Toy Mode Test Runner.
Executes complete Phase 2 pipeline over 7 days in a 10°x10° crop with visual diagnostic plots.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.assemble.build_training_dataset import Phase2DatasetBuilder
from src.utils.logging_config import setup_logger

logger = setup_logger("run_phase2_toy_mode")

def generate_phase2_diagnostic_plots(output_paths: dict, plot_dir: str):
    """Generates 2x2 diagnostic plot of Phase 2 features."""
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = os.path.join(plot_dir, "phase2_diagnostics_overview.png")

    ds_inputs = xr.open_zarr(output_paths["inputs_zarr"])
    ds_anomaly = xr.open_zarr(output_paths["anomaly_zarr"])
    ds_aux = xr.open_zarr(output_paths["auxiliary_zarr"])

    lat = ds_inputs["lat"].values
    lon = ds_inputs["lon"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. E - P Moisture Flux
    ax1 = axes[0, 0]
    ep_data = ds_inputs["inputs"].sel(channel="e_minus_p_flux").isel(time=0).values
    im1 = ax1.pcolormesh(lon, lat, ep_data, cmap="coolwarm", shading="auto")
    fig.colorbar(im1, ax=ax1, orientation="vertical", label="mm/day")
    ax1.set_title("E - P Moisture Flux (Day 0)\n[Pos = AS Salinifying, Neg = BoB Freshening]")

    # 2. Geostrophic Current Speed
    ax2 = axes[0, 1]
    ug = ds_inputs["inputs"].sel(channel="geostrophic_u").isel(time=0).values
    vg = ds_inputs["inputs"].sel(channel="geostrophic_v").isel(time=0).values
    speed_g = np.sqrt(ug**2 + vg**2)
    im2 = ax2.pcolormesh(lon, lat, speed_g, cmap="magma", shading="auto")
    fig.colorbar(im2, ax=ax2, orientation="vertical", label="m/s")
    ax2.set_title("Geostrophic Current Speed (Day 0)\n[Equatorially Tapered]")

    # 3. 100m Temperature Anomaly Target
    ax3 = axes[1, 0]
    anom_100m = ds_anomaly["anomaly"].sel(depth=100.0).isel(time=0).values
    im3 = ax3.pcolormesh(lon, lat, anom_100m, cmap="RdBu_r", shading="auto")
    fig.colorbar(im3, ax=ax3, orientation="vertical", label="°C")
    ax3.set_title("100m Subsurface Temperature Anomaly Target (Day 0)")

    # 4. Mixed Layer Depth Auxiliary Target
    ax4 = axes[1, 1]
    mld_data = ds_aux["auxiliary_targets"].sel(aux_target="mixed_layer_depth").isel(time=0).values
    im4 = ax4.pcolormesh(lon, lat, mld_data, cmap="Blues_r", shading="auto")
    fig.colorbar(im4, ax=ax4, orientation="vertical", label="meters")
    ax4.set_title("Mixed Layer Depth (de Boyer Montégut)")

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    logger.info(f"Phase 2 diagnostic overview plot saved to: {plot_file}")

def main():
    logger.info("=== STARTING PHASE 2 TOY MODE ===")
    test_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-10-01", "2024-10-07", freq="D")]
    toy_out_dir = os.path.join(REPO_ROOT, "data", "processed", "phase2_toy")

    builder = Phase2DatasetBuilder(toy_mode=True)
    out_paths = builder.build_all_components(
        date_list=test_dates,
        output_dir=toy_out_dir,
        training_end_date="2025-12-31"
    )

    plot_dir = os.path.join(REPO_ROOT, "data", "processed", "validation_plots")
    generate_phase2_diagnostic_plots(out_paths, plot_dir)

    logger.info("=== PHASE 2 TOY MODE COMPLETED SUCCESSFULLY: STATUS PASS ===")

if __name__ == "__main__":
    main()
