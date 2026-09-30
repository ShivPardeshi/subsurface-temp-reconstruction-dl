"""
Fast, small-scale end-to-end test runner for OceanEmbed Phase 1 pipeline (Toy Mode).
Processes 7 days over a 10°x10° sub-crop (Bay of Bengal focus) in seconds.
"""

import os
import sys
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.harmonize.build_datacube import DataCubeBuilder
from src.utils.io_utils import save_datacube_to_zarr
from src.data.validation.sanity_checks import validate_datacube
from src.utils.logging_config import setup_logger

logger = setup_logger("run_toy_mode")

def main():
    logger.info("=== STARTING OCEANEMBED PHASE 1 TOY MODE ===")

    # 7-day test window (e.g. Oct 1 - Oct 7, 2024 or 2025)
    test_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-10-01", "2024-10-07", freq="D")]

    # 1. Initialize Builder in Toy Mode (10° x 10° sub-crop: 12°N-22°N, 82°E-92°E)
    builder = DataCubeBuilder(toy_mode=True)

    # 2. Build Data Cube
    ds = builder.build_cube_for_dates(test_dates)
    logger.info(f"Assembled Toy Data Cube: shape={ds['cube'].shape}, dims={dict(ds.sizes)}")

    # 3. Save to Zarr Store
    output_zarr = os.path.join(REPO_ROOT, "data", "processed", "toy_datacube.zarr")
    save_datacube_to_zarr(ds, output_zarr, time_chunk_size=7, overwrite=True)
    logger.info(f"Saved Toy Zarr Store to: {output_zarr}")

    # 4. Run Automated Sanity Checks & Visual Validation
    plot_dir = os.path.join(REPO_ROOT, "data", "processed", "validation_plots")
    report = validate_datacube(output_zarr, output_plot_dir=plot_dir)

    logger.info(f"=== TOY MODE COMPLETED: STATUS {report['status']} ===")
    if report["status"] != "PASS":
        sys.exit(1)

if __name__ == "__main__":
    main()
