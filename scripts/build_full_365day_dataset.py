"""Build Full 365-Day 2025 Dataset for OceanEmbed Phase 1 & Phase 2.

Harmonizes all 25 surface satellite channels across 365 days (2025-01-01 to 2025-12-31),
fits 2-harmonic annual climatology across the full annual cycle, computes 15-depth
temperature anomalies and auxiliary targets (MLD, BLT, Salinity Max), and outputs
chunked Zarr stores.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_config import setup_logger
from src.data.harmonize.build_datacube import DataCubeBuilder
from src.assemble.build_training_dataset import Phase2DatasetBuilder
from src.utils.io_utils import save_datacube_to_zarr

logger = setup_logger("build_365day_dataset")

def main():
    date_list = [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-01-01", "2025-12-31", freq="D")]
    T = len(date_list)
    logger.info(f"Starting Full 365-Day 2025 Dataset Build: {T} days ({date_list[0]} to {date_list[-1]})")

    t0 = time.time()

    # Phase 1: Build & Save Harmonized Surface Data Cube (25 channels x 365 days)
    logger.info("=== STEP 1: Building Phase 1 Harmonized Surface Data Cube ===")
    builder_p1 = DataCubeBuilder(toy_mode=False)
    ds_datacube = builder_p1.build_cube_for_dates(date_list)

    p1_zarr_path = "data/processed/oceanembed_datacube.zarr"
    logger.info(f"Saving Phase 1 Data Cube to {p1_zarr_path}...")
    save_datacube_to_zarr(ds_datacube, p1_zarr_path, time_chunk_size=30, overwrite=True)
    logger.info(f"Phase 1 Data Cube saved successfully in {time.time() - t0:.1f}s.")

    # Phase 2: Assemble Training Inputs, Anomalies, Auxiliary Targets, Climatology
    logger.info("=== STEP 2: Assembling Phase 2 Training Dataset (Targets & Derived Channels) ===")
    t_p2 = time.time()
    builder_p2 = Phase2DatasetBuilder(toy_mode=False)
    output_dir = "data/processed/phase2_dataset"
    results = builder_p2.build_all_components(
        date_list=date_list,
        output_dir=output_dir,
        training_end_date="2025-12-31",
        precomputed_datacube=ds_datacube
    )


    logger.info(f"Phase 2 Assembly completed in {time.time() - t_p2:.1f}s.")
    logger.info(f"Full pipeline completed in {time.time() - t0:.1f}s. Outputs:")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

if __name__ == "__main__":
    main()
