"""
Full orchestration entry point for OceanEmbed Phase 1 Ingestion and Harmonization Pipeline.
"""

import os
import sys
import argparse
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.harmonize.build_datacube import DataCubeBuilder
from src.utils.io_utils import save_datacube_to_zarr
from src.data.validation.sanity_checks import validate_datacube
from src.utils.logging_config import setup_logger

logger = setup_logger("phase1_pipeline")

def main():
    parser = argparse.ArgumentParser(description="Run OceanEmbed Phase 1 Data Ingestion Pipeline")
    parser.add_argument("--start_date", type=str, default="2024-10-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2024-10-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_zarr", type=str, default="data/processed/oceanembed_datacube.zarr", help="Output Zarr store path")
    parser.add_argument("--plot_dir", type=str, default="data/processed/validation_plots", help="Validation plot directory")
    parser.add_argument("--chunk_size", type=int, default=30, help="Time dimension chunk size")
    args = parser.parse_args()

    logger.info(f"=== STARTING PHASE 1 PIPELINE: {args.start_date} to {args.end_date} ===")

    date_list = [d.strftime("%Y-%m-%d") for d in pd.date_range(args.start_date, args.end_date, freq="D")]

    # 1. Initialize Full Domain Builder (112 x 240 grid)
    builder = DataCubeBuilder(toy_mode=False)

    # 2. Build Data Cube
    ds = builder.build_cube_for_dates(date_list)
    logger.info(f"Data Cube Assembled: shape={ds['cube'].shape}, dims={dict(ds.sizes)}")

    # 3. Save to Chunked Zarr Store
    zarr_out = os.path.join(REPO_ROOT, args.output_zarr) if not os.path.isabs(args.output_zarr) else args.output_zarr
    save_datacube_to_zarr(ds, zarr_out, time_chunk_size=args.chunk_size, overwrite=True)
    logger.info(f"Saved Zarr Store to: {zarr_out}")

    # 4. Validate
    plot_dir = os.path.join(REPO_ROOT, args.plot_dir) if not os.path.isabs(args.plot_dir) else args.plot_dir
    report = validate_datacube(zarr_out, output_plot_dir=plot_dir)

    logger.info(f"=== PHASE 1 PIPELINE COMPLETED: STATUS {report['status']} ===")
    if report["status"] != "PASS":
        sys.exit(1)

if __name__ == "__main__":
    main()
