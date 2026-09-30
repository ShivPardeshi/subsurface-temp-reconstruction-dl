"""
Full-domain orchestration pipeline for OceanEmbed Phase 2.
"""

import os
import sys
import argparse
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.assemble.build_training_dataset import Phase2DatasetBuilder
from src.utils.logging_config import setup_logger
from scripts.run_phase2_toy_mode import generate_phase2_diagnostic_plots

logger = setup_logger("phase2_pipeline")

def main():
    parser = argparse.ArgumentParser(description="Run OceanEmbed Phase 2 Feature Engineering Pipeline")
    parser.add_argument("--start_date", type=str, default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2025-01-07", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_dir", type=str, default="data/processed/phase2_dataset", help="Output directory")
    parser.add_argument("--training_cutoff", type=str, default="2025-12-31", help="Training period end date")
    parser.add_argument("--plot_dir", type=str, default="data/processed/validation_plots", help="Diagnostics plot dir")
    args = parser.parse_args()

    logger.info(f"=== STARTING FULL PHASE 2 PIPELINE: {args.start_date} to {args.end_date} ===")

    date_list = [d.strftime("%Y-%m-%d") for d in pd.date_range(args.start_date, args.end_date, freq="D")]
    out_dir = os.path.join(REPO_ROOT, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    builder = Phase2DatasetBuilder(toy_mode=False)
    out_paths = builder.build_all_components(
        date_list=date_list,
        output_dir=out_dir,
        training_end_date=args.training_cutoff
    )

    plot_dir = os.path.join(REPO_ROOT, args.plot_dir) if not os.path.isabs(args.plot_dir) else args.plot_dir
    generate_phase2_diagnostic_plots(out_paths, plot_dir)

    logger.info("=== FULL PHASE 2 PIPELINE COMPLETED: STATUS PASS ===")

if __name__ == "__main__":
    main()
