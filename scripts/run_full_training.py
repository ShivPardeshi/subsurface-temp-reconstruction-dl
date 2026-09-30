"""Stages B–D: Full Baseline and Ablation Training Runner.

Executes:
- Stage B: Baseline full architecture (`baseline_config.yaml`)
- Stage C: Region-conditioning ablation (`no_region_ablation.yaml`)
- Stage D: Depth-cascade ablation (`no_cascade_ablation.yaml`)
"""

from typing import Dict, Any
from pathlib import Path
import os
import sys
import argparse
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.train import train_model
from src.utils.logging_config import get_logger

logger = get_logger("run_full_training")

CONFIG_MAP = {
    "baseline": "src/training/config_registry/baseline_config.yaml",
    "no_region": "src/training/config_registry/no_region_ablation.yaml",
    "no_cascade": "src/training/config_registry/no_cascade_ablation.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="Run OceanEmbed Stage B/C/D Full Training")
    parser.add_argument(
        "--stage",
        type=str,
        default="baseline",
        choices=["baseline", "no_region", "no_cascade"],
        help="Training stage or ablation to execute",
    )
    parser.add_argument("--config", type=str, default=None, help="Explicit config path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--steps", type=int, default=None, help="Override step count")
    args = parser.parse_args()

    config_path = args.config or CONFIG_MAP[args.stage]
    logger.info(f"Executing Full Training Stage: {args.stage} using {config_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = train_model(
        config_path=config_path,
        override_epochs=args.epochs,
        override_steps=args.steps,
        device=device,
    )
    logger.info(f"Completed run: {results['run_name']} | Best Val RMSE: {results['best_val_rmse']:.4f}")


if __name__ == "__main__":
    main()
