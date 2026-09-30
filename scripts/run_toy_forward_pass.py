"""Script: Run Toy Forward & Backward Pass Verification.

Executes end-to-end forward pass, loss calculation, backward pass,
and gradient stability checks across all 5 architectural stages.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.verification.shape_checks import run_full_pipeline_shape_check
from src.utils.logging_config import get_logger

logger = get_logger("run_toy_forward_pass")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    success = run_full_pipeline_shape_check(batch_size=2, grid_h=40, grid_w=40, device=device)
    if success:
        logger.info("=== FORWARD/BACKWARD STABILITY PASS: STATUS OK ===")
        sys.exit(0)
    else:
        logger.error("=== FORWARD/BACKWARD STABILITY PASS: STATUS FAIL ===")
        sys.exit(1)
