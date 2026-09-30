"""Script: Run Overfit Tiny Batch Verification Test.

Trains the complete network architecture on a tiny batch of real samples
to confirm loss convergence to near-zero.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.verification.overfit_tiny_batch import run_overfit_test
from src.utils.logging_config import get_logger

logger = get_logger("run_overfit_test")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    passed, loss_history = run_overfit_test(num_steps=150, lr=2e-3, device=device)
    if passed:
        logger.info("=== OVERFIT TEST: STATUS PASS ===")
        sys.exit(0)
    else:
        logger.error("=== OVERFIT TEST: STATUS FAIL ===")
        sys.exit(1)
