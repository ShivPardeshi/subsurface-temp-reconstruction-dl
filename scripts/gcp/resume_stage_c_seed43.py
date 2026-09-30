"""Resume Stage C (Seed 43) from last checkpoint and train to 20,000 steps."""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
from src.training.train import train_model


def set_seed(seed: int = 43):
    """Set global random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    seed = 43
    print("=" * 80)
    print(f"RESUMING STAGE C ABLATION (SEED {seed}) TO 20,000 STEPS")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    ckpt_path = Path("checkpoints/ablation_no_region_20k_seed43/last_checkpoint.pt")
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    current_step = ckpt.get("step", 0)
    print(f"Detected checkpoint at step {current_step} / 20000 steps.")

    t0 = time.time()
    results = train_model(
        config_path="src/training/config_registry/ablation_no_region_20k_seed43_config.yaml",
        override_steps=20000,
        device=device,
        resume=True,
    )
    dur = time.time() - t0
    print(f"\nCOMPLETED STAGE C SEED {seed} (20,000 STEPS) in {dur:.1f}s ({dur/60.0:.2f} mins)")
    print(f"Final Validation RMSE (Stage C Seed {seed}): {results['best_val_rmse']:.4f}°C")
    print("=" * 80)


if __name__ == "__main__":
    main()
