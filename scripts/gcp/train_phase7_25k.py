"""Train Phase 7 Dampened Scaling & Strengthened Variance Loss for 25,000 steps on GCP."""

import sys
import time
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

import torch
import numpy as np
from src.training.train import train_model


def set_seed(seed: int = 42):
    """Set global random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    seed = 42
    print("=" * 80)
    print(f"STARTING PHASE 7 DAMPENED SCALING & VARIANCE LOSS RUN (25,000 STEPS, SEED {seed})")
    print("KEY ARCHITECTURAL ADVANCEMENTS:")
    print("  1. Dampened Target Scaling (p=0.5): sigma_d^0.5 in [0.477, 1.058]")
    print("     - Eliminates 38% of physical error amplification in the thermocline")
    print("     - Retains high SNR in the deep abyss (variance = 0.228)")
    print("  2. Strengthened Physical Variance Loss Weighting: beta = 1.50 (19.6x priority)")
    print("  3. Retains all 16 proven foundational and physical modules:")
    print("     - 74 spatial channels (SSHA + gradient magnitude)")
    print("     - 14 conditioning dimensions (lapse rate, dz, auxiliary predictions)")
    print("     - Tri-stratum mini-column sampling (surface, thermocline, deep)")
    print("     - Sobel horizontal gradient consistency loss")
    print("     - Active auxiliary head feedback (MLD, BLT, Salinity Max depth)")
    print("     - 25-step quadratic DDIM reverse trajectory")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    run_dir = Path("checkpoints/phase7_dampened_scaling_thermocline_25k")
    if run_dir.exists():
        backup_dir = Path(f"checkpoints/phase7_dampened_scaling_thermocline_25k_backup_{int(time.time())}")
        print(f"Backing up existing {run_dir} -> {backup_dir}")
        shutil.move(str(run_dir), str(backup_dir))

    t0 = time.time()
    results = train_model(
        config_path="src/training/config_registry/phase7_dampened_scaling_thermocline_25k.yaml",
        override_steps=25000,
        device=device,
        resume=False,
    )
    dur = time.time() - t0
    print(f"\nCOMPLETED PHASE 7 RUN (25,000 STEPS) in {dur:.1f}s ({dur/60.0:.2f} mins)")
    print(f"Best Val RMSE: {results['best_val_rmse']:.4f}°C")
    print(f"Run Directory: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
