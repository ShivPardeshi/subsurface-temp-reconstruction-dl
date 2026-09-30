"""Train Stage B Equalized Baseline for 2,000 steps on GCP."""

import sys
import time
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from src.training.train import train_model

def main():
    print("=" * 80)
    print("STARTING STAGE B EQUALIZED BASELINE TRAINING (2,000 STEPS)")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Clean existing checkpoints/baseline_2k if present to train fresh from step 0
    run_dir = Path("checkpoints/baseline_2k")
    if run_dir.exists():
        import shutil
        backup_dir = Path(f"checkpoints/baseline_2k_backup_{int(time.time())}")
        print(f"Backing up existing {run_dir} -> {backup_dir}")
        shutil.move(str(run_dir), str(backup_dir))

    t0 = time.time()
    results = train_model(
        config_path="src/training/config_registry/baseline_2k_config.yaml",
        override_steps=2000,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    duration = time.time() - t0

    print("=" * 80)
    print(f"COMPLETED STAGE B (2,000 STEPS) in {duration:.1f}s ({duration/60.0:.2f} mins)")
    print(f"Best Val RMSE: {results['best_val_rmse']:.4f}°C")
    print("=" * 80)

if __name__ == "__main__":
    main()
