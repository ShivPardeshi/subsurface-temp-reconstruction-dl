"""Train Stage B and Stage C from scratch for 20,000 steps with Seed 43 on GCP."""

import sys
import time
import os
import shutil
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
    print(f"STARTING TRACK A SECOND-SEED VALIDATION (SEED = {seed})")
    print("20,000 STEPS FROM SCRATCH: STAGE B & STAGE C")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # 1. Train Stage B (Baseline: 20,000 Steps from Scratch, Seed 43)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print(f"PHASE 1: STAGE B (BASELINE SEED {seed}) — STEP 0 TO 20,000 STEPS")
    print("#" * 80)
    set_seed(seed)

    run_b_dir = Path("checkpoints/baseline_20k_seed43")
    if run_b_dir.exists():
        backup_b = Path(f"checkpoints/baseline_20k_seed43_backup_{int(time.time())}")
        print(f"Backing up existing {run_b_dir} -> {backup_b}")
        shutil.move(str(run_b_dir), str(backup_b))

    t0_b = time.time()
    results_b = train_model(
        config_path="src/training/config_registry/baseline_20k_seed43_config.yaml",
        override_steps=20000,
        device=device,
        resume=False,
    )
    dur_b = time.time() - t0_b
    print(f"\nCOMPLETED STAGE B SEED {seed} (20,000 STEPS) in {dur_b:.1f}s ({dur_b/60.0:.2f} mins)")
    print(f"Best Val RMSE (Stage B Seed {seed}): {results_b['best_val_rmse']:.4f}°C")

    # -------------------------------------------------------------------------
    # 2. Train Stage C (No Region: 20,000 Steps from Scratch, Seed 43)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print(f"PHASE 2: STAGE C (NO REGION SEED {seed}) — STEP 0 TO 20,000 STEPS")
    print("#" * 80)
    set_seed(seed)

    run_c_dir = Path("checkpoints/ablation_no_region_20k_seed43")
    if run_c_dir.exists():
        backup_c = Path(f"checkpoints/ablation_no_region_20k_seed43_backup_{int(time.time())}")
        print(f"Backing up existing {run_c_dir} -> {backup_c}")
        shutil.move(str(run_c_dir), str(backup_c))

    t0_c = time.time()
    results_c = train_model(
        config_path="src/training/config_registry/ablation_no_region_20k_seed43_config.yaml",
        override_steps=20000,
        device=device,
        resume=False,
    )
    dur_c = time.time() - t0_c
    print(f"\nCOMPLETED STAGE C SEED {seed} (20,000 STEPS) in {dur_c:.1f}s ({dur_c/60.0:.2f} mins)")
    print(f"Best Val RMSE (Stage C Seed {seed}): {results_c['best_val_rmse']:.4f}°C")

    # -------------------------------------------------------------------------
    # Summary & Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TRACK A SECOND-SEED (SEED {seed}) VALIDATION COMPLETE!")
    print(f"Stage B Best Val RMSE: {results_b['best_val_rmse']:.4f}°C (Duration: {dur_b/60.0:.2f}m)")
    print(f"Stage C Best Val RMSE: {results_c['best_val_rmse']:.4f}°C (Duration: {dur_c/60.0:.2f}m)")
    delta = results_c['best_val_rmse'] - results_b['best_val_rmse']
    pct = (delta / results_c['best_val_rmse']) * 100.0
    print(f"Delta (Stage C - Stage B): {delta:+.4f}°C (Advantage: {pct:+.2f}%)")
    print(f"Total Combined GPU Time: {(dur_b + dur_c)/60.0:.2f} mins")
    print("=" * 80)


if __name__ == "__main__":
    main()
