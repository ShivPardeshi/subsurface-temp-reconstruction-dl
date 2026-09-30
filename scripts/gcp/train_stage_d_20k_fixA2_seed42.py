"""Train Stage D (No Depth Cascade) from scratch for 20,000 steps with Seed 42 and Fix A2 on GCP."""

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
    print(f"STARTING SEED {seed} STAGE D RETRAINING WITH ALL FIXES ACTIVE")
    print("20,000 STEPS FROM SCRATCH: STAGE D (ABLATION: NO DEPTH CASCADE)")
    print("Fixes Active: Fix A1 (eta=0.3), Fix A2 (Loss Norm), Part B (20-200m Thermocline)")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    run_d_dir = Path("checkpoints/ablation_no_cascade_20k_fixA2_seed42")
    if run_d_dir.exists():
        backup_d = Path(f"checkpoints/ablation_no_cascade_20k_fixA2_seed42_backup_{int(time.time())}")
        print(f"Backing up existing {run_d_dir} -> {backup_d}")
        shutil.move(str(run_d_dir), str(backup_d))

    t0_d = time.time()
    results_d = train_model(
        config_path="src/training/config_registry/ablation_no_cascade_20k_fixA2_seed42_config.yaml",
        override_steps=20000,
        device=device,
        resume=False,
    )
    dur_d = time.time() - t0_d
    print(f"\nCOMPLETED STAGE D SEED {seed} FIX A2 (20,000 STEPS) in {dur_d:.1f}s ({dur_d/60.0:.2f} mins)")
    print(f"Best Val RMSE (Stage D Seed {seed} Fix A2): {results_d['best_val_rmse']:.4f}°C")

    # Summary & Comparison against Stage B (Baseline: 0.5426°C)
    stage_b_rmse = 0.5426
    stage_d_rmse = results_d['best_val_rmse']
    delta_cascade = stage_d_rmse - stage_b_rmse
    pct_cascade = (delta_cascade / stage_d_rmse) * 100.0

    print("\n" + "=" * 80)
    print(f"STAGE D (NO DEPTH CASCADE) 20,000-STEP RUN COMPLETE!")
    print(f"Stage B (Baseline with Cascade): {stage_b_rmse:.4f}°C")
    print(f"Stage D (Ablation without Cascade): {stage_d_rmse:.4f}°C (Duration: {dur_d/60.0:.2f}m)")
    print(f"Depth Cascade Advantage (Stage D - Stage B): {delta_cascade:+.4f}°C (Improvement: {pct_cascade:+.2f}%)")
    print(f"Total GPU Time: {dur_d/60.0:.2f} mins")
    print("=" * 80)


if __name__ == "__main__":
    main()
