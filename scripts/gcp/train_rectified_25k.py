"""Train Rectified Pure Diffusion from scratch for 25,000 steps on GCP with all 8 bottlenecks resolved."""

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
    print(f"STARTING RECTIFIED PURE DIFFUSION 25,000-STEP TRAINING RUN (SEED {seed})")
    print("ALL 8 ARCHITECTURAL BOTTLENECKS RECTIFIED:")
    print("  1. Target Normalization (standardized anomalies via anomaly_depth_scales.json)")
    print("  2. Full 15-Depth Validation Evaluation & Checkpoint Selection (0-1000m)")
    print("  3. Realistic Cascade Conditioning Jitter (sigma=0.35) & 20% Dropout")
    print("  4. Explicit Climatological Background Temperature Conditioning (9-dim)")
    print("  5. Min-SNR-gamma Loss Weighting (gamma=5.0)")
    print("  6. Multi-Depth Mini-Column Training (consecutive & stratified pairs)")
    print("  7. Vertical Stratification Gradient Loss & Static Stability Penalty")
    print("  8. Quadratic DDIM Sampling Trajectory")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    run_dir = Path("checkpoints/phase5_rectified_pure_diffusion_25k")
    if run_dir.exists():
        backup_dir = Path(f"checkpoints/phase5_rectified_pure_diffusion_25k_backup_{int(time.time())}")
        print(f"Backing up existing {run_dir} -> {backup_dir}")
        shutil.move(str(run_dir), str(backup_dir))

    t0 = time.time()
    results = train_model(
        config_path="src/training/config_registry/phase5_rectified_pure_diffusion_25k.yaml",
        override_steps=25000,
        device=device,
        resume=False,
    )
    dur = time.time() - t0
    print(f"\nCOMPLETED RECTIFIED PURE DIFFUSION RUN (25,000 STEPS) in {dur:.1f}s ({dur/60.0:.2f} mins)")
    print(f"Best Val RMSE: {results['best_val_rmse']:.4f}°C")
    print(f"Run Directory: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
