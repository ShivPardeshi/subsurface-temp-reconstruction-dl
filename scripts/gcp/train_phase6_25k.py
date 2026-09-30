"""Train Phase 6 Thermocline Breakthrough from scratch for 25,000 steps on GCP with all 16 bottlenecks resolved."""

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
    print(f"STARTING PHASE 6 THERMOCLINE BREAKTHROUGH 25,000-STEP TRAINING RUN (SEED {seed})")
    print("ALL 16 ARCHITECTURAL BOTTLENECKS RESOLVED:")
    print("  1-8: Target norm, 15-depth val, cascade jitter, depth emb, Min-SNR, column training, strat loss, DDIM quad")
    print("  9. Physical Variance Loss Weighting w_var(d) = (sigma_d / bar_sigma)^1.25")
    print("  10. Tri-Stratum Balanced Mini-Column Sampling (Surface, Thermocline Core, Deep)")
    print("  11. Direct SSHA and Gradient Magnitude Injection into spatial_cond (72 channels)")
    print("  12. Vertical Climatological Lapse Rate Gamma_clim(d) and Layer Thickness dz Conditioning")
    print("  13. Sobel Horizontal Gradient Consistency Loss on x0_hat")
    print("  14. Active Auxiliary Head Conditioning (MLD, BLT, Salinity Max depth) into Denoiser")
    print("  15. Depth-Adaptive Cascade Training Jitter")
    print("  16. 25-Step Quadratic DDIM Reverse Sampling Trajectory")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    run_dir = Path("checkpoints/phase6_thermocline_breakthrough_25k")
    if run_dir.exists():
        backup_dir = Path(f"checkpoints/phase6_thermocline_breakthrough_25k_backup_{int(time.time())}")
        print(f"Backing up existing {run_dir} -> {backup_dir}")
        shutil.move(str(run_dir), str(backup_dir))

    t0 = time.time()
    results = train_model(
        config_path="src/training/config_registry/phase6_thermocline_breakthrough_25k.yaml",
        override_steps=25000,
        device=device,
        resume=False,
    )
    dur = time.time() - t0
    print(f"\nCOMPLETED PHASE 6 RUN (25,000 STEPS) in {dur:.1f}s ({dur/60.0:.2f} mins)")
    print(f"Best Val RMSE: {results['best_val_rmse']:.4f}°C")
    print(f"Run Directory: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
