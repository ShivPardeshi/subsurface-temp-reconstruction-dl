"""Train Phase 8 Zone-Adaptive Scaling & Multi-Domain Convergence for 15,000 steps on GCP V100."""

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
    print(f"STARTING PHASE 8 ZONE-ADAPTIVE 15k FINE-TUNING RUN (SEED {seed})")
    print("KEY ARCHITECTURAL ADVANCEMENTS:")
    print("  1. Zone-Adaptive Standardization:")
    print("     - Surface (0-30m):     p=0.75 -> restores surface skill")
    print("     - Thermocline (50-200m): p=0.50 -> IDENTICAL to Phase 7, gains 100% preserved")
    print("     - Transition (300m):    p=0.75 -> gradual smooth gradient bridge")
    print("     - Deep Abyss (500-1000m): p=1.00 -> full standardization, restores Phase 6 deep accuracy")
    print("  2. Zone-Adaptive Physical Variance Loss Weighting (beta(d)):")
    print("     - Surface: beta=1.00")
    print("     - Thermocline: beta=1.50 (exact Phase 7 priority)")
    print("     - Deep: beta=0.50 (eliminates deep-ocean gradient attenuation)")
    print("  3. Initialization: Warm start from Phase 7 Best Checkpoint (25,000 steps)")
    print("  4. Platform: NVIDIA V100 Tensor Cores (fp16 mixed precision)")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    run_dir = Path("checkpoints/phase8_zone_adaptive_15k")
    if run_dir.exists():
        backup_dir = Path(f"checkpoints/phase8_zone_adaptive_15k_backup_{int(time.time())}")
        print(f"Backing up existing {run_dir} -> {backup_dir}")
        shutil.move(str(run_dir), str(backup_dir))

    t0 = time.time()
    results = train_model(
        config_path="src/training/config_registry/phase8_zone_adaptive_15k.yaml",
        override_steps=15000,
        device=device,
        resume=False,
    )
    dur = time.time() - t0
    print(f"\nCOMPLETED PHASE 8 RUN (15,000 STEPS) in {dur:.1f}s ({dur/60.0:.2f} mins)")
    print(f"Best Val RMSE: {results['best_val_rmse']:.4f}°C")
    print(f"Run Directory: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
