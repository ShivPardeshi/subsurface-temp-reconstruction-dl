"""Genuine Cloud Training Orchestrator for Stages B, C, and D.

Executes:
1. Stage B (Baseline Full Architecture): Real training run (e.g. 2,000 steps).
2. Stage C (Ablation: No Region Conditioning): 1,000 steps.
3. Stage D (Ablation: No Depth Cascade): 1,000 steps.

Records exact timestamps, step counts, throughput, VRAM, and budget.
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.train import train_model
from src.utils.logging_config import setup_logger

logger = setup_logger("genuine_training_bcd")


def run_stage(stage_name: str, config_path: str, steps: int, clean: bool = True):
    print("=" * 80)
    print(f"STARTING GENUINE TRAINING: {stage_name.upper()} ({steps} STEPS)")
    print(f"Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print("=" * 80)

    # If clean run requested, remove existing checkpoints for clean step 0 start
    run_dir = Path(f"checkpoints/{stage_name}")
    if clean and run_dir.exists():
        backup_dir = Path(f"checkpoints/{stage_name}_backup_{int(time.time())}")
        print(f"Backing up existing {run_dir} -> {backup_dir}")
        shutil.move(str(run_dir), str(backup_dir))

    t_start = time.time()
    dt_start = datetime.datetime.now(datetime.timezone.utc)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results = train_model(
        config_path=config_path,
        override_steps=steps,
        device=device,
    )
    t_end = time.time()
    dt_end = datetime.datetime.now(datetime.timezone.utc)
    wall_duration = t_end - t_start

    summary = {
        "stage": stage_name,
        "config": config_path,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "start_time_utc": dt_start.isoformat(),
        "end_time_utc": dt_end.isoformat(),
        "wall_clock_seconds": round(wall_duration, 2),
        "wall_clock_minutes": round(wall_duration / 60.0, 2),
        "start_step": 0,
        "end_step": results["total_steps"],
        "best_val_rmse": round(results["best_val_rmse"], 4),
        "budget": results.get("budget", {}),
    }

    print("=" * 80)
    print(f"COMPLETED {stage_name.upper()}: {results['total_steps']} steps in {wall_duration:.1f}s")
    print(f"Best Val RMSE: {results['best_val_rmse']:.4f}°C")
    print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Genuine Training Runner for Stages B, C, D")
    parser.add_argument("--steps-b", type=int, default=10000, help="Steps for Stage B baseline")
    parser.add_argument("--steps-c", type=int, default=2000, help="Steps for Stage C ablation")
    parser.add_argument("--steps-d", type=int, default=2000, help="Steps for Stage D ablation")
    parser.add_argument("--output-json", type=str, default="logs/genuine_training_summary_bcd.json")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    all_summaries = {}

    # Stage B
    all_summaries["stage_b"] = run_stage(
        stage_name="baseline",
        config_path="src/training/config_registry/baseline_config.yaml",
        steps=args.steps_b,
    )

    # Stage C
    all_summaries["stage_c"] = run_stage(
        stage_name="ablation_no_region",
        config_path="src/training/config_registry/no_region_ablation.yaml",
        steps=args.steps_c,
    )

    # Stage D
    all_summaries["stage_d"] = run_stage(
        stage_name="ablation_no_cascade",
        config_path="src/training/config_registry/no_cascade_ablation.yaml",
        steps=args.steps_d,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "#" * 80)
    print("ALL 3 TRAINING STAGES FINISHED SUCCESSFULLY!")
    print(f"Consolidated results saved to: {args.output_json}")
    print("#" * 80)


if __name__ == "__main__":
    main()
