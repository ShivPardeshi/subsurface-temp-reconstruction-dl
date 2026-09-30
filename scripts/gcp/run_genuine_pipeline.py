"""Genuine Training and Evaluation Pipeline for GCP L4 VM.

Executes:
1. Verification of 365-day dataset in data/processed/phase2_dataset.
2. Genuine Retraining:
   - Stage B (Baseline): 2,000 steps
   - Stage C (Ablation: No Region Conditioning): 1,000 steps
   - Stage D (Ablation: No Depth Cascade): 1,000 steps
3. Genuine Multi-Seasonal Evaluation on Held-Out Test Data & All 7 Priority Zones.
4. Summary Report Generation.
"""

import sys
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import zarr

def main():
    print("=" * 80)
    print("STARTING GENUINE RETRAINING & EVALUATION PIPELINE ON GCP")
    print("=" * 80)

    # Check GPU
    assert torch.cuda.is_available(), "CUDA is not available!"
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Check dataset
    in_z = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr")
    anom_z = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr")
    print(f"Verified Inputs Shape: {in_z['inputs'].shape}")
    print(f"Verified Anomaly Shape: {anom_z['anomaly'].shape}")
    assert in_z["inputs"].shape[0] == 365, "Expected 365 days of inputs!"
    assert anom_z["anomaly"].shape[0] == 365, "Expected 365 days of anomalies!"

    # 1. Run Stages B, C, D
    print("\n>>> LAUNCHING GENUINE RETRAINING: STAGES B, C, D <<<")
    t0 = time.time()
    cmd_train = [
        sys.executable, "scripts/run_genuine_stages_bcd.py",
        "--steps-b", "10000",
        "--steps-c", "2000",
        "--steps-d", "2000"
    ]
    ret_train = subprocess.run(cmd_train)
    if ret_train.returncode != 0:
        print(f"Training failed with code {ret_train.returncode}!")
        sys.exit(ret_train.returncode)
    print(f"Training completed in {time.time() - t0:.1f}s")

    # 2. Run Genuine Evaluation
    print("\n>>> LAUNCHING GENUINE EVALUATION <<<")
    t1 = time.time()
    cmd_eval = [
        sys.executable, "scripts/run_genuine_evaluation.py",
        "--device", "cuda:0"
    ]
    ret_eval = subprocess.run(cmd_eval)
    if ret_eval.returncode != 0:
        print(f"Evaluation failed with code {ret_eval.returncode}!")
        sys.exit(ret_eval.returncode)
    # 3. Format Evaluation Report
    print("\n>>> FORMATTING FINAL EVALUATION REPORT <<<")
    cmd_rep = [sys.executable, "scripts/format_final_evaluation_report.py"]
    ret_rep = subprocess.run(cmd_rep)
    if ret_rep.returncode != 0:
        print(f"Report generation failed with code {ret_rep.returncode}!")
        sys.exit(ret_rep.returncode)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE! ALL CHECKPOINTS AND REPORTS GENERATED.")
    print("=" * 80)

if __name__ == "__main__":
    main()
