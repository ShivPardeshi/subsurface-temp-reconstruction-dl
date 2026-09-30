"""Remote GCP Environment and GPU Verification Script.

Executes on the GCP L4 VM to confirm:
1. NVIDIA driver visibility via nvidia-smi.
2. PyTorch CUDA capability and GPU device name (NVIDIA L4).
3. Forward and backward passes on CUDA device.
4. Overfit test on GPU.
"""

import sys
import subprocess
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from src.verification.shape_checks import run_full_pipeline_shape_check
from src.verification.overfit_tiny_batch import run_overfit_test
from src.utils.logging_config import get_logger

logger = get_logger("verify_gcp_env")


def check_system_gpu() -> bool:
    logger.info("=== STEP 1: Checking NVIDIA Driver and GPU Hardware ===")
    try:
        res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info(f"nvidia-smi output:\n{res.stdout}")
    except Exception as e:
        logger.warning(f"Could not run nvidia-smi: {e}")

    if not torch.cuda.is_available():
        logger.error("PyTorch reports CUDA is NOT available!")
        return False

    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"PyTorch CUDA Available: True | Device Count: {gpu_count}")
    logger.info(f"GPU Name: {gpu_name} | VRAM: {vram_gb:.2f} GB")

    return True


def verify_all_on_cuda():
    logger.info("=== STEP 2: Running Full Pipeline Shape & Gradient Pass on CUDA ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    success_shape = run_full_pipeline_shape_check(batch_size=2, grid_h=112, grid_w=240, device=device)

    logger.info("=== STEP 3: Running Overfit Verification on CUDA ===")
    success_overfit, _ = run_overfit_test(num_steps=100, lr=2e-3, device=device)

    if success_shape and success_overfit:
        logger.info("================================================================")
        logger.info("=== GCP ENVIRONMENT VERIFICATION COMPLETE: ALL CHECKS PASS ===")
        logger.info(">>> PLEASE STOP THE VM NOW TO AVOID CHARGES:                <<<")
        logger.info(">>> gcloud compute instances stop <INSTANCE_NAME> --zone=<ZONE> <<<")
        logger.info("================================================================")
        return True
    else:
        logger.error("GCP Environment verification failed.")
        return False


if __name__ == "__main__":
    if check_system_gpu():
        success = verify_all_on_cuda()
        sys.exit(0 if success else 1)
    else:
        sys.exit(1)
