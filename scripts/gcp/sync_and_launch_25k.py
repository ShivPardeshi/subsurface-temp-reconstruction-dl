"""Sync rectified files to GCP VM and launch the 25k training run in tmux."""

import os
import sys
import subprocess
import tarfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTANCE_NAME = "oceanembed-l4-training"
ZONE = "us-central1-a"
REMOTE_DIR = "/home/123ta/oceanembed"

FILES_TO_SYNC = [
    "data/processed/anomaly_depth_scales.json",
    "src/training/dataset.py",
    "src/sampling/depth_cascade.py",
    "src/training/losses.py",
    "src/sampling/ddim_sampler.py",
    "src/training/train.py",
    "src/training/config_registry/phase5_rectified_pure_diffusion_25k.yaml",
    "scripts/gcp/train_rectified_25k.py",
]

def create_bundle(tar_path: Path):
    print(f"Creating bundle: {tar_path}...")
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel_path in FILES_TO_SYNC:
            full_path = REPO_ROOT / rel_path
            if not full_path.exists():
                raise FileNotFoundError(f"Missing required file: {full_path}")
            print(f"  Adding: {rel_path}")
            tar.add(full_path, arcname=rel_path)
    print(f"Bundle created successfully ({tar_path.stat().st_size / 1024:.1f} KB).")

def main():
    bundle_name = "rectified_update.tar.gz"
    bundle_path = REPO_ROOT / bundle_name
    create_bundle(bundle_path)

    print(f"\nUploading {bundle_name} to {INSTANCE_NAME}...")
    scp_cmd = f"gcloud compute scp {bundle_path} {INSTANCE_NAME}:{bundle_name} --zone={ZONE}"
    res = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"SCP failed:\n{res.stderr}")
        sys.exit(1)
    print("Upload complete.")

    print("\nExtracting bundle on VM...")
    extract_cmd = f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="tar -xzf ~/{bundle_name} -C {REMOTE_DIR}/ && rm ~/{bundle_name}"'
    res = subprocess.run(extract_cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Extraction failed:\n{res.stderr}")
        sys.exit(1)
    print("Extraction complete.")

    # Clean local tarball
    if bundle_path.exists():
        bundle_path.unlink()

    print("\nUploading run_rectified_25k.sh to VM...")
    sh_local = REPO_ROOT / "scripts/gcp/run_rectified_25k.sh"
    scp_sh = f"gcloud compute scp {sh_local} {INSTANCE_NAME}:{REMOTE_DIR}/run_rectified_25k.sh --zone={ZONE}"
    res = subprocess.run(scp_sh, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"SCP of runner script failed:\n{res.stderr}")
        sys.exit(1)

    print("\nLaunching training via run_rectified_25k.sh...")
    launch_cmd = f"gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command=\"bash {REMOTE_DIR}/run_rectified_25k.sh\""
    res = subprocess.run(launch_cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Launch failed:\n{res.stderr}")
        sys.exit(1)
    print(res.stdout)

    time.sleep(3)

    print("\nChecking training status...")
    check_cmd = f"gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command=\"tmux ls && sleep 2 && head -n 40 {REMOTE_DIR}/train_rectified_25k.log || true\""
    res = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

if __name__ == "__main__":
    main()
