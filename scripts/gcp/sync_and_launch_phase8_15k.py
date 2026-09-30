"""Sync Phase 8 files, datasets, and checkpoints to GCP V100 VM and launch training."""

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


def run_cmd(cmd, check=True):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0 and check:
        print(f"FAILED (code {res.returncode}):\n{res.stderr}\n{res.stdout}")
        sys.exit(1)
    return res


def setup_remote_directories():
    print(f"Ensuring remote directory structure on {INSTANCE_NAME}...")
    cmd = (
        f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="'
        f'mkdir -p {REMOTE_DIR}/data/processed/phase2_dataset '
        f'{REMOTE_DIR}/checkpoints/phase7_dampened_scaling_thermocline_25k '
        f'{REMOTE_DIR}/checkpoints/phase8_zone_adaptive_15k '
        f'{REMOTE_DIR}/src/training/config_registry '
        f'{REMOTE_DIR}/scripts/gcp '
        f'{REMOTE_DIR}/reports"'
    )
    run_cmd(cmd)


def package_and_upload_code():
    bundle_name = "phase8_code_bundle.tar.gz"
    bundle_path = REPO_ROOT / bundle_name

    code_paths = [
        "src",
        "scripts/gcp/train_phase8_15k.py",
        "scripts/gcp/evaluate_phase8_15k.py",
        "scripts/gcp/run_phase8_15k.sh",
        "src/training/config_registry/phase8_zone_adaptive_15k.yaml",
        "data/processed/anomaly_depth_scales_zone_adaptive_p8.json",
        "data/processed/channel_normalization_stats_ocean_only.json",
        "requirements.txt",
    ]

    print(f"Creating code bundle {bundle_name}...")
    with tarfile.open(bundle_path, "w:gz") as tar:
        for p in code_paths:
            full = REPO_ROOT / p
            if full.exists():
                print(f"  Adding {p}")
                tar.add(full, arcname=p)
            else:
                print(f"  Warning: {p} does not exist locally")

    print(f"Uploading {bundle_name} to VM...")
    scp_cmd = f"gcloud compute scp {bundle_path} {INSTANCE_NAME}:{bundle_name} --zone={ZONE}"
    run_cmd(scp_cmd)

    print("Extracting code bundle on VM...")
    ext_cmd = f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="tar -xzf ~/{bundle_name} -C {REMOTE_DIR}/ && rm ~/{bundle_name}"'
    run_cmd(ext_cmd)

    if bundle_path.exists():
        bundle_path.unlink()
    print("Code bundle deployed successfully.")


def upload_datasets_and_checkpoint():
    print("Checking remote data on VM...")
    check_cmd = f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="ls -l {REMOTE_DIR}/data/processed/phase2_dataset/oceanembed_training_inputs.zarr 2>/dev/null || echo MISSING"'
    res = run_cmd(check_cmd, check=False)

    if "MISSING" in res.stdout or res.returncode != 0:
        print("Data not found on VM. Packing and uploading datasets (~1.5 GB)...")
        data_bundle = REPO_ROOT / "phase8_data_bundle.tar"
        with tarfile.open(data_bundle, "w") as tar:
            data_dir = REPO_ROOT / "data/processed/phase2_dataset"
            print(f"  Archiving {data_dir}...")
            tar.add(data_dir, arcname="data/processed/phase2_dataset")

        print("Uploading data bundle to VM (this may take 2-4 minutes)...")
        scp_data = f"gcloud compute scp {data_bundle} {INSTANCE_NAME}:phase8_data_bundle.tar --zone={ZONE}"
        run_cmd(scp_data)

        print("Extracting datasets on VM...")
        ext_data = f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="tar -xf ~/phase8_data_bundle.tar -C {REMOTE_DIR}/ && rm ~/phase8_data_bundle.tar"'
        run_cmd(ext_data)

        if data_bundle.exists():
            data_bundle.unlink()
        print("Datasets deployed.")
    else:
        print("Datasets already present on VM.")

    # Upload Phase 7 checkpoint for warm start initialization
    p7_ckpt = REPO_ROOT / "checkpoints/phase7_dampened_scaling_thermocline_25k/last_checkpoint.pt"
    if p7_ckpt.exists():
        print(f"Uploading Phase 7 warm start checkpoint ({p7_ckpt.stat().st_size / 1024 / 1024:.1f} MB)...")
        scp_ckpt = f"gcloud compute scp {p7_ckpt} {INSTANCE_NAME}:{REMOTE_DIR}/checkpoints/phase7_dampened_scaling_thermocline_25k/last_checkpoint.pt --zone={ZONE}"
        run_cmd(scp_ckpt)
        print("Phase 7 checkpoint uploaded successfully.")


def install_remote_dependencies():
    print("Installing Python dependencies on VM...")
    pip_cmd = (
        f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="'
        f'source /opt/conda/bin/activate 2>/dev/null || true; '
        f'pip install -q zarr netCDF4 xarray pandas pyyaml scikit-learn torchvision matplotlib tqdm scipy"'
    )
    run_cmd(pip_cmd)
    print("Dependencies ready.")


def launch_training():
    print(f"\nLaunching Phase 8 training on {INSTANCE_NAME}...")
    launch_cmd = f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="bash {REMOTE_DIR}/scripts/gcp/run_phase8_15k.sh"'
    res = run_cmd(launch_cmd)
    print(res.stdout)

    time.sleep(4)

    print("\nChecking training startup log...")
    check_cmd = f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="head -n 45 {REMOTE_DIR}/train_phase8_15k.log || true"'
    res = run_cmd(check_cmd, check=False)
    print(res.stdout)


def main():
    print("=" * 80)
    print(f"SYNCHRONIZING AND LAUNCHING PHASE 8 ON {INSTANCE_NAME} ({ZONE})")
    print("=" * 80)
    setup_remote_directories()
    package_and_upload_code()
    upload_datasets_and_checkpoint()
    install_remote_dependencies()
    launch_training()
    print("=" * 80)
    print("PHASE 8 TRAINING SUCCESSFULLY LAUNCHED ON V100 GPU!")
    print("=" * 80)


if __name__ == "__main__":
    main()
