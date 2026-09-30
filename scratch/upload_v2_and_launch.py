import os
import sys
import subprocess
import tarfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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


def main():
    # 1. Clean remote phase2_dataset
    print("Cleaning remote dataset directory...")
    run_cmd(f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="rm -rf {REMOTE_DIR}/data/processed/phase2_dataset && mkdir -p {REMOTE_DIR}/data/processed/phase2_dataset"')

    # 2. Package phase2_dataset_v2
    tar_path = REPO_ROOT / "phase8_v2_data.tar"
    print(f"Creating archive from phase2_dataset_v2 ({tar_path})...")
    with tarfile.open(tar_path, "w") as tar:
        v2_dir = REPO_ROOT / "data/processed/phase2_dataset_v2"
        for item in v2_dir.iterdir():
            print(f"  Adding {item.name}")
            tar.add(item, arcname=item.name)

    print(f"Archive created ({tar_path.stat().st_size / 1024 / 1024:.1f} MB). Uploading to VM...")
    run_cmd(f"gcloud compute scp {tar_path} {INSTANCE_NAME}:phase8_v2_data.tar --zone={ZONE}")

    print("Extracting archive into remote phase2_dataset...")
    run_cmd(f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="tar -xf ~/phase8_v2_data.tar -C {REMOTE_DIR}/data/processed/phase2_dataset/ && rm ~/phase8_v2_data.tar"')

    if tar_path.exists():
        tar_path.unlink()

    # 3. Upload code bundle
    print("Updating code bundle...")
    code_tar = REPO_ROOT / "code.tar.gz"
    with tarfile.open(code_tar, "w:gz") as tar:
        for p in [
            "src",
            "scripts/gcp/train_phase8_15k.py",
            "scripts/gcp/evaluate_phase8_15k.py",
            "scripts/gcp/run_phase8_15k.sh",
            "src/training/config_registry/phase8_zone_adaptive_15k.yaml",
            "data/processed/anomaly_depth_scales_zone_adaptive_p8.json",
            "data/processed/channel_normalization_stats_ocean_only.json",
        ]:
            tar.add(REPO_ROOT / p, arcname=p)
    run_cmd(f"gcloud compute scp {code_tar} {INSTANCE_NAME}:code.tar.gz --zone={ZONE}")
    run_cmd(f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="tar -xzf ~/code.tar.gz -C {REMOTE_DIR}/ && rm ~/code.tar.gz"')
    if code_tar.exists():
        code_tar.unlink()

    # 4. Launch training in tmux
    print("\nLaunching Phase 8 training on VM...")
    res = run_cmd(f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="bash {REMOTE_DIR}/scripts/gcp/run_phase8_15k.sh"')
    print(res.stdout)

    time.sleep(5)

    # 5. Check startup log
    print("\nChecking training startup log:")
    check_res = run_cmd(f'gcloud compute ssh {INSTANCE_NAME} --zone={ZONE} --command="head -n 45 {REMOTE_DIR}/train_phase8_15k.log || true"', check=False)
    print(check_res.stdout)


if __name__ == "__main__":
    main()
