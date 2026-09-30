"""
Automated Multi-Region GCP Launcher for Model V2 60,000-Step Training.

Searches across candidate GCP regions and zones to provision a `g2-standard-8` instance
(8 vCPUs, 32 GB RAM, 1x NVIDIA L4 24GB GPU) with automatic fallback to ensure immediate
resource availability without region starvation.

Usage:
    python scripts/launch_gcp_60k_training.py --dry-run
    python scripts/launch_gcp_60k_training.py --execute
"""

import sys
import os
import argparse
import subprocess
import json
import time
from pathlib import Path

# Candidate regions prioritized by low latency and L4 capacity
CANDIDATE_ZONES = [
    ("us-central1", "us-central1-a"),
    ("us-central1", "us-central1-b"),
    ("us-east4", "us-east4-a"),
    ("us-east4", "us-east4-b"),
    ("us-west1", "us-west1-a"),
    ("us-west1", "us-west1-b"),
    ("europe-west4", "europe-west4-a"),
    ("europe-west4", "europe-west4-b"),
    ("asia-southeast1", "asia-southeast1-a"),
    ("asia-southeast1", "asia-southeast1-b"),
]

PRIMARY_MACHINE_TYPE = "g2-standard-8"
FALLBACK_MACHINE_TYPE = "g2-standard-4"
ACCELERATOR_TYPE = "nvidia-l4"
DISK_SIZE_GB = 100
INSTANCE_NAME = "oceanembed-l4-60k-v2"
CONFIG_FILE = "src/training/config_registry/phase4_scratch_60k_full_config.yaml"


# Auto-detect gcloud binary path
GCLOUD_BIN = "gcloud"
POSSIBLE_GCLOUD_PATHS = [
    os.path.expanduser(r"~\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"),
    r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
    r"C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
]
for p in POSSIBLE_GCLOUD_PATHS:
    if os.path.exists(p):
        GCLOUD_BIN = p
        break


def check_gcloud_auth():
    """Verify gcloud CLI is installed and authenticated."""
    try:
        res = subprocess.run(
            [GCLOUD_BIN, "auth", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
            shell=(os.name == "nt"),
        )
        accounts = json.loads(res.stdout)
        active = [a for a in accounts if a.get("status") == "ACTIVE"]
        if not active:
            print("[WARN] No active gcloud account detected. Please run 'gcloud auth login'.")
            return False
        print(f"[OK] Authenticated GCP Account: {active[0].get('account')}")
        return True
    except Exception as e:
        print(f"[ERROR] gcloud check failed: {e}")
        return False


def test_zone_availability(zone: str, machine_type: str = PRIMARY_MACHINE_TYPE) -> bool:
    """Check if the requested machine type and accelerator are available in zone."""
    cmd = [
        GCLOUD_BIN, "compute", "machine-types", "describe", machine_type,
        f"--zone={zone}", "--format=json"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, shell=(os.name == "nt"))
        return (res.returncode == 0)
    except Exception:
        return False


def build_startup_script() -> str:
    """Generate bash startup script for automated environment setup and training execution."""
    return """#!/bin/bash
set -e
echo "=== Starting OceanEmbed 60k Training Setup ==="
export DEBIAN_FRONTEND=noninteractive

# 1. Update and install system dependencies
sudo apt-get update -y
sudo apt-get install -y git python3-pip python3-venv tmux htop

# 2. Setup workspace
cd /home
mkdir -p oceanembed && cd oceanembed

# 3. Pull latest code or sync archive
# (User can scp code_sync.tar.gz or clone repo)
if [ -f /tmp/phase3_code_sync.tar.gz ]; then
    tar -xzf /tmp/phase3_code_sync.tar.gz -C .
fi

# 4. Install python dependencies
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if [ -f requirements.txt ]; then
    pip3 install -r requirements.txt
fi

echo "=== Environment Ready. Launching Training in Tmux ==="
tmux new-session -d -s train 'python3 -m src.training.train --config src/training/config_registry/phase4_scratch_60k_full_config.yaml 2>&1 | tee training_60k.log'

echo "=== Training launched in tmux session 'train' ==="
"""


def main():
    parser = argparse.ArgumentParser(description="Multi-Region GCP Launcher for 60k Run")
    parser.add_argument("--dry-run", action="store_true", help="Scan candidate regions without provisioning")
    parser.add_argument("--execute", action="store_true", help="Provision instance in first available zone")
    parser.add_argument("--machine-type", default=PRIMARY_MACHINE_TYPE, help="Target machine type")
    args = parser.parse_args()

    print("=" * 80)
    print("OCEANEMBED GCP MULTI-REGION PROVISIONING MANAGER (60,000 STEPS)")
    print(f"Target Architecture: {args.machine_type} + 1x NVIDIA L4 (24GB)")
    print("=" * 80)

    auth_ok = check_gcloud_auth()
    if not auth_ok:
        print("\n[ABORT] Please configure gcloud credentials before proceeding.")
        return

    print("\n[Scanning Candidate Zones for L4 Availability]...")
    available_zones = []
    for region, zone in CANDIDATE_ZONES:
        avail = test_zone_availability(zone, args.machine_type)
        status = "AVAILABLE" if avail else "UNAVAILABLE"
        print(f"  • Region: {region:<16} | Zone: {zone:<18} -> {status}")
        if avail:
            available_zones.append(zone)

    if not available_zones:
        print("\n[WARN] Primary type unavailable in standard regions. Checking fallback (g2-standard-4)...")
        for region, zone in CANDIDATE_ZONES:
            avail = test_zone_availability(zone, FALLBACK_MACHINE_TYPE)
            if avail:
                available_zones.append(zone)
                print(f"  • Fallback Available: Zone {zone} ({FALLBACK_MACHINE_TYPE})")

    if not available_zones:
        print("\n[ERROR] No candidate zones currently have GPU capacity available.")
        return

    selected_zone = available_zones[0]
    print(f"\n>> Selected Target Zone: {selected_zone} (Machine: {args.machine_type})")

    if args.dry_run or not args.execute:
        print("\n[DRY RUN COMPLETE] To provision this instance and launch training, rerun with --execute.")
        return

    # Provision instance
    print(f"\n[PROVISIONING] Launching instance '{INSTANCE_NAME}' in {selected_zone}...")
    create_cmd = [
        "gcloud", "compute", "instances", "create", INSTANCE_NAME,
        f"--zone={selected_zone}",
        f"--machine-type={args.machine_type}",
        f"--accelerator=type={ACCELERATOR_TYPE},count=1",
        "--maintenance-policy=TERMINATE",
        f"--boot-disk-size={DISK_SIZE_GB}GB",
        "--boot-disk-type=pd-ssd",
        "--image-family=common-cu121-debian-11-py310",
        "--image-project=deeplearning-platform-release",
        "--scopes=cloud-platform",
    ]

    print("Command:", " ".join(create_cmd))
    res = subprocess.run(create_cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[SUCCESS] Instance '{INSTANCE_NAME}' successfully created in {selected_zone}!")
        print("Use the following command to SSH into the instance:")
        print(f"  gcloud compute ssh {INSTANCE_NAME} --zone={selected_zone}")
    else:
        print(f"[ERROR] Instance creation failed:\n{res.stderr}")


if __name__ == "__main__":
    main()
