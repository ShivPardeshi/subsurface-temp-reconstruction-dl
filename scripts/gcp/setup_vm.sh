#!/usr/bin/env bash
# ==============================================================================
# OceanEmbed (PS26066) — GCP VM Provisioning Script (Track B)
# Target: 1x NVIDIA L4 GPU (24GB VRAM) for Phase 4 Training
# ==============================================================================

set -euo pipefail

# Configuration variables (modify according to your GCP project/region)
PROJECT_ID=${GCP_PROJECT_ID:-"ocean-embed-508404"}
ZONE=${GCP_ZONE:-"us-central1-a"}  # Flagship US datacenter with high L4 availability
INSTANCE_NAME=${GCP_INSTANCE_NAME:-"oceanembed-l4-training"}
MACHINE_TYPE=${GCP_MACHINE_TYPE:-"g2-standard-4"}  # 4 vCPUs, 16GB RAM, 1x NVIDIA L4
DISK_SIZE="200GB"
DISK_TYPE="pd-ssd"

echo "=== OceanEmbed GCP L4 Provisioning ==="
echo "Project:       ${PROJECT_ID}"
echo "Zone:          ${ZONE}"
echo "Instance Name: ${INSTANCE_NAME}"
echo "Machine Type:  ${MACHINE_TYPE}"
echo "Disk Size:     ${DISK_SIZE} (${DISK_TYPE})"
echo "======================================"

# Step 1: Set GCP project
gcloud config set project "${PROJECT_ID}"

# Step 2: Create VM with Deep Learning PyTorch/CUDA pre-installed image
gcloud compute instances create "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --maintenance-policy="TERMINATE" \
    --boot-disk-size="${DISK_SIZE}" \
    --boot-disk-type="${DISK_TYPE}" \
    --boot-disk-auto-delete \
    --image-family="pytorch-2-9-cu129-ubuntu-2204-nvidia-580" \
    --image-project="deeplearning-platform-release" \
    --metadata="install-nvidia-driver=True" \
    --scopes="https://www.googleapis.com/auth/cloud-platform"

echo "=== VM Provisioned Successfully! ==="
echo "Next steps:"
echo "1. Connect via SSH:  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
echo "2. Inside VM, clone repo & install requirements:"
echo "   git clone <REPO_URL> oceanembed && cd oceanembed && pip install -r requirements.txt"
echo "3. Run environment verification:"
echo "   python scripts/gcp/verify_gcp_env.py"
echo "4. STOP THE VM immediately after verification to prevent billing:"
echo "   gcloud compute instances stop ${INSTANCE_NAME} --zone=${ZONE}"
