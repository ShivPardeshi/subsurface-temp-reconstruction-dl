#!/usr/bin/env bash
# ==============================================================================
# Resume Stopped GCP VM for Phase 4 Training
# ==============================================================================

set -euo pipefail

PROJECT_ID=${GCP_PROJECT_ID:-"ocean-embed-508404"}
ZONE=${GCP_ZONE:-"us-central1-a"}
INSTANCE_NAME=${GCP_INSTANCE_NAME:-"oceanembed-l4-training"}

echo "=== Resuming OceanEmbed L4 VM ==="
echo "Project:  ${PROJECT_ID}"
echo "Zone:     ${ZONE}"
echo "Instance: ${INSTANCE_NAME}"

# 1. Start VM
gcloud compute instances start "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}"

# 2. Wait for SSH availability & verify GPU
echo "Checking GPU visibility on resumed VM..."
gcloud compute ssh "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --command="nvidia-smi"

echo "=== VM Resumed and Ready for Training! ==="
