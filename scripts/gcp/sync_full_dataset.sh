#!/usr/bin/env bash
# ==============================================================================
# Sync Code and Preprocessed Datasets to GCP VM Persistent Disk
# ==============================================================================

set -euo pipefail

PROJECT_ID=${GCP_PROJECT_ID:-"ocean-embed-508404"}
ZONE=${GCP_ZONE:-"us-central1-a"}
INSTANCE_NAME=${GCP_INSTANCE_NAME:-"oceanembed-l4-training"}
REMOTE_DIR="/home/123ta/oceanembed"

echo "=== Packaging and Synchronizing to GCP VM ==="
BUNDLE_NAME="oceanembed_sync_bundle.tar.gz"

# 1. Package code, configs, scripts, tests, and processed datasets (excluding raw data)
tar --exclude="data/raw" \
    --exclude=".pytest_cache" \
    --exclude="__pycache__" \
    --exclude="*.tmp" \
    -czf "${BUNDLE_NAME}" \
    src scripts configs tests data/processed requirements.txt pytest.ini conftest.py README.md handoff.md

echo "Transferring bundle (${BUNDLE_NAME}) to VM..."
gcloud compute scp "${BUNDLE_NAME}" "${INSTANCE_NAME}:${REMOTE_DIR}/" --project="${PROJECT_ID}" --zone="${ZONE}"

echo "Unpacking on VM..."
gcloud compute ssh "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --command="cd ${REMOTE_DIR} && tar -xzf ${BUNDLE_NAME} && rm ${BUNDLE_NAME}"

# Clean local bundle
rm "${BUNDLE_NAME}"

echo "=== Sync Complete! Verified on remote disk. ==="
