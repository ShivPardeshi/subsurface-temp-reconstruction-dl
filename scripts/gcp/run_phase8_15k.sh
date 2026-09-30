#!/usr/bin/env bash
# ==============================================================================
# Phase 8: Zone-Adaptive Depth Scaling 15k Fine-Tuning Execution Script
# Target Hardware: NVIDIA Tesla V100 GPU (16GB VRAM)
# ==============================================================================

set -euo pipefail
cd /home/123ta/oceanembed

echo "=== LAUNCHING PHASE 8 ZONE-ADAPTIVE 15k FINE-TUNING RUN IN TMUX ==="
tmux kill-session -t phase8_train 2>/dev/null || true

tmux new-session -d -s phase8_train "
    cd /home/123ta/oceanembed &&
    export PYTHONPATH=. &&
    python3 -u scripts/gcp/train_phase8_15k.py 2>&1 | tee /home/123ta/oceanembed/train_phase8_15k.log
"

echo "Phase 8 training session launched in tmux session 'phase8_train'."
echo "Log file: /home/123ta/oceanembed/train_phase8_15k.log"
