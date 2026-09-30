#!/bin/bash
set -e
cd /home/123ta/oceanembed
source /home/123ta/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc 2>/dev/null || true
conda activate oceanembed 2>/dev/null || true
tmux kill-session -t phase7_25k 2>/dev/null || true
tmux new-session -d -s phase7_25k "python3 scripts/gcp/train_phase7_25k.py 2>&1 | tee train_phase7_25k.log"
echo "Phase 7 25k training session launched in tmux session 'phase7_25k'."
