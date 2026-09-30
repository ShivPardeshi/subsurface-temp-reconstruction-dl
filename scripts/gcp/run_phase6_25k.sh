#!/usr/bin/env bash
set -e
cd /home/123ta/oceanembed

# Kill any existing session with this name
tmux kill-session -t phase6_25k 2>/dev/null || true

# Create detached session
tmux new-session -d -s phase6_25k

# Send execution command into tmux session
tmux send-keys -t phase6_25k "cd /home/123ta/oceanembed && python3 scripts/gcp/train_phase6_25k.py 2>&1 | tee train_phase6_25k.log" C-m

echo "Session phase6_25k started successfully:"
tmux ls
