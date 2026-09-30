#!/usr/bin/env bash
set -e
cd /home/123ta/oceanembed

# Kill any existing session with this name
tmux kill-session -t rectified_25k 2>/dev/null || true

# Create detached session
tmux new-session -d -s rectified_25k

# Send execution command into tmux session
tmux send-keys -t rectified_25k "cd /home/123ta/oceanembed && python3 scripts/gcp/train_rectified_25k.py 2>&1 | tee train_rectified_25k.log" C-m

echo "Session rectified_25k started successfully:"
tmux ls
