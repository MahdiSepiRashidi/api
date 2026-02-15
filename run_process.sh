#!/bin/bash

# Install uv via pip
pip install uv

# Update package list and install tmux
apt-get update
apt-get install -y tmux


# Run the process using tmux in a detached session
tmux new-session -d -s my_process 'uv run main.py'
# keep container alive
tail -f /dev/null