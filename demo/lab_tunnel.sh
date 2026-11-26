#!/usr/bin/env bash
# Connect the demo's live Qwen scoring to the lab GPU.
# Forwards this laptop's 127.0.0.1:11434 to the lab's isolated project Ollama (127.0.0.1:12466 on the lab),
# the same runner and model digest (qwen2.5:7b-instruct 845dbda0...) used for the cohort scores.
# Needs the `erisk` host in ~/.ssh/config and the campus network/VPN. Keep this terminal open; Ctrl+C to stop.
# If 11434 is busy (a local Ollama is running), stop that first.
set -e
HOST="${LAB_HOST:-erisk}"
REMOTE_PORT="${LAB_OLLAMA_PORT:-12466}"
echo "Tunnel 127.0.0.1:11434 -> $HOST:127.0.0.1:$REMOTE_PORT (Ctrl+C to stop)"
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:11434:127.0.0.1:"$REMOTE_PORT" "$HOST"
