#!/bin/sh
# Private runtime bundle. Shared Ollama files/services are never modified.
set -eu
umask 077
task_root="$HOME/erisk_llm_signals"
mkdir -p "$task_root/runtime/bin" "$task_root/runtime/lib/ollama" "$task_root/private/slot_cache"
cp "$HOME/.local/bin/ollama" "$task_root/runtime/bin/ollama"
chmod 700 "$task_root/runtime/bin/ollama"
for item in "$HOME"/.local/lib/ollama/*; do
    name=$(basename "$item")
    if [ "$name" != llama-server ] && [ ! -e "$task_root/runtime/lib/ollama/$name" ]; then
        ln -s "$item" "$task_root/runtime/lib/ollama/$name"
    fi
done
cat > "$task_root/runtime/lib/ollama/llama-server" <<'WRAPPER'
#!/bin/sh
exec "$HOME/.local/lib/ollama/llama-server" "$@" --slot-save-path "$HOME/erisk_llm_signals/private/slot_cache"
WRAPPER
chmod 700 "$task_root/runtime/lib/ollama/llama-server"
export OLLAMA_HOST=127.0.0.1:12466 OLLAMA_NO_CLOUD=1
export OLLAMA_DEBUG=0 OLLAMA_DEBUG_LOG_REQUESTS=0 OLLAMA_FLASH_ATTENTION=0
export OLLAMA_NUM_PARALLEL=1 LLAMA_ARG_CACHE_RAM=0
exec "$task_root/runtime/bin/ollama" serve
