#!/bin/bash
# Runtime provisioning script for vastai/comfy image.
# Installs ComfyUI-GGUF custom node + downloads Flux models from Hugging Face,
# then launches ComfyUI in a tmux session so SSH stays usable for debug.
#
# Designed to be idempotent: re-running on an already-provisioned instance is safe.
# Hosted at: https://raw.githubusercontent.com/castellidylan-tech/comfyui-flux-image/main/setup_comfy_runtime.sh

set -e
LOG=/var/log/comfy_setup.log
exec > >(tee -a $LOG) 2>&1

echo "=== [$(date)] setup_comfy_runtime.sh START ==="

# vastai/comfy image structure: ComfyUI lives under /opt/ComfyUI by default, with a
# Python env already set up. If they restructure, fall back to whatever exists.
COMFY_DIR=${COMFY_DIR:-/opt/ComfyUI}
if [ ! -d "$COMFY_DIR" ]; then
  for candidate in /workspace/ComfyUI /home/user/ComfyUI /root/ComfyUI; do
    if [ -d "$candidate" ]; then COMFY_DIR=$candidate; break; fi
  done
fi
echo "Using ComfyUI at: $COMFY_DIR"

# 1. ComfyUI-GGUF custom node (idempotent)
GGUF_DIR="$COMFY_DIR/custom_nodes/ComfyUI-GGUF"
if [ ! -d "$GGUF_DIR" ]; then
  echo "Installing ComfyUI-GGUF custom node..."
  git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git "$GGUF_DIR"
  pip install --break-system-packages gguf || pip install gguf
else
  echo "ComfyUI-GGUF already present"
fi

# 2. Download models in parallel with retry
MODEL_DIR="$COMFY_DIR/models"
mkdir -p "$MODEL_DIR/unet" "$MODEL_DIR/text_encoders" "$MODEL_DIR/vae"

dl_with_retry() {
  local url=$1 out=$2 attempt
  if [ -f "$out" ] && [ -s "$out" ]; then echo "skip (exists): $out"; return 0; fi
  for attempt in 1 2 3; do
    echo "DL [$attempt/3]: $out"
    if curl -L --fail --retry 2 --retry-delay 3 -o "$out.tmp" "$url"; then
      mv "$out.tmp" "$out"
      echo "OK: $out ($(du -h "$out" | cut -f1))"
      return 0
    fi
    rm -f "$out.tmp"
    sleep 5
  done
  echo "FAIL after 3 attempts: $out"
  return 1
}

dl_with_retry "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_S.gguf?download=true" \
  "$MODEL_DIR/unet/flux1-dev-Q4_K_S.gguf" &

dl_with_retry "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors?download=true" \
  "$MODEL_DIR/text_encoders/t5xxl_fp8_e4m3fn.safetensors" &

dl_with_retry "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true" \
  "$MODEL_DIR/text_encoders/clip_l.safetensors" &

dl_with_retry "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.safetensors?download=true" \
  "$MODEL_DIR/vae/flux_ae.safetensors" &

wait
echo "All model downloads finished"

# 3. Launch ComfyUI in tmux so SSH stays available
if ! command -v tmux >/dev/null; then
  apt-get update && apt-get install -y tmux
fi

# Kill any existing comfy session, then start fresh
tmux kill-session -t comfy 2>/dev/null || true
tmux new-session -d -s comfy "cd $COMFY_DIR && python3 main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch 2>&1 | tee /var/log/comfy.log"
echo "ComfyUI launched in tmux 'comfy' (attach: tmux attach -t comfy)"

echo "=== [$(date)] setup_comfy_runtime.sh DONE ==="
