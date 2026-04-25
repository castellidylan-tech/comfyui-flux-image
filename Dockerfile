# Custom ComfyUI + Flux image for Salad cloud sprite generation pipeline.
#
# Strategy: bake everything into the image so cold start = just pull image + start ComfyUI.
# No runtime model downloads, no volume mounts (Salad doesn't have persistent volumes).
#
# Build: docker build -t <USER>/comfyui-flux:latest .
# Push:  docker push <USER>/comfyui-flux:latest
# Use:   set CONTAINER_IMAGE in salad_pipeline.py to "<USER>/comfyui-flux:latest"

FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Pip + torch (matching CUDA 13.x — required by current ComfyUI which imports torchaudio with cu130 deps)
RUN pip install --upgrade pip --break-system-packages && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

# ComfyUI
WORKDIR /opt
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /opt/ComfyUI
RUN pip install -r requirements.txt --break-system-packages

# Custom node: GGUF support (for Flux Q4 quantized models)
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git \
    && pip install --upgrade gguf --break-system-packages

# Models (baked into image so cold start has zero downloads)
WORKDIR /opt/ComfyUI/models

# Flux dev Q4_K_S GGUF (~6.6 GB) — main diffusion model
RUN mkdir -p unet && cd unet && \
    curl -L -o flux1-dev-Q4_K_S.gguf \
    "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_S.gguf?download=true"

# Text encoders
RUN mkdir -p text_encoders && cd text_encoders && \
    curl -L -o t5xxl_fp8_e4m3fn.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors?download=true" && \
    curl -L -o clip_l.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true"

# VAE (mirror that doesn't require HF auth)
RUN mkdir -p vae && cd vae && \
    curl -L -o flux_ae.safetensors \
    "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.safetensors?download=true"

# Expose ComfyUI HTTP API port
EXPOSE 8188

# Start ComfyUI listening on all interfaces (so Salad's load balancer can reach it)
WORKDIR /opt/ComfyUI
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--disable-auto-launch"]
