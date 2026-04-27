# Custom ComfyUI + Flux + IP-Adapter image for cloud GPU sprite generation pipeline.
#
# Strategy: bake EVERYTHING into the image so cold start = pull + start ComfyUI.
# No runtime model downloads, no onstart scripts (they were 400-error-prone on Vast).
#
# Total image size: ~28 GB (was 13 GB — added ControlNet 6.6GB + IP-Adapter 5.3GB + SigLIP 3.5GB + LoRA 0.3GB)
# Pull time on Vast 3.4 Gb/s: ~2 min. On 800 Mb/s: ~5 min.
#
# Build: docker build -t daigami/comfyui-flux:latest .
# Push:  docker push daigami/comfyui-flux:latest
# Use:   CONTAINER_IMAGE in vastai_pipeline.py is "daigami/comfyui-flux:latest"

FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git curl ca-certificates libgl1 libglib2.0-0 tmux \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Torch matching CUDA 13.x (required by current ComfyUI which imports torchaudio with cu130 deps).
# --break-system-packages because Ubuntu 24.04 has PEP 668 protection.
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

# ComfyUI
WORKDIR /opt
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /opt/ComfyUI
RUN pip install -r requirements.txt --break-system-packages

# Custom node: GGUF support (for Flux Q4 quantized models)
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git \
    && pip install --upgrade gguf --break-system-packages

# Custom node: IP-Adapter for Flux (Shakker-Labs) + apply community patches PR #108
# Maintainer unresponsive since 2025-06. PR #108 by redaah fixes ComfyUI v0.14+ compat.
COPY apply_ipadapter_patches.py /opt/apply_ipadapter_patches.py
RUN git clone --depth 1 https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux.git \
    && pip install einops==0.8.0 "transformers>=4.37.2" diffusers "sentencepiece>=0.2.0" "protobuf>=4.25.5" --break-system-packages \
    && python3 /opt/apply_ipadapter_patches.py /opt/ComfyUI/custom_nodes/ComfyUI-IPAdapter-Flux

# Models (baked into image so cold start has zero downloads)
WORKDIR /opt/ComfyUI/models

# Flux dev Q4_K_S GGUF (~6.6 GB) — main diffusion model
RUN mkdir -p unet && cd unet && \
    curl -L -o flux1-dev-Q4_K_S.gguf \
    "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_S.gguf?download=true"

# Text encoders (T5xxl fp8 + CLIP-L)
RUN mkdir -p text_encoders && cd text_encoders && \
    curl -L -o t5xxl_fp8_e4m3fn.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors?download=true" && \
    curl -L -o clip_l.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true"

# VAE
RUN mkdir -p vae && cd vae && \
    curl -L -o flux_ae.safetensors \
    "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.safetensors?download=true"

# LoRA umempart Modern Pixel Art (~344 MB) — UmeAiRT, MIT license
RUN mkdir -p loras && cd loras && \
    curl -L -o ume_modern_pixelart.safetensors \
    "https://huggingface.co/UmeAiRT/FLUX.1-dev-LoRA-Modern_Pixel_art/resolve/main/ume_modern_pixelart.safetensors?download=true"

# ControlNet Flux Union Pro (~6.6 GB) — Shakker-Labs (handles depth/canny/openpose)
RUN mkdir -p controlnet && cd controlnet && \
    curl -L -o flux_union_pro.safetensors \
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors?download=true"

# IP-Adapter Flux (~5.3 GB) — InstantX, flux-1-dev-non-commercial-license
RUN mkdir -p ipadapter-flux && cd ipadapter-flux && \
    curl -L -o ip-adapter.bin \
    "https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/resolve/main/ip-adapter.bin?download=true"

# SigLIP vision encoder for IP-Adapter (~3.5 GB safetensors + small config)
RUN mkdir -p clip_vision/google--siglip-so400m-patch14-384 \
    && cd clip_vision/google--siglip-so400m-patch14-384 \
    && for f in model.safetensors config.json preprocessor_config.json tokenizer.json tokenizer_config.json special_tokens_map.json spiece.model; do \
        curl -L -o "$f" "https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/$f?download=true"; \
    done

# Expose ComfyUI HTTP API port
EXPOSE 8188

# Start ComfyUI listening on all interfaces
WORKDIR /opt/ComfyUI
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--disable-auto-launch"]
