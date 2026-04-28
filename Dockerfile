# SDXL stack qualité max pour Vast 3090 24GB.
# Pivoted from Flux Q4_K_S (which was a leftover from 4060 Ti 8GB constraint).
# On a 24GB VRAM GPU there's no reason to quantize — full FP16 SDXL is the way.
#
# Stack :
#   SDXL Base 1.0 FP16  + SDXL Refiner FP16 (qualité top-tier)
#   ControlNet Union SDXL Pro Max (Xinsir, all-in-one : depth/canny/openpose/etc.)
#   IP-Adapter Plus SDXL (H94, mature depuis 2 ans, marche out-of-the-box)
#   PixelArtXL LoRA (au cas où — workflow par défaut ne l'utilise pas)
#   ComfyUI-IPAdapter-Plus (custom node de cubiq, mature et maintenu)
#
# Image size estimée : ~24 GB
# Pull time on Vast 800 Mb/s : ~4 min (vs 5-10 min pour Flux 28 GB)
#
# Build: docker build -t daigami/comfyui-sdxl:latest .
# Push:  docker push daigami/comfyui-sdxl:latest

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

# Torch CUDA 13.x
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

# ComfyUI
WORKDIR /opt
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /opt/ComfyUI
RUN pip install -r requirements.txt --break-system-packages

# Custom node: IP-Adapter Plus (cubiq) — mature SDXL IP-Adapter implementation
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

# Models — bake everything for self-contained image
WORKDIR /opt/ComfyUI/models

# SDXL Base 1.0 FP16 (~6.5 GB) — main checkpoint, top-tier quality
RUN mkdir -p checkpoints && cd checkpoints && \
    curl -L -o sd_xl_base_1.0.safetensors \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true"

# SDXL Refiner 1.0 FP16 (~6.5 GB) — pour upscale qualité finale (optional in workflow but baked in)
RUN cd checkpoints && \
    curl -L -o sd_xl_refiner_1.0.safetensors \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors?download=true"

# ControlNet Union ProMax SDXL (Xinsir, ~2.5 GB) — handles depth/canny/openpose/scribble/normal/segmentation in one model
RUN mkdir -p controlnet && cd controlnet && \
    curl -L -o controlnet_union_sdxl_promax.safetensors \
    "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors?download=true"

# IP-Adapter Plus SDXL (H94, ~700 MB) — the Plus version transfers identity better than the basic
RUN mkdir -p ipadapter && cd ipadapter && \
    curl -L -o ip-adapter-plus_sdxl_vit-h.safetensors \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors?download=true"

# CLIP-H/14 vision encoder (~2.5 GB) — required by ip-adapter-plus_*_vit-h
RUN mkdir -p clip_vision && cd clip_vision && \
    curl -L -o CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true"

# PixelArtXL LoRA (Nerijs, ~150 MB) — optional, baked at low cost
RUN mkdir -p loras && cd loras && \
    curl -L -o pixel-art-xl.safetensors \
    "https://huggingface.co/nerijs/pixel-art-xl/resolve/main/pixel-art-xl.safetensors?download=true"

# Expose ComfyUI HTTP API port
EXPOSE 8188

# Start ComfyUI listening on all interfaces
WORKDIR /opt/ComfyUI
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--disable-auto-launch"]
