FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip install --upgrade pip

WORKDIR /app

# Install Python deps before copying source so this layer is cached on code changes
COPY requirements.txt .
RUN pip install --no-cache-dir runpod requests \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace VAE so cold starts don't need network for it
RUN python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"

COPY . .

# If checkpoints live on a RunPod Network Volume they will be mounted at
# /runpod-volume/checkpoints — override the defaults via env vars:
#   CKPT_PATH=/runpod-volume/checkpoints/latentsync_unet.pt
#   UNET_CONFIG_PATH=/runpod-volume/configs/unet/stage2.yaml
#
# Otherwise bake them in by placing them under checkpoints/ before building.

CMD ["python", "handler.py"]
