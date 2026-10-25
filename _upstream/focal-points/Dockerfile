# GPU base (CUDA 12.1 runtime, Ubuntu 22.04)
#
# Build from the REPO ROOT (so the shared poseidon_compute package in ../util is in context):
#     docker build -f focal-points/Dockerfile -t focal-points .
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TRANSFORMERS_NO_TORCHVISION=1 \
    TORCH_COMPILE_DISABLE=1 \
    HF_HOME=/hf \
    TORCH_HOME=/cache \
    TRITON_CACHE_DIR=/cache \
    TORCHINDUCTOR_CACHE_DIR=/cache \
    XDG_CACHE_HOME=/cache \
    MPLCONFIGDIR=/cache

# System deps + Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates git build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Virtualenv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m pip install --upgrade pip setuptools wheel

# Workdirs
RUN mkdir -p /workspace /hf /cache /logs /results /plots && chmod -R 777 /workspace /cache /logs /results /plots /hf
WORKDIR /workspace

# Install Python deps
COPY focal-points/requirements.txt /workspace/requirements.txt

# PyTorch CUDA 12.1 wheels (official index)
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch

# Shared compute/inference layer (poseidon_compute) — installed before project deps so imports resolve.
COPY util /opt/poseidon-util
RUN python -m pip install --no-cache-dir -e /opt/poseidon-util

# Project dependencies
RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt

# Copy project files
COPY focal-points /workspace

# Set HOME
ENV HOME=/workspace

# Entrypoint 
ENTRYPOINT ["/bin/bash", "/workspace/experiments.sh"]
CMD ["-m", "meta-llama/Llama-3.3-70B-Instruct", "-n", "30", "-q", "8bit", "-p", "true"]