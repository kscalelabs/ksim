FROM ghcr.io/nvidia/jax:upstream-t5x-2025-03-05 AS base

# Set up environment
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    JAX_PLATFORM_NAME=cuda

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    vim \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip 

# Safe computation stage
FROM base AS jax-base

# Install ksim
RUN git clone https://github.com/kscalelabs/ksim.git && \
    cd ksim && \
    pip install -e .

# Set up working directory
WORKDIR /ksim

# Default command
CMD ["bash"]
