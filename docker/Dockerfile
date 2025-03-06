FROM ghcr.io/nvidia/jax:upstream-t5x-2025-03-05 AS base

# Set up environment
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    JAX_PLATFORM_NAME=cuda

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    vim wget curl git ffmpeg

RUN apt install -y libgl1 libglx-mesa0 mesa-utils libglx0 libopengl0

# Upgrade pip
RUN pip install --upgrade pip 

# Safe computation stage
FROM base AS jax-base

# Headless mode
ENV MUJOCO_GL="egl"

# Install ksim
RUN git clone https://github.com/kscalelabs/ksim.git && \
    cd ksim && \
    pip install --upgrade -e .

# Set up working directory
WORKDIR /ksim

# Default command
CMD ["bash"]
