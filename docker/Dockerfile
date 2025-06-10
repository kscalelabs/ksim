FROM ghcr.io/nvidia/jax:jax

RUN apt-get update && \
    apt-get install -y \
        libgl1 \
        libopengl0 \
        libglib2.0-0t64 \
        libx11-6 \
        libxext6 \
        libsm6 \
        libxrender1 \
        libxi6 \
        libxrandr2 \
        libxcursor1 \
        libxinerama1 \
        libxxf86vm1 \
        libxfixes3 \
        libglfw3 \
        && rm -rf /var/lib/apt/lists/*

# use local version
# WORKDIR /ksim
# COPY ../ /ksim
# RUN pip install -e .

# use pre-built lib from pip repo
RUN pip install ksim
