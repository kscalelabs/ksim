#!/bin/bash

IMAGE_NAME="ksim"

docker build -t $IMAGE_NAME .
docker run --gpus all \
    --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm --network=host \
    -it $IMAGE_NAME
