#!/bin/bash

echo 'Building Dockerfile with image name d4rl_torel'
docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg GID=1234 \
    --build-arg REQS="$(cat requirements.txt | tr '\n' ' ')" \
    -t d4rl_torel \
    -f Dockerfile \
    .