#!/bin/bash

echo 'Building Dockerfile with image name sorel_torel'
docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg GID=1234 \
    --build-arg REQS="$(cat requirements.txt | tr '\n' ' ')" \
    -t sorel_torel \
    -f Dockerfile \
    .