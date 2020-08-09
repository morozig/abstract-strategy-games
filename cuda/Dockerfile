FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update && \
    apt-get install -y curl && \
    cd ~ && \
    curl -sL https://deb.nodesource.com/setup_14.x -o nodesource_setup.sh \
    /bin/sh nodesource_setup.sh &&
    apt-get install nodejs