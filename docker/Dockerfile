# Use an official Python base image with CUDA support
# FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and Python 3.9
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    git wget sudo htop curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
libgl1-mesa-dev \
libgl1-mesa-glx \
libglew-dev \
libosmesa6-dev \
software-properties-common \
net-tools \
vim \
virtualenv \
xpra \
xserver-xorg-dev \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

RUN pip install wandb
RUN pip install "cython<3"
# Copy requirements.txt to the container
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY mujoco-py/vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

COPY mujoco-py/vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY mujoco-py /mujoco-py
WORKDIR /mujoco-py
RUN python setup.py install

# Set working directory
WORKDIR /workspace