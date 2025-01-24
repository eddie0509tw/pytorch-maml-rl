# Use an official PyTorch base image
ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDNN="8"
# ARG UBUNTU="22.04"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-runtime-ubuntu${UBUNTU}
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 \
    zlib1g-dev libxslt1-dev libssl-dev libffi-dev libxml2-dev \
    libgl1-mesa-dev libglew-dev libosmesa6-dev \
    software-properties-common net-tools vim virtualenv xpra unzip \
    xserver-xorg-dev patchelf wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.8 and set up a virtual environment
RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3.8-venv python3-pip \
    && python3.8 -m venv /venv \
    && /venv/bin/pip install --upgrade pip

# Ensure the virtual environment is used by default
ENV PATH="/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN /venv/bin/pip install --upgrade pip==22.0.3
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt
RUN /venv/bin/pip install opencv-python==3.4.17.61 
RUN /venv/bin/pip install gym==0.19.0
RUN /venv/bin/pip install wandb

# RUN mkdir -p /root/.mujoco \
#     && wget https://www.roboti.us/download/mujoco200_linux.zip \
#     && unzip mujoco200_linux.zip -d /root/.mujoco/ \
#     && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
#     && wget https://www.roboti.us/file/mjkey.txt \
#     && mv mjkey.txt /root/.mujoco/mjkey.txt \
#     && rm mujoco200_linux.zip

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/nvidia"
ENV PATH="${LD_LIBRARY_PATH}:${PATH}"
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libGLEW.so"

COPY mujoco-py/vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

COPY mujoco-py/vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY mujoco-py /mujoco-py
WORKDIR /mujoco-py
RUN python setup.py install

# Set working directory
WORKDIR /workspace
