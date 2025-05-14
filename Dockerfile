# 베이스: CUDA 12.1 + Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 기본 설정
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Python, 기본 툴 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libosmesa6-dev \
    libglfw3 \
    libglew-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglfw3-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxxf86vm-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    && apt-get clean

# pip 최신화
RUN python3.10 -m pip install --upgrade pip

# 필요한 Python 패키지 설치
RUN python3.10 -m pip install \
    numpy==1.26.4 \
    scipy==1.12.0 \
    pandas>=1.2 \
    jupyterlab \
    scikit-learn>=0.22 \
    opencv-python>=4.2 \
    pyyaml>=5.1 \
    yacs>=0.1.6 \
    einops>=0.3 \
    tensorboard \
    psutil \
    tqdm \
    matplotlib \
    simplejson \
    fvcore \
    av \
    gym==0.20.0 \
    flax==0.7.5 \
    imageio \
    tfp-nightly==0.20.0.dev20230524 \
    wandb \
    ml_collections \
    mujoco-py==2.1.2.14 \
    mujoco==2.3.7 \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    "jax[cuda12_pip]==0.4.19"

# dm_control은 따로 설치
RUN git clone https://github.com/deepmind/dm_control.git && \
    cd dm_control && python3.10 -m pip install .

# brax 설치
RUN git clone https://github.com/google/brax.git && \
    cd brax && python3.10 -m pip install -e .

# 기본 명령어
CMD ["/bin/bash"]
