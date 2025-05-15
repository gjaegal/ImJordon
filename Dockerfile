# NVIDIA CUDA 기반 이미지 사용 (PyTorch와 JAX를 위한 CUDA 12 지원)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/conda/bin:${PATH}"

# 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglew-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    wget \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libavutil-dev \
    libavdevice-dev \
    libavfilter-dev \
    libswresample-dev \
    patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# 기본 Conda 환경 설정
RUN conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda config --set channel_priority flexible

# 메인 환경 생성 (Python 3.10)
RUN conda create -n main python=3.10 -y
ENV PATH="/opt/conda/envs/main/bin:${PATH}"
SHELL ["/bin/bash", "-c"]

# 기본 패키지 설치 
RUN conda install -n main -y \
    jupyterlab \
    pandas>=1.2 \
    numpy=1.26.4 \
    scipy=1.12.0 \
    scikit-learn>=0.22 \
    opencv>=4.2 \
    pyyaml>=5.1 \
    yacs>=0.1.6 \
    einops>=0.3 \
    tensorboard \
    psutil \
    tqdm \
    matplotlib \
    simplejson \
    pip \
    && conda clean -ya

# PyTorch 설치 (CUDA 12.1 지원)
RUN conda install -n main -y pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia && conda clean -ya

# MuJoCo 및 추가 패키지 설치
RUN source activate main && \
    pip install --no-cache-dir \
    mujoco==2.3.7 \
    mujoco-py==2.1.2.14 \
    gym==0.20.0 \
    dm_control==1.0.14 \
    brax==0.0.16 \
    imageio \
    tfp-nightly==0.20.0.dev20230524 \
    wandb \
    ml_collections \
    fvcore \
    av

# JAX 설치 (CUDA 12 지원)
RUN source activate main && \
    pip install --no-cache-dir \
    "jax[cuda12_pip]==0.4.19" --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    flax==0.7.5

# TimeSformer 관련 추가 패키지 설치
RUN source activate main && \
    pip install --no-cache-dir einops>=0.3

# 작업 디렉토리 설정
WORKDIR /workspace

# 기본 실행 명령 설정 
CMD ["bash", "-c", "source activate main && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
