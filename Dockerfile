# Multi-stage Dockerfile for UAV Reinforcement Learning Suite
# Supports both CPU and GPU versions

# ===========================
# Stage 1: Base Image
# ===========================
ARG PYTHON_VERSION=3.8
ARG CUDA_VERSION=11.1
ARG UBUNTU_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libgphoto2-dev \
    libeigen3-dev \
    libhdf5-dev \
    doxygen \
    vim \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install pip and upgrade
RUN python -m pip install --upgrade pip setuptools wheel

# ===========================
# Stage 2: Python Dependencies
# ===========================
FROM base as python-deps

# Set working directory
WORKDIR /tmp

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for Docker environment
RUN pip install --no-cache-dir \
    jupyterlab \
    notebook \
    ipython \
    ipywidgets \
    plotly \
    dash

# ===========================
# Stage 3: Application
# ===========================
FROM python-deps as app

# Create app directory and user
RUN groupadd -r drone && useradd -r -g drone -d /app -s /bin/bash drone
WORKDIR /app

# Copy application code
COPY --chown=drone:drone . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/runs /app/logs /app/models /app/results && \
    chown -R drone:drone /app

# Set up AirSim configuration directory
RUN mkdir -p /home/drone/Documents/AirSim && \
    chown -R drone:drone /home/drone

# Switch to non-root user
USER drone

# ===========================
# Stage 4: Development Environment
# ===========================
FROM app as development

USER root

# Install development tools
RUN pip install --no-cache-dir \
    black \
    flake8 \
    pytest \
    pytest-cov \
    mypy \
    pre-commit

USER drone

# ===========================
# Stage 5: Production Environment
# ===========================
FROM app as production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch, numpy, gym; print('Health check passed')" || exit 1

# Expose ports
EXPOSE 8888 6006 8050

# Set default command
CMD ["python", "train_ppo.py", "--help"]

# ===========================
# Stage 6: Jupyter Environment
# ===========================
FROM app as jupyter

# Expose Jupyter port
EXPOSE 8888

# Create Jupyter config
RUN mkdir -p /home/drone/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/drone/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /home/drone/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /home/drone/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = False" >> /home/drone/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /home/drone/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /home/drone/.jupyter/jupyter_notebook_config.py

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# ===========================
# Final stage selection
# ===========================
FROM ${BUILD_TARGET:-production} as final