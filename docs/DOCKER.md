# Docker Deployment Guide

This guide explains how to build and run the UAV Reinforcement Learning Suite using Docker.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 1.28+
- NVIDIA Docker runtime (for GPU support)
- At least 8GB of available disk space

### GPU Support Setup

For GPU acceleration, install NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Simple setup - CPU training
docker-compose -f docker-compose.simple.yml up

# Full setup with GPU support
docker-compose up drone-rl-gpu

# Start Jupyter Lab
docker-compose up jupyter

# Development environment
docker-compose up dev
```

### Using Build Scripts

```bash
# Build production image
./docker/build.sh

# Build development image
./docker/build.sh -t development

# Build and push to registry
./docker/build.sh --push -r your-registry.com/drone-rl
```

### Using Run Scripts

```bash
# Default training
./docker/run.sh

# Interactive development
./docker/run.sh --dev

# Start Jupyter Lab
./docker/run.sh --jupyter

# Train specific algorithm
./docker/run.sh --training ppo
./docker/run.sh --training sac
```

## Build Targets

The Dockerfile supports multiple build targets:

### Production (Default)
- Optimized for training and inference
- Minimal runtime dependencies
- Non-root user for security

```bash
docker build --target production -t drone-rl:production .
```

### Development
- Includes development tools (black, flake8, pytest)
- Interactive debugging capabilities
- Code linting and testing

```bash
docker build --target development -t drone-rl:dev .
```

### Jupyter
- Pre-configured Jupyter Lab environment
- All visualization tools included
- Notebook-friendly setup

```bash
docker build --target jupyter -t drone-rl:jupyter .
```

## Manual Docker Commands

### Build Image
```bash
docker build -t drone-rl:latest .

# With GPU support
docker build --build-arg CUDA_VERSION=11.1 -t drone-rl:gpu .

# CPU only
docker build --build-arg BASE_IMAGE=python:3.8-slim -t drone-rl:cpu .
```

### Run Container

#### Basic Training
```bash
docker run --rm -it \
  --gpus all \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/models:/app/models \
  drone-rl:latest
```

#### Interactive Development
```bash
docker run --rm -it \
  --gpus all \
  -v $(pwd):/app \
  -p 8888:8888 \
  -p 6006:6006 \
  drone-rl:dev \
  /bin/bash
```

#### Jupyter Lab
```bash
docker run --rm -it \
  --gpus all \
  -v $(pwd):/app \
  -p 8888:8888 \
  drone-rl:jupyter
```

#### With AirSim Integration
```bash
docker run --rm -it \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/Documents/AirSim:/home/drone/Documents/AirSim \
  -p 6006:6006 \
  drone-rl:latest
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_VISIBLE_DEVICES` | GPU devices to use | `all` |
| `CUDA_VISIBLE_DEVICES` | CUDA device selection | `0` |
| `PYTHONPATH` | Python module path | `/app` |
| `OMP_NUM_THREADS` | OpenMP threads | `4` |
| `MKL_NUM_THREADS` | MKL threads | `4` |

## Volume Mounts

### Recommended Mounts
```bash
-v $(pwd)/runs:/app/runs              # TensorBoard logs
-v $(pwd)/models:/app/models          # Saved models
-v $(pwd)/results:/app/results        # Training results
-v $(pwd)/data:/app/data              # Training data
-v $(pwd)/configs:/app/configs        # Configuration files
```

### AirSim Integration
```bash
-v $HOME/Documents/AirSim:/home/drone/Documents/AirSim
```

### X11 Forwarding (for GUI)
```bash
-e DISPLAY=$DISPLAY
-v /tmp/.X11-unix:/tmp/.X11-unix:rw
```

## Port Mappings

| Port | Service | Description |
|------|---------|-------------|
| 6006 | TensorBoard | Training visualization |
| 8888 | Jupyter | Notebook interface |
| 8050 | Dash/Plotly | Interactive plots |
| 41451 | AirSim API | Simulation interface |

## Docker Compose Services

### Available Services

- `drone-rl-gpu`: GPU-enabled training
- `drone-rl-cpu`: CPU-only training
- `jupyter`: Jupyter Lab environment
- `dev`: Development environment
- `tensorboard`: Standalone TensorBoard
- `airsim`: AirSim simulation (optional)

### Service Commands
```bash
# Start specific service
docker-compose up <service-name>

# Scale training services
docker-compose up --scale drone-rl-gpu=2

# View logs
docker-compose logs -f <service-name>

# Stop all services
docker-compose down
```

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.1-base nvidia-smi

# Verify Docker GPU support
docker info | grep -i nvidia
```

#### Permission Issues
```bash
# Fix volume permissions
sudo chown -R $USER:$USER runs/ models/ results/

# Or run with user mapping
docker run --user $(id -u):$(id -g) ...
```

#### X11 Connection Issues
```bash
# Allow X11 forwarding
xhost +local:docker

# Or use more secure approach
xhost +SI:localuser:$(whoami)
```

#### Memory Issues
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Check container resources
docker stats
```

### Performance Tuning

#### CPU Optimization
```bash
docker run \
  -e OMP_NUM_THREADS=8 \
  -e MKL_NUM_THREADS=8 \
  --cpus="8" \
  drone-rl:latest
```

#### Memory Limits
```bash
docker run \
  --memory="16g" \
  --memory-swap="24g" \
  drone-rl:latest
```

#### Shared Memory
```bash
docker run \
  --shm-size=2g \
  drone-rl:latest
```

## Multi-Stage Build Details

The Dockerfile uses multi-stage builds for optimization:

1. **Base**: System dependencies and Python setup
2. **Python-deps**: Python package installation
3. **App**: Application code and user setup
4. **Development**: Additional dev tools
5. **Production**: Optimized runtime
6. **Jupyter**: Notebook environment

## Security Considerations

- Containers run as non-root user `drone`
- Minimal attack surface in production image
- No sensitive data in image layers
- Security updates applied during build

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Build Docker image
  run: |
    docker build -t ${{ env.IMAGE_NAME }}:${{ github.sha }} .
    docker tag ${{ env.IMAGE_NAME }}:${{ github.sha }} ${{ env.IMAGE_NAME }}:latest

- name: Push to registry
  run: |
    docker push ${{ env.IMAGE_NAME }}:${{ github.sha }}
    docker push ${{ env.IMAGE_NAME }}:latest
```

### GitLab CI Example
```yaml
build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

## Best Practices

1. **Layer Caching**: Order Dockerfile instructions by frequency of change
2. **Multi-stage Builds**: Use separate stages for different environments
3. **Security**: Run as non-root user, scan for vulnerabilities
4. **Size Optimization**: Remove unnecessary packages and files
5. **Resource Limits**: Set appropriate CPU and memory limits
6. **Health Checks**: Include health check commands
7. **Logging**: Use structured logging with appropriate levels
8. **Secrets**: Never include secrets in images, use external sources