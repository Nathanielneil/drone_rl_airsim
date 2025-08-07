#!/bin/bash

# Docker build script for UAV Reinforcement Learning Suite
set -e

# Default values
TARGET="production"
TAG="latest"
PLATFORM="linux/amd64"
PUSH=false
REGISTRY=""
CUDA_VERSION="11.1"
PYTHON_VERSION="3.8"
UBUNTU_VERSION="20.04"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --target TARGET     Build target (production, development, jupyter) [default: production]"
    echo "  -g, --tag TAG           Docker image tag [default: latest]"
    echo "  -p, --platform PLATFORM Docker platform [default: linux/amd64]"
    echo "  -r, --registry REGISTRY Docker registry URL for pushing"
    echo "  --push                  Push image to registry after build"
    echo "  --cuda-version VERSION  CUDA version [default: 11.1]"
    echo "  --python-version VERSION Python version [default: 3.8]"
    echo "  --ubuntu-version VERSION Ubuntu version [default: 20.04]"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Build production image"
    echo "  $0 -t development                    # Build development image"
    echo "  $0 -t jupyter -g v1.0.0             # Build Jupyter image with tag v1.0.0"
    echo "  $0 --push -r myregistry.com/drone-rl # Build and push to registry"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -g|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --ubuntu-version)
            UBUNTU_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate target
if [[ "$TARGET" != "production" && "$TARGET" != "development" && "$TARGET" != "jupyter" ]]; then
    echo -e "${RED}Error: Invalid target '$TARGET'. Must be one of: production, development, jupyter${NC}"
    exit 1
fi

# Set image name
if [[ -n "$REGISTRY" ]]; then
    IMAGE_NAME="$REGISTRY/drone-rl"
else
    IMAGE_NAME="drone-rl"
fi

FULL_IMAGE_NAME="$IMAGE_NAME:$TARGET-$TAG"

echo -e "${BLUE}Building Docker image...${NC}"
echo -e "Target: ${YELLOW}$TARGET${NC}"
echo -e "Image: ${YELLOW}$FULL_IMAGE_NAME${NC}"
echo -e "Platform: ${YELLOW}$PLATFORM${NC}"
echo -e "CUDA Version: ${YELLOW}$CUDA_VERSION${NC}"
echo -e "Python Version: ${YELLOW}$PYTHON_VERSION${NC}"
echo -e "Ubuntu Version: ${YELLOW}$UBUNTU_VERSION${NC}"
echo ""

# Build the Docker image
docker build \
    --platform "$PLATFORM" \
    --target "$TARGET" \
    --build-arg BUILD_TARGET="$TARGET" \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    --build-arg UBUNTU_VERSION="$UBUNTU_VERSION" \
    -t "$FULL_IMAGE_NAME" \
    .

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ Docker image built successfully: $FULL_IMAGE_NAME${NC}"
    
    # Also tag as latest for the target
    docker tag "$FULL_IMAGE_NAME" "$IMAGE_NAME:$TARGET-latest"
    echo -e "${GREEN}✓ Tagged as: $IMAGE_NAME:$TARGET-latest${NC}"
    
    # Push to registry if requested
    if [[ "$PUSH" == true ]]; then
        if [[ -z "$REGISTRY" ]]; then
            echo -e "${RED}Error: Registry URL required for push operation${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}Pushing to registry...${NC}"
        docker push "$FULL_IMAGE_NAME"
        docker push "$IMAGE_NAME:$TARGET-latest"
        
        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✓ Image pushed successfully${NC}"
        else
            echo -e "${RED}✗ Failed to push image${NC}"
            exit 1
        fi
    fi
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "To run the container:"
echo -e "${YELLOW}docker run --rm -it --gpus all $FULL_IMAGE_NAME${NC}"