#!/bin/bash

# Docker run script for UAV Reinforcement Learning Suite
set -e

# Default values
IMAGE="drone-rl:production-latest"
CONTAINER_NAME="drone-rl"
GPU_ENABLED=true
JUPYTER_PORT=8888
TENSORBOARD_PORT=6006
COMMAND=""
INTERACTIVE=false
DAEMON=false
AIRSIM_CONFIG_PATH="${HOME}/Documents/AirSim"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Options:"
    echo "  -i, --image IMAGE       Docker image to run [default: drone-rl:production-latest]"
    echo "  -n, --name NAME         Container name [default: drone-rl]"
    echo "  -c, --cpu-only          Disable GPU support"
    echo "  -j, --jupyter-port PORT Jupyter port mapping [default: 8888]"
    echo "  -t, --tensorboard-port PORT TensorBoard port [default: 6006]"
    echo "  -a, --airsim-config PATH AirSim config path [default: \$HOME/Documents/AirSim]"
    echo "  -d, --daemon            Run in daemon mode"
    echo "  --interactive           Run in interactive mode with bash"
    echo "  --dev                   Shortcut for development mode"
    echo "  --jupyter               Shortcut for Jupyter mode"
    echo "  --training ALGORITHM    Quick training mode (ppo, sac, dqn, etc.)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                # Run default training"
    echo "  $0 --interactive                 # Interactive bash session"
    echo "  $0 --jupyter                     # Start Jupyter Lab"
    echo "  $0 --training ppo                # Start PPO training"
    echo "  $0 --dev                         # Development environment"
    echo "  $0 python dqn.py                 # Custom command"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -c|--cpu-only)
            GPU_ENABLED=false
            shift
            ;;
        -j|--jupyter-port)
            JUPYTER_PORT="$2"
            shift 2
            ;;
        -t|--tensorboard-port)
            TENSORBOARD_PORT="$2"
            shift 2
            ;;
        -a|--airsim-config)
            AIRSIM_CONFIG_PATH="$2"
            shift 2
            ;;
        -d|--daemon)
            DAEMON=true
            shift
            ;;
        --interactive)
            INTERACTIVE=true
            COMMAND="/bin/bash"
            shift
            ;;
        --dev)
            IMAGE="drone-rl:development-latest"
            INTERACTIVE=true
            COMMAND="/bin/bash"
            shift
            ;;
        --jupyter)
            IMAGE="drone-rl:jupyter-latest"
            COMMAND="jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"
            shift
            ;;
        --training)
            TRAINING_ALGO="$2"
            case $TRAINING_ALGO in
                ppo)
                    COMMAND="python train_ppo.py"
                    ;;
                sac)
                    COMMAND="python SAC.py"
                    ;;
                dqn)
                    COMMAND="python dqn.py"
                    ;;
                rainbow)
                    COMMAND="python rainbow.py"
                    ;;
                a3c)
                    COMMAND="python a3c.py"
                    ;;
                td3)
                    COMMAND="python td3.py"
                    ;;
                *)
                    echo -e "${RED}Error: Unknown training algorithm '$TRAINING_ALGO'${NC}"
                    echo "Supported algorithms: ppo, sac, dqn, rainbow, a3c, td3"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            print_usage
            exit 1
            ;;
        *)
            # Remaining arguments are the command to run
            COMMAND="$*"
            break
            ;;
    esac
done

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Build Docker run command
DOCKER_CMD="docker run --rm"

# Add daemon or interactive flags
if [[ "$DAEMON" == true ]]; then
    DOCKER_CMD="$DOCKER_CMD -d"
elif [[ "$INTERACTIVE" == true ]] || [[ -z "$COMMAND" ]]; then
    DOCKER_CMD="$DOCKER_CMD -it"
fi

# Add GPU support
if [[ "$GPU_ENABLED" == true ]]; then
    DOCKER_CMD="$DOCKER_CMD --gpus all"
    DOCKER_CMD="$DOCKER_CMD -e NVIDIA_VISIBLE_DEVICES=all"
fi

# Add container name
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"

# Add environment variables
DOCKER_CMD="$DOCKER_CMD -e DISPLAY=\${DISPLAY:-:0}"
DOCKER_CMD="$DOCKER_CMD -e QT_X11_NO_MITSHM=1"

# Add port mappings
DOCKER_CMD="$DOCKER_CMD -p $JUPYTER_PORT:8888"
DOCKER_CMD="$DOCKER_CMD -p $TENSORBOARD_PORT:6006"
DOCKER_CMD="$DOCKER_CMD -p 8050:8050"

# Add volume mounts
DOCKER_CMD="$DOCKER_CMD -v \$(pwd)/runs:/app/runs"
DOCKER_CMD="$DOCKER_CMD -v \$(pwd)/logs:/app/logs"
DOCKER_CMD="$DOCKER_CMD -v \$(pwd)/models:/app/models"
DOCKER_CMD="$DOCKER_CMD -v \$(pwd)/results:/app/results"
DOCKER_CMD="$DOCKER_CMD -v /tmp/.X11-unix:/tmp/.X11-unix:rw"

# Add AirSim config if it exists
if [[ -d "$AIRSIM_CONFIG_PATH" ]]; then
    DOCKER_CMD="$DOCKER_CMD -v $AIRSIM_CONFIG_PATH:/home/drone/Documents/AirSim"
fi

# Add image
DOCKER_CMD="$DOCKER_CMD $IMAGE"

# Add command if specified
if [[ -n "$COMMAND" ]]; then
    DOCKER_CMD="$DOCKER_CMD $COMMAND"
fi

echo -e "${BLUE}Starting Docker container...${NC}"
echo -e "Image: ${YELLOW}$IMAGE${NC}"
echo -e "Container: ${YELLOW}$CONTAINER_NAME${NC}"
echo -e "GPU Enabled: ${YELLOW}$GPU_ENABLED${NC}"
echo -e "Command: ${YELLOW}${COMMAND:-default}${NC}"
echo ""

# Create necessary directories
mkdir -p runs logs models results

echo -e "${GREEN}Running: ${YELLOW}$DOCKER_CMD${NC}"
echo ""

# Execute the Docker command
eval $DOCKER_CMD

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ Container started successfully${NC}"
    if [[ "$DAEMON" == true ]]; then
        echo -e "Container is running in daemon mode."
        echo -e "View logs: ${YELLOW}docker logs -f $CONTAINER_NAME${NC}"
        echo -e "Stop container: ${YELLOW}docker stop $CONTAINER_NAME${NC}"
    fi
    echo -e "Access TensorBoard: ${YELLOW}http://localhost:$TENSORBOARD_PORT${NC}"
    echo -e "Access Jupyter: ${YELLOW}http://localhost:$JUPYTER_PORT${NC}"
else
    echo -e "${RED}✗ Failed to start container${NC}"
    exit 1
fi