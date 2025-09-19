#!/bin/bash
# Installation script for Drone RL AirSim

set -e  # Exit on any error

echo "🚁 Installing Drone RL AirSim"
echo "================================"

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required, found Python $python_version"
    exit 1
fi

echo "✅ Python $python_version found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version by default)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected, installing CUDA version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 Installing CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install base requirements
echo "📚 Installing base requirements..."
pip install -r requirements/base.txt

# Install training requirements if requested
if [ "$1" = "--training" ] || [ "$1" = "--all" ]; then
    echo "🎯 Installing training requirements..."
    pip install -r requirements/training.txt
fi

# Install evaluation requirements if requested
if [ "$1" = "--evaluation" ] || [ "$1" = "--all" ]; then
    echo "📊 Installing evaluation requirements..."
    pip install -r requirements/evaluation.txt
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{training,evaluation,models}
mkdir -p experiments/{logs,results,configs}
mkdir -p models/{checkpoints,final}

# Set up pre-commit hooks (optional)
if [ "$1" = "--dev" ] || [ "$1" = "--all" ]; then
    echo "🔧 Setting up development tools..."
    pip install pre-commit black flake8 isort
    pre-commit install
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Install and start AirSim"
echo "3. Run a quick test: python scripts/test_installation.py"
echo "4. Start training: python experiments/scripts/train.py --algorithm sac"
echo ""
echo "For detailed usage instructions, see docs/user_guide/quickstart.md"