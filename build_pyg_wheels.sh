#!/bin/bash
set -e

echo "=========================================="
echo "PyTorch Geometric ROCm Installation"
echo "=========================================="

# Load modules
echo "[INFO] Loading modules..."
ml cpeGNU/24.11
ml gcc/12.2.0
ml rocm/6.3.3
ml craype-accel-amd-gfx90a
ml miniconda3/25.3.1-1-cpeGNU-24.11

# Activate conda environment
source activate pyg_rocm

# Set environment variables
export PYTORCH_ROCM_ARCH="gfx90a"
export FORCE_CUDA=0
export FORCE_HIP=1
export CC=$(which gcc)
export CXX=$(which g++)
export MAX_JOBS=4

echo "[INFO] Environment:"
echo "  Python: $(python --version)"
echo "  Compiler: $(g++ --version | head -1)"
echo "  ROCm: $ROCM_PATH"

# Create wheels directory
mkdir -p wheels
current_path=$(pwd)

# Package list
packages=("pytorch_cluster" "pytorch_scatter" "pytorch_sparse" "pytorch_spline_conv")

# Clone and build each package
for package in "${packages[@]}"; do
    echo "[INFO] Processing $package..."
    
    # Clean existing directory
    if [ -d "$package" ]; then
        rm -rf "$package"
    fi
    
    # Clone with submodules
    git clone --recursive "https://github.com/rusty1s/${package}.git"
    
    cd "$package"
    
    # Initialize submodules if they weren't cloned properly
    git submodule update --init --recursive
    
    # Clean and build
    rm -rf build/ dist/ *.egg-info/
    python setup.py bdist_wheel
    
    # Copy wheel to wheels directory
    cp dist/*.whl "$current_path/wheels/"
    
    # Install the wheel
    pip install dist/*.whl --force-reinstall
    
    cd "$current_path"
done

# Download and install specific PyTorch Geometric version
echo "[INFO] Installing PyTorch Geometric 2.6.1..."
wget "https://files.pythonhosted.org/packages/03/9f/157e913626c1acfb3b19ce000b1a6e4e4fb177c0bc0ea0c67ca5bd714b5a/torch_geometric-2.6.1-py3-none-any.whl" -O wheels/torch_geometric-2.6.1-py3-none-any.whl
pip install wheels/torch_geometric-2.6.1-py3-none-any.whl --force-reinstall

echo "[SUCCESS] Installation completed!"
echo "[INFO] All wheels saved in: $current_path/wheels/"
ls -la wheels/

# Test
python -c "
import torch_geometric
import torch_cluster, torch_scatter, torch_sparse, torch_spline_conv
print('✓ All components imported successfully')
print(f'PyTorch Geometric: {torch_geometric.__version__}')
"