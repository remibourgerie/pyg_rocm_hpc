#!/bin/bash

# PyTorch Geometric CPU Installation Script
# This script automates the installation of PyTorch (CPU) and PyTorch Geometric
# Compatible with Python 3.12

set -e  # Exit on any error

echo "========================================================"
echo "PyTorch Geometric CPU Installation Script"
echo "========================================================"

# Configuration
WORK_DIR="$(pwd)"  # Use current directory (pyg_dardel)
ENV_NAME="pyg_cpu"
PYTHON_VERSION="3.12"
TORCH_VERSION="2.7.1"  # Compatible with PyG 2.6.1
TORCHVISION_VERSION="0.22.1"
TORCHAUDIO_VERSION="2.7.1"
PYG_VERSION="2.6.1"

echo "Configuration:"
echo "  Working directory: ${WORK_DIR}"
echo "  Environment name: ${ENV_NAME}"
echo "  Python version: ${PYTHON_VERSION}"
echo "  PyTorch version: ${TORCH_VERSION} (CPU)"
echo "  PyTorch Geometric version: ${PYG_VERSION}"
echo ""

# Function to check if conda environment exists
check_env_exists() {
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "[WARN] Conda environment '${ENV_NAME}' already exists"
        read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "[INFO] Removing existing environment..."
            conda env remove -n ${ENV_NAME} -y
        else
            echo "[INFO] Exiting to avoid conflicts"
            exit 1
        fi
    fi
}


# Check if environment already exists
check_env_exists

# Step 1: Verify working directory
echo "[INFO] Step 1: Using current directory as working directory..."
echo "[OK] Working directory: $(pwd)"

# Step 2: Create conda environment
echo "[INFO] Step 2: Creating conda environment..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
echo "[OK] Created conda environment: ${ENV_NAME}"

# Step 3: Activate conda environment
echo "[INFO] Step 3: Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
echo "[OK] Activated environment: ${ENV_NAME}"
echo "[INFO] Python location: $(which python)"
echo "[INFO] Python version: $(python --version)"

# Step 4: Install PyTorch (CPU version)
echo "[INFO] Step 4: Installing PyTorch (CPU version)..."
pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/cpu
echo "[OK] PyTorch CPU installation completed"

# Step 5: Install PyTorch Geometric and dependencies
echo "[INFO] Step 5: Installing PyTorch Geometric ${PYG_VERSION} and dependencies..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html
pip install torch-geometric==${PYG_VERSION}
echo "[OK] PyTorch Geometric ${PYG_VERSION} installation completed"

# Step 6: Install additional useful packages
echo "[INFO] Step 6: Installing additional packages..."
pip install numpy pandas matplotlib scikit-learn jupyter
echo "[OK] Additional packages installed"

# Step 7: Verify installation
echo "[INFO] Step 7: Verifying installation..."
python -c "
import torch
import torch_geometric
import torch_scatter
import torch_sparse
import torch_cluster
import torch_spline_conv

print('=' * 50)
print('Installation Verification')
print('=' * 50)
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch Geometric version: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')
print('')
print('PyG Extensions:')
print(f'  torch-scatter: {torch_scatter.__version__}')
print(f'  torch-sparse: {torch_sparse.__version__}')
print(f'  torch-cluster: {torch_cluster.__version__}')
print(f'  torch-spline-conv: {torch_spline_conv.__version__}')
print('')
print('Basic functionality test:')
from torch_geometric.data import Data
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
print(f'Created graph with {data.num_nodes} nodes and {data.num_edges} edges')
print('[SUCCESS] All components working correctly!')
"

if [ $? -eq 0 ]; then
    echo "[OK] Installation verification successful"
else
    echo "[ERROR] Installation verification failed"
    exit 1
fi

# Final instructions
echo ""
echo "========================================================"
echo "Installation Summary"
echo "========================================================"
echo "[SUCCESS] Installation completed!"
echo ""
echo "To use the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To test the installation:"
echo "  python -c \"import torch, torch_geometric; print('PyTorch Geometric ready!')\"" 
echo ""
echo "Environment details:"
echo "  Name: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}"
echo "  PyTorch: ${TORCH_VERSION} (CPU only)"
echo "  PyTorch Geometric: ${PYG_VERSION}"
echo "  Location: ${WORK_DIR}"
echo ""
echo "Installed packages:"
echo "  - torch (CPU)"
echo "  - torch-geometric"
echo "  - torch-scatter"
echo "  - torch-sparse" 
echo "  - torch-cluster"
echo "  - torch-spline-conv"
echo "  - numpy, pandas, matplotlib, scikit-learn, jupyter"
echo ""
echo "Next steps:"
echo "  1. Activate the environment: conda activate ${ENV_NAME}"
echo "  2. Start coding with PyTorch Geometric!"
echo "  3. For Jupyter notebooks: jupyter notebook"
echo "========================================================"