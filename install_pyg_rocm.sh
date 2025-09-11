#!/bin/bash

# PyTorch Geometric ROCm Installation Script
# This script automates the installation of PyTorch with ROCm support and PyTorch Geometric
# Compatible with ROCm 6.3 and Python 3.12

set -e  # Exit on any error

echo "========================================================"
echo "PyTorch Geometric + ROCm Installation Script"
echo "========================================================"

# Configuration
WORK_DIR="/cfs/klemming/projects/supr/remibo_kth_phd/pyg_dardel"
ENV_NAME="pyg_rocm"
PYTHON_VERSION="3.12"
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
TORCHAUDIO_VERSION="2.7.1"
ROCM_VERSION="6.3"
PYG_PACKAGE_URL="https://github.com/Looong01/pyg-rocm-build/releases/download/9/torch-2.6-rocm-6.2.4-py312-linux_x86_64.zip"
PYG_PACKAGE_NAME="torch-2.6-rocm-6.2.4-py312-linux_x86_64.zip"

echo "Configuration:"
echo "  Temp directory: ${WORK_DIR}"
echo "  Environment name: ${ENV_NAME}"
echo "  Python version: ${PYTHON_VERSION}"
echo "  PyTorch version: ${TORCH_VERSION}"
echo "  ROCm version: ${ROCM_VERSION}"
echo ""

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "[ERROR] $1 is not installed or not in PATH"
        exit 1
    fi
}

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

# Check prerequisites
echo "[INFO] Checking prerequisites..."
check_command conda
check_command wget
check_command unzip

# Check if environment already exists
check_env_exists

# Step 1: Create and navigate to temp directory
echo "[INFO] Step 1: Setting up temporary directory..."
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}
echo "[OK] Working directory: $(pwd)"

# Step 2: Create conda environment
echo "[INFO] Step 2: Creating conda environment..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
echo "[OK] Created conda environment: ${ENV_NAME}"

# Step 3: Activate conda environment
echo "[INFO] Step 3: Activating conda environment..."
source activate ${ENV_NAME}
echo "[OK] Activated environment: ${ENV_NAME}"
echo "[INFO] Python location: $(which python)"
echo "[INFO] Python version: $(python --version)"

# Step 4: Install PyTorch with ROCm support
echo "[INFO] Step 4: Installing PyTorch with ROCm support..."
pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION}
echo "[OK] PyTorch installation completed"

# Step 5: Check for existing wheels or build them
echo "[INFO] Step 5: Checking for PyTorch Geometric wheels..."
WHEELS_DIR="${WORK_DIR}/wheels"
mkdir -p ${WHEELS_DIR}

# Check if wheels already exist in wheels directory
WHEEL_FILES=(
    "torch_cluster-*.whl"
    "torch_scatter-*.whl"
    "torch_sparse-*.whl"
    "torch_spline_conv-*.whl"
    "torch_geometric-*.whl"
)

WHEELS_EXIST=true
for wheel_pattern in "${WHEEL_FILES[@]}"; do
    if ! ls ${WHEELS_DIR}/${wheel_pattern} 1> /dev/null 2>&1; then
        echo "[INFO] Missing wheel: ${wheel_pattern}"
        WHEELS_EXIST=false
        break
    fi
done

if [ "$WHEELS_EXIST" = true ]; then
    echo "[OK] All PyTorch Geometric wheels found in ${WHEELS_DIR}"
else
    echo "[INFO] Building PyTorch Geometric wheels..."
    if [ ! -f "build_pyg_wheels.sh" ]; then
        echo "[ERROR] build_pyg_wheels.sh not found in current directory"
        echo "[ERROR] Please ensure build_pyg_wheels.sh is present"
        exit 1
    fi
    
    # Make build script executable and run it
    chmod +x build_pyg_wheels.sh
    ./build_pyg_wheels.sh
    
    # Move built wheels to wheels directory
    if [ -d "dist" ]; then
        echo "[INFO] Moving built wheels to ${WHEELS_DIR}..."
        find dist -name "*.whl" -exec cp {} ${WHEELS_DIR}/ \;
        echo "[OK] Wheels moved to ${WHEELS_DIR}"
    else
        echo "[ERROR] No built wheels found after running build script"
        exit 1
    fi
fi

# Step 6: Download PyTorch Geometric main package if not present
echo "[INFO] Step 6: Checking PyTorch Geometric main package..."
PYG_WHEEL="torch_geometric-2.6.1-py3-none-any.whl"
if [ ! -f "${WHEELS_DIR}/${PYG_WHEEL}" ]; then
    echo "[INFO] Downloading PyTorch Geometric main package..."
    wget "https://files.pythonhosted.org/packages/03/9f/157e913626c1acfb3b19ce000b1a6e4e4fb177c0bc0ea0c67ca5bd714b5a/torch_geometric-2.6.1-py3-none-any.whl" -O "${WHEELS_DIR}/${PYG_WHEEL}"
    echo "[OK] Downloaded PyTorch Geometric main package"
else
    echo "[OK] PyTorch Geometric main package already present"
fi

# Step 7: Install PyTorch Geometric components from wheels directory
echo "[INFO] Step 7: Installing PyTorch Geometric components from wheels..."
echo "[INFO] Installing all wheel files in: ${WHEELS_DIR}"
pip install ${WHEELS_DIR}/*.whl --force-reinstall
echo "[OK] PyTorch Geometric installation completed"

# Step 8: Submit test job via SLURM
echo "[INFO] Step 8: Submitting test job via SLURM..."
cd ${WORK_DIR}
if [ -f "test_pyg_rocm.slurm" ]; then
    echo "[INFO] Found test_pyg_rocm.slurm, submitting job..."
    JOB_ID=$(sbatch test_pyg_rocm.slurm | grep -o '[0-9]*')
    echo "[OK] Test job submitted with ID: ${JOB_ID}"
    echo "[INFO] Monitor job status with: squeue -u $(whoami)"
    echo "[INFO] Check results when complete: cat test_pyg_rocm_${JOB_ID}.out"
else
    echo "[WARN] test_pyg_rocm.slurm not found in current directory"
    echo "[INFO] Please create the SLURM script and run: sbatch test_pyg_rocm.slurm"
    echo "[INFO] Or run basic verification manually:"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA/ROCm available: {torch.cuda.is_available()}')"
    if python -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')" 2>/dev/null; then
        echo "[OK] PyTorch Geometric successfully imported"
    else
        echo "[WARN] PyTorch Geometric import failed - check installation"
    fi
fi

# Step 9: Cleanup (optional)
echo "[INFO] Step 9: Cleanup options..."
read -p "Do you want to clean up downloaded files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[INFO] Cleaning up temporary files..."
    rm -f ${PYG_PACKAGE_NAME}
    rm -rf ${EXTRACT_DIR}
    echo "[OK] Cleanup completed"
else
    echo "[INFO] Keeping temporary files in: ${WORK_DIR}"
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
echo "To test the installation, run:"
echo "  python -c \"import torch, torch_geometric; print('All good!')\""
echo ""
echo "Environment details:"
echo "  Name: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}"
echo "  PyTorch: ${TORCH_VERSION} (ROCm ${ROCM_VERSION})"
echo "  Location: ${WORK_DIR}"
echo ""
echo "Next steps:"
echo "  1. Activate the environment: conda activate ${ENV_NAME}"
echo "  2. If test job wasn't submitted automatically:"
echo "     sbatch test_pyg_rocm.slurm"
echo "  3. Check the output file for results:"
echo "     cat test_pyg_rocm_<job_id>.out"
echo "  4. Install any additional packages you need"
echo "========================================================"