#!/bin/bash -l

# SLURM job parameters
#SBATCH --account=naiss2025-22-184
#SBATCH --job-name=PyGTest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --error=errors.out
#SBATCH --output=logs.out

echo "=========================================="
echo "PyTorch Geometric + ROCm Test Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=========================================="


# Load required modules
echo "[INFO] Loading ROCm modules..."
ml rocm/6.3.3
ml craype-accel-amd-gfx90a

# Load conda and activate environment
echo "[INFO] Activating conda environment..."
ml miniconda3/25.3.1-1-cpeGNU-24.11
source activate pyg_rocm

# Verify environment
echo "[INFO] Environment verification:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  Working directory: $(pwd)"

# Check if ROCm libraries are available
echo "[INFO] Checking ROCm library availability..."
ldconfig -p | grep -i hip || echo "[WARN] HIP libraries not found in ldconfig"
ldconfig -p | grep -i rocm || echo "[WARN] ROCm libraries not found in ldconfig"


echo "[INFO] ROCm environment variables:"
echo "  HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "  ROCM_PATH: $ROCM_PATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Run the test
echo "[INFO] Starting PyTorch Geometric + ROCm tests..."
srun --ntasks=1 python /cfs/klemming/projects/supr/remibo_kth_phd/pyg_dardel/test_pyg_rocm.py

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
