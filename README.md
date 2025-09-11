# 🚀 PyG Dardel - Unlocking PyTorch Geometric on HPC

**Bringing Graph Neural Networks to AMD GPUs** — Automated PyTorch Geometric installation with ROCm support for [HPC Dardel](https://www.pdc.kth.se/hpc-services/computing-systems/dardel) and beyond.

## 💡 The Challenge

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) doesn't officially support AMD ROCm out of the box. While the PyG team maintains CUDA builds, ROCm users face a complex compilation process. Existing community solutions like [Looong01's ROCm builds](https://github.com/Looong01/pyg-rocm-build) target Ubuntu 22 with newer compilers, making them incompatible with HPC environments running older toolchains.

## 🎯 What This Solves

This project addresses the HPC-specific challenges by providing:

- **🔧 HPC-Compatible Wheels**: Pre-built extensions compiled specifically for Cray systems with older compilers
- **⚡ AMD GPU Acceleration**: Full PyG functionality on MI250X with ROCm 6.3.3
- **📦 Complete Environment**: Ready-to-use conda setup that just works on HPC infrastructure
- **🧪 Comprehensive Testing**: Verify your installation works with real GNN workloads
- **🏗️ Reproducible Build System**: Scripts to rebuild wheels for your specific environment

> **The Key Innovation**: While community wheels exist for Ubuntu, HPC systems require wheels built with compatible compiler toolchains. This repository solves that compatibility gap.

## Repository Contents

| Component | Description |
|-----------|-------------|
| `install_pyg_rocm.sh` | Complete HPC installation with ROCm 6.3.3 for AMD MI250X GPUs |
| `install_pyg_cpu.sh` | Local CPU installation for development and testing |
| `build_pyg_wheels.sh` | Custom PyG extension wheel builder optimized for ROCm |
| `test_pyg_rocm.py` | Comprehensive test suite validating PyG + ROCm functionality |
| `test_pyg_rocm.slurm` | Production-ready SLURM job script for HPC testing |
| `wheels/` | Pre-built PyG extensions compiled for ROCm (torch_cluster, torch_scatter, etc.) |
| `home.bashrc` | Conda environment initialization for HPC workflow |

## Quick Start

### HPC Dardel (ROCm + GPU)
```bash
chmod +x *.sh
./install_pyg_rocm.sh
```

### Local Development (CPU)
```bash
./install_pyg_cpu.sh
conda activate pyg_cpu
```

## Technical Specifications

### HPC Environment (Dardel)
- **System**: Cray EX with CPE 24.11
- **GPUs**: AMD MI250X with ROCm 6.3.3
- **Software**: Python 3.12, PyTorch 2.7.1, PyG 2.6.1
- **Environment**: `pyg_rocm` conda environment

### Local Development
- **Software**: Python 3.12, PyTorch 2.7.1 (CPU-only), PyG 2.6.1  
- **Environment**: `pyg_cpu` conda environment

### Installation Behavior
- **HPC**: Creates environment, installs dependencies, and automatically submits test job
- **Local**: Creates mirrored environment for development and testing

## Usage Examples

### SLURM Job Template
```bash
#!/bin/bash -l
#SBATCH --account=your-allocation    # Replace with your allocation
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=64GB

# Load required modules
ml rocm/6.3.3
ml craype-accel-amd-gfx90a
ml miniconda3/25.3.1-1-cpeGNU-24.11

# Activate environment and run
conda activate pyg_rocm
python your_script.py
```

### Testing Installation
```bash
# Quick verification
python -c "import torch, torch_geometric; print('✓ PyG installation successful!')"

# Comprehensive testing
python test_pyg_rocm.py

# Or submit test job
sbatch test_pyg_rocm.slurm
```

## Prerequisites

- Access to HPC Dardel or compatible Cray system
- Required tools: `conda`, `wget`, `unzip`, `git`, `gcc`
- For HPC: SLURM job scheduler, AMD MI250X GPUs

## Community Impact

This work was developed to bridge the gap between PyTorch Geometric's CUDA-centric ecosystem and the growing AMD ROCm community, specifically addressing HPC environment challenges. While [Looong01's Ubuntu-based wheels](https://github.com/Looong01/pyg-rocm-build) provided a foundation, they weren't compatible with HPC systems using older compiler toolchains. 

This repository provides HPC-specific solutions by:
- **Rebuilding wheels** with compatible compilers for Cray/HPC environments
- **Creating complete installation workflows** that handle the complexity of HPC module systems
- **Providing validation tools** to ensure everything works in production HPC environments

The goal is to lower the barrier for HPC researchers wanting to explore Graph Neural Networks on AMD hardware without getting stuck in compilation hell.

## Compatibility

**Primary Target**: [HPC Dardel](https://www.pdc.kth.se/hpc-services/computing-systems/dardel) at PDC (KTH Royal Institute of Technology)

**Potential Compatibility**: Other Cray EX systems with software stack 24.11 (community testing welcome)

## Acknowledgments

- PyTorch Geometric team for the incredible [PyG framework](https://github.com/pyg-team/pytorch_geometric)
- [@Looong01](https://github.com/Looong01) for maintaining [ROCm-compiled PyG wheels](https://github.com/Looong01/pyg-rocm-build)
- PDC Center for High Performance Computing for providing Dardel infrastructure

---

*Developed at KTH Royal Institute of Technology*