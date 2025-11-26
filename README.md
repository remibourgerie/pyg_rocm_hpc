# PyTorch Geometric ROCm Installation for HPC Dardel

![ROCm 6.3.3](https://img.shields.io/badge/ROCm-6.3.3-orange) ![CPE 24.11](https://img.shields.io/badge/CPE-24.11-blue) ![Python 3.12](https://img.shields.io/badge/Python-3.12-green) ![PyTorch 2.7.1](https://img.shields.io/badge/PyTorch-2.7.1-red)

Pre-built PyTorch Geometric wheels with ROCm support for Cray HPC systems, plus reference installation scripts.

## What This Repository Provides

### 🎯 Primary Goal: Portable Wheels
The [`wheels/`](wheels/) directory contains **pre-built PyG extension wheels** compatible with Cray systems running software stack 24.11 and ROCm 6.3.3. These wheels should work on any similar Cray EX system without modification.

### 📝 Secondary: Reference Installation Scripts
The installation scripts (`install_pyg_rocm.sh`, etc.) are **environment-specific examples** tailored for [HPC Dardel](https://www.pdc.kth.se/hpc-services/computing-systems/dardel) at PDC/KTH. They contain hardcoded paths and module names specific to that environment. **These scripts are provided as templates to adapt to your own HPC system**, not for direct use.

### 💡 Use Case
- **Wheels**: Download and use directly on compatible Cray systems
- **Scripts**: Study and adapt to match your HPC environment's module system, paths, and allocations

## Problem Statement

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) does not officially support AMD ROCm. While the PyG team maintains CUDA builds, ROCm users must compile extensions from source. Existing community solutions like [Looong01's ROCm builds](https://github.com/Looong01/pyg-rocm-build) target Ubuntu 22 with newer compilers, making them incompatible with HPC environments running older toolchains.

## Solution

This repository provides:

- **HPC-Compatible Wheels**: Pre-built extensions compiled for Cray systems with CPE 24.11 and ROCm 6.3.3
- **AMD GPU Support**: Full PyG functionality on MI250X GPUs
- **Reference Installation Scripts**: Example workflows for HPC deployment (adapt to your environment)
- **Testing Suite**: Validation tools for GNN workloads
- **Build Scripts**: Reproducible wheel building for custom configurations

## Compatibility

### Pre-built Wheels
**Target Systems**: Cray EX systems with:
- Software stack CPE 24.11 (or similar compiler toolchain)
- ROCm 6.3.3
- AMD MI250X GPUs (or compatible gfx90a architecture)
- Python 3.12

**Included Wheels**:
- `torch_cluster-1.6.3-cp312-cp312-linux_x86_64.whl`
- `torch_scatter-2.1.2-cp312-cp312-linux_x86_64.whl`
- `torch_sparse-0.6.18-cp312-cp312-linux_x86_64.whl`
- `torch_spline_conv-1.2.2-cp312-cp312-linux_x86_64.whl`
- `torch_geometric-2.6.1-py3-none-any.whl`

The wheels should be portable across different Cray sites running this configuration.

### Installation Scripts
**Specific to**: [HPC Dardel](https://www.pdc.kth.se/hpc-services/computing-systems/dardel) at PDC (KTH Royal Institute of Technology)

**For other HPC systems**: Adapt the scripts by updating:
- Module names and versions (`ml` commands)
- File paths (repository clone location, conda paths)
- SLURM account/allocation names
- Job scheduler parameters

Community contributions for other HPC systems are welcome!

## Installation

### Option 1: Using Pre-built Wheels (Other Cray Systems)

If you're on a compatible Cray system with CPE 24.11 and ROCm 6.3.3, you can directly install the wheels:

```bash
# Clone or download this repository
git clone https://github.com/yourusername/pyg_rocm_hpc.git
cd pyg_rocm_hpc

# Install PyTorch with ROCm support first
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.2

# Install the pre-built PyG wheels
pip install wheels/torch_scatter-*.whl
pip install wheels/torch_sparse-*.whl
pip install wheels/torch_cluster-*.whl
pip install wheels/torch_spline_conv-*.whl
pip install wheels/torch_geometric-*.whl

# Verify installation
python -c "import torch, torch_geometric; print('PyG installation successful')"
```

### Option 2: Using Installation Scripts (HPC Dardel)

For HPC Dardel users, use the provided installation scripts (creates conda environment and submits test job):

```bash
chmod +x *.sh
./install_pyg_rocm.sh
```

### Option 3: Local Development (CPU)

For local CPU development and testing:

```bash
./install_pyg_cpu.sh
conda activate pyg_cpu
```

## Repository Contents

| Component | Description |
|-----------|-------------|
| `install_pyg_rocm.sh` | Complete HPC installation with ROCm 6.3.3 for AMD MI250X GPUs |
| `install_pyg_cpu.sh` | Local CPU installation for development and testing |
| `build_pyg_wheels.sh` | Custom PyG extension wheel builder optimized for ROCm |
| `test_pyg_rocm.py` | Comprehensive test file validating PyG + ROCm functionality |
| `test_pyg_rocm.slurm` | Production-ready SLURM job script for HPC testing |
| `wheels/` | Pre-built PyG extensions compiled for ROCm (torch_cluster, torch_scatter, etc.) |
| `home.bashrc` | Conda environment initialization for HPC workflow |

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
python -c "import torch, torch_geometric; print('PyG installation successful')"

# Comprehensive testing
python test_pyg_rocm.py

# Or submit test job
sbatch test_pyg_rocm.slurm
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

## Prerequisites

- Access to HPC Dardel or compatible Cray system
- Required tools: `conda`, `wget`, `unzip`, `git`, `gcc`
- For HPC: SLURM job scheduler, AMD MI250X GPUs

## Community Impact

This work bridges the gap between PyTorch Geometric's CUDA-centric ecosystem and AMD ROCm on HPC systems by providing compiler-compatible wheels and complete installation workflows for Cray environments.

## Disclaimer

This repository provides installation scripts and pre-built wheels for educational and research purposes. Please note:

- **No Warranty**: This software is provided "as is" without any warranties or guarantees
- **HPC Specificity**: Scripts are tailored for HPC Dardel and may require modifications for other systems
- **Pre-built Wheels**: The included wheels are compiled from open source PyG extensions but are not officially supported by the PyG team
- **System Requirements**: Ensure you have proper permissions and allocations before running HPC installations
- **Use at Your Own Risk**: Users are responsible for verifying compatibility with their specific HPC environment

For production use, always test thoroughly in your environment first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
