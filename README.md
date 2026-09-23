# GroundingDINO Portable PyTorch 2.14+ Fork

This repository contains a specialized fork of [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) that has been refactored and modernized to compile seamlessly with **PyTorch 2.14+** and **CUDA 13.0** using the **C++20** standard.

Older versions of GroundingDINO suffer from compilation failures on modern architectures due to deprecated PyTorch ATen C++ APIs (e.g., `scalar_type().is_cuda()`, `data<T>()`, and missing atomic headers). This fork patches all of those issues directly in the C++ / CUDA source code, making it highly portable.

Additionally, it provides an intelligent, automated build script that handles setting up isolated CUDA compilation environments, ensuring `nvcc` and `CUDA_HOME` conflicts do not break the `_C` extension build process.

## Prerequisites

- **Linux** (Tested on Ubuntu — x86_64 and aarch64/Jetson)
- One of:
  - **Conda** (Miniconda or Anaconda), or
  - **Pixi** (auto-installed by the setup script if missing)

## Installation (1-Step Setup)

We provide two fully automated setup options. Both create an isolated Python 3.10 environment with PyTorch 2.14+, compile the CUDA extensions, and download model weights. **Pick one:**

### Option A — Pixi (Recommended)

```bash
bash environments/pixi_setup/setup.sh
```

### Option B — Conda

```bash
bash environments/conda_setup/setup.sh
```

See [`environments/README.md`](environments/README.md) for full details on each option.

## Usage

### With Pixi

```bash
pixi run --manifest-path environments/pixi_setup/pixi.toml python src/main.py --help

# Or open an interactive shell:
pixi shell --manifest-path environments/pixi_setup/pixi.toml
```

### With Conda

```bash
conda activate groundingdino_env
python src/main.py --help
```

## Troubleshooting: CUDA_HOME and Compilation

### Why does GroundingDINO fail to build `_C` ops on most systems?
The custom `ms_deform_attn` CUDA kernels must be compiled using an `nvcc` compiler that exactly matches the CUDA version your PyTorch binary was built against. If `CUDA_HOME` is unset or points to a mismatched system-wide CUDA toolkit (e.g., in `/usr/local/cuda`), the compilation will either fail silently (falling back to a slow CPU stub that crashes at runtime) or throw cryptic C++ errors.

### How this fork solves it:
The `setup.sh` script installs `cuda-toolkit` directly into the Conda environment and dynamically sets `export CUDA_HOME="$CONDA_PREFIX"`. This forces the `setup.py` build system to use the perfect isolated CUDA headers and `nvcc` compiler, guaranteeing a successful compilation on any host system regardless of the system-wide NVIDIA drivers or global CUDA toolkit versions.
