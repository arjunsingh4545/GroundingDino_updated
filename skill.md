---
name: GroundingDINO Custom Operations Setup & Architecture
description: Provides critical architectural context and C++ compatibility guidelines for AI agents working in this repository.
---

# GroundingDINO AI Engineering Skill

Welcome, agent. This repository is a specialized fork of GroundingDINO. You must understand its structure and the specific modifications made to its CUDA C++ extensions in order to navigate, debug, or extend this codebase effectively.

## Directory Structure

This project is separated into two primary domains:

- **`src/`**: Contains the high-level custom Python application logic, such as CLI wrappers, image processors, and inference pipelines that consume the GroundingDINO package.
- **`groundingdino/`**: Contains the core model architecture and the crucial C++ / CUDA extension source code (`csrc/`). This directory is built and installed locally as an editable pip package (`pip install -e .`).

## C++ & CUDA Refactoring Rules (PyTorch 2.14+ Compatibility)

The `csrc/` directory inside `groundingdino/` contains the `MultiScaleDeformableAttention` custom CUDA kernels. The original source was written for PyTorch 1.x and fails to compile on modern PyTorch 2.14+ environments due to deprecated ATen APIs.

If you ever need to touch the C++ or CUDA code in this repository, strictly adhere to the following rules that have already been applied:

1. **Tensor Device Checks (`.is_cuda()`)**:
   - ❌ **DEPRECATED**: `value.type().is_cuda()` or `value.scalar_type().is_cuda()`
   - ✅ **MODERN**: `value.is_cuda()`
   - *Reason*: Modern PyTorch removes these properties from the type objects.

2. **Tensor Data Pointers (`.data_ptr<T>()`)**:
   - ❌ **DEPRECATED**: `value.data<scalar_t>()`
   - ✅ **MODERN**: `value.data_ptr<scalar_t>()`
   - *Reason*: The old `.data()` method is entirely removed in recent PyTorch C++ APIs.

3. **AT_DISPATCH Macro Typing**:
   - When calling `AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.scalar_type(), ...)`, you MUST use `.scalar_type()` as the first argument, not `.type()`.

4. **Atomic Headers**:
   - ❌ **DEPRECATED**: `#include <THC/THCAtomics.cuh>`
   - ✅ **MODERN**: `#include <ATen/cuda/Atomic.cuh>`
   - *Reason*: The old THC library has been completely retired in PyTorch 2.x.

## Environment & Execution Guidelines for AI Agents

When executing commands or writing testing scripts, you must respect the Conda environment rules to avoid silent failures or CUDA kernel crashes.

1. **Always use the target Conda Environment**:
   The required environment is named `groundingdino_env`. Before running any `python` scripts, ensure your shell commands activate it:
   ```bash
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate groundingdino_env
   ```

2. **Crucial Variable: `CUDA_HOME`**:
   To successfully build the `groundingdino` pip package, the `setup.py` needs to find the `nvcc` compiler that matches the PyTorch installation. 
   When operating in the `groundingdino_env` (which has `cuda-toolkit` installed), you MUST export `CUDA_HOME` to the conda prefix before running `pip install -e .`:
   ```bash
   export CUDA_HOME="$CONDA_PREFIX"
   ```
   Failure to set this will cause `setup.py` to silently fail to compile the CUDA extension, falling back to a slow and buggy CPU stub.
