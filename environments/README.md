# Environment Setup

This directory contains two independent setup options for the GroundingDINO environment. **Choose one** — they both achieve the same result.

| Option | Directory | Best for |
|--------|-----------|----------|
| **Conda** | [`conda_setup/`](./conda_setup/) | Traditional conda/mamba users; systems with Anaconda already installed |
| **Pixi** | [`pixi_setup/`](./pixi_setup/) | Reproducible, lock-file-driven builds; Jetson (aarch64) + workstation (x86_64) from one config |

---

## Option 1 — Conda (Linux & Windows)

> **Prerequisite**: Conda (Miniconda / Anaconda) must be installed and on your `PATH`.
> For Windows users, **Visual Studio Build Tools (MSVC)** must also be installed to compile the C++ extensions.

**For Linux/macOS:**
```bash
bash environments/conda_setup/setup.sh
```

**For Windows (Command Prompt):**
```cmd
environments\conda_setup\setup.bat
```

After setup, activate the environment before running scripts:

```bash
conda activate groundingdino_env
python src/main.py --help
```

**Files:**
- [`conda_setup/setup.sh`](./conda_setup/setup.sh) — setup script (Linux)
- [`conda_setup/setup.bat`](./conda_setup/setup.bat) — setup script (Windows)
- [`conda_setup/envs/`](./conda_setup/envs/) — architecture-specific conda environment specs (auto-selected)

---

## Option 2 — Pixi (Recommended)

> **No prerequisites** — the script will auto-install Pixi if it's not found.

```bash
bash environments/pixi_setup/setup.sh
```

After setup, run scripts through Pixi (from the project root):

```bash
pixi run python src/main.py --help

# Or drop into an interactive shell:
pixi shell
```

**Files:**
- [`pixi_setup/setup.sh`](./pixi_setup/setup.sh) — setup script
- [`../pixi.toml`](../pixi.toml) — pixi workspace spec (x86_64 + aarch64)
- [`../pixi.lock`](../pixi.lock) — resolved lock file

---

## What each setup does

1. Creates an isolated Python 3.10 environment with PyTorch 2.14+ and a matching CUDA toolkit.
2. Compiles the GroundingDINO `MultiScaleDeformableAttention` CUDA C++ extensions.
3. Downloads the GroundingDINO model weights.
