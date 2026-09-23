# Environment Setup

This directory contains two independent setup options for the GroundingDINO environment. **Choose one** — they both achieve the same result.

| Option | Directory | Best for |
|--------|-----------|----------|
| **Conda** | [`conda_setup/`](./conda_setup/) | Traditional conda/mamba users; systems with Anaconda already installed |
| **Pixi** | [`pixi_setup/`](./pixi_setup/) | Reproducible, lock-file-driven builds; Jetson (aarch64) + workstation (x86_64) from one config |

---

## Option 1 — Conda

> **Prerequisite**: Conda (Miniconda / Anaconda) must be installed and on your `PATH`.

```bash
bash environments/conda_setup/setup.sh
```

After setup, activate the environment before running scripts:

```bash
conda activate groundingdino_env
python src/main.py --help
```

**Files:**
- [`conda_setup/setup.sh`](./conda_setup/setup.sh) — setup script
- [`conda_setup/environment.yml`](./conda_setup/environment.yml) — conda environment spec

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
