#!/usr/bin/env bash
set -e

# ==============================================================================
# GroundingDINO Portable Environment Setup Script (Conda)
# ==============================================================================
# Run from project root:  bash environments/conda_setup/setup.sh

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Resolve project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🚀 Initializing GroundingDINO Portable Setup (Conda)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# 1. Check for conda
if ! command -v conda &>/dev/null; then
  echo -e "${RED}❌ Conda is not installed or not in your PATH. Please install Miniconda or Anaconda first.${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Conda detected.${NC}"

# 2. Initialize conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo -e "${GREEN}✓ Conda initialized for bash.${NC}"

# 3. Create or update conda environment
ENV_NAME="groundingdino_env"
ENV_FILE="$SCRIPT_DIR/environment.yml"
echo -e "\n${YELLOW}⏳ Setting up the Conda environment '$ENV_NAME' from environment.yml...${NC}"
if conda env list | grep -q "^$ENV_NAME\s"; then
  echo -e "${YELLOW}Environment already exists. Updating it...${NC}"
  conda env update -f "$ENV_FILE" --prune
else
  conda env create -f "$ENV_FILE"
fi
echo -e "${GREEN}✓ Environment setup complete.${NC}"

# 4. Activate environment
echo -e "\n${YELLOW}⏳ Activating '$ENV_NAME'...${NC}"
conda activate $ENV_NAME
echo -e "${GREEN}✓ Activated conda environment: $(which python)${NC}"

# 5. Verify nvcc availability and export CUDA_HOME
echo -e "\n${YELLOW}⏳ Verifying CUDA compiler (nvcc)...${NC}"
if ! command -v nvcc &>/dev/null; then
  echo -e "${RED}❌ nvcc not found in the environment. CUDA Toolkit installation may have failed.${NC}"
  exit 1
fi

export CUDA_HOME="$CONDA_PREFIX"
echo -e "${GREEN}✓ nvcc verified: $(nvcc --version | head -n 1)${NC}"
echo -e "${GREEN}✓ CUDA_HOME exported as: $CUDA_HOME${NC}"

# 6. Build the C++ extensions
echo -e "\n${YELLOW}⏳ Building GroundingDINO C++ CUDA Extensions...${NC}"
cd "$PROJECT_ROOT/groundingdino"
pip install -e . --no-build-isolation
cd "$PROJECT_ROOT"
echo -e "${GREEN}✓ GroundingDINO C++ Extension Built Successfully!${NC}"

# 7. Download model weights
echo -e "\n${YELLOW}⏳ Downloading GroundingDINO Model Weights...${NC}"
python src/download_weights.py
echo -e "${GREEN}✓ Weights downloaded!${NC}"

# 8. Success message
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  🎉 Setup Complete! The environment is ready.${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "You can now run the scripts in the ${YELLOW}src/${NC} directory."
echo -e "Don't forget to activate the environment before running your code:\n"
echo -e "    ${YELLOW}conda activate $ENV_NAME${NC}\n"
