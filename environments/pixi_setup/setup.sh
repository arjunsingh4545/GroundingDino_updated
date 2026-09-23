#!/usr/bin/env bash
# ==============================================================================
# GroundingDINO Portable Setup (Pixi — x86_64 & aarch64)
# ==============================================================================
# Designed to run seamlessly on a fresh system.
# Run from project root:  bash environments/pixi_setup/setup.sh
# ==============================================================================

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo -e "${BLUE}ℹ  $*${NC}"; }
success() { echo -e "${GREEN}✓  $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠  $*${NC}"; }
fail()    { echo -e "${RED}❌ $*${NC}"; exit 1; }

# Resolve project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || fail "Could not cd to project root: $PROJECT_ROOT"

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🚀 GroundingDINO Portable Setup (Pixi)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── 1. Detect architecture ───────────────────────────────────────────────────
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)
    info "Architecture detected: ${BOLD}x86_64 Workstation${NC}"
    ;;
  aarch64)
    info "Architecture detected: ${BOLD}aarch64 (Jetson / ARM)${NC}"
    ;;
  *)
    fail "Unsupported architecture: $ARCH. Only x86_64 and aarch64 are supported."
    ;;
esac

# ── 2. Verify NVIDIA driver / GPU (soft check — warn only) ──────────────────
echo ""
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  if [ -n "$GPU_NAME" ]; then
    success "NVIDIA GPU found: ${BOLD}$GPU_NAME${NC}  (driver $DRIVER_VER)"
  else
    warn "nvidia-smi is installed but no GPU was detected. CUDA builds may fail."
  fi
elif [ "$ARCH" = "aarch64" ] && [ -d /usr/local/cuda ]; then
  success "Jetson CUDA toolkit found at /usr/local/cuda"
else
  warn "nvidia-smi not found. CUDA compilation may fail if no GPU/driver is available."
  warn "Continuing anyway — Pixi will attempt to resolve packages."
fi

# ── 3. Install Pixi if missing ──────────────────────────────────────────────
echo ""
if command -v pixi &>/dev/null; then
  PIXI_VER=$(pixi --version 2>/dev/null || echo "unknown")
  success "Pixi already installed: ${BOLD}$PIXI_VER${NC}  ($(which pixi))"
else
  warn "Pixi is not installed. Attempting automatic installation..."
  if ! command -v curl &>/dev/null; then
    fail "curl is required to install Pixi but was not found.\n   Install curl first:  sudo apt-get install -y curl"
  fi
  if curl -fsSL https://pixi.sh/install.sh | bash; then
    # The installer adds pixi to ~/.pixi/bin — source the updated PATH
    export PATH="$HOME/.pixi/bin:$PATH"
    if command -v pixi &>/dev/null; then
      success "Pixi installed successfully: $(pixi --version)"
    else
      fail "Pixi installer finished but 'pixi' is still not on PATH.\n   Try opening a new terminal and re-running this script."
    fi
  else
    fail "Automatic Pixi installation failed.\n   Install manually: https://pixi.sh"
  fi
fi

# ── 4. Verify pixi.toml exists ──────────────────────────────────────────────
echo ""
MANIFEST_PATH="$SCRIPT_DIR/pixi.toml"
if [ ! -f "$MANIFEST_PATH" ]; then
  fail "pixi.toml not found at $MANIFEST_PATH"
fi
success "Found pixi.toml"

# ── 5. Resolve packages & build ─────────────────────────────────────────────
echo ""
info "Resolving packages and building the environment..."
info "This may take several minutes on the first run."
echo ""

if pixi run --manifest-path "$MANIFEST_PATH" setup; then
  echo ""
  success "Pixi environment built and all tasks completed!"
else
  EXIT_CODE=$?
  echo ""
  echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${RED}  ⚠  Build failed (exit code $EXIT_CODE)${NC}"
  echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  echo -e "${YELLOW}Common causes & fixes:${NC}"
  echo -e "  1. ${BOLD}Missing NVIDIA driver${NC}: Install the latest NVIDIA driver for your GPU."
  echo -e "     Check with:  ${CYAN}nvidia-smi${NC}"
  echo -e ""
  echo -e "  2. ${BOLD}CUDA version mismatch${NC}: The pixi.toml expects CUDA 12.4."
  echo -e "     On Jetson, ensure /usr/local/cuda is CUDA 12.x+."
  echo -e ""
  echo -e "  3. ${BOLD}Missing C/C++ compiler${NC}: Install build essentials."
  echo -e "     On Ubuntu/Debian:  ${CYAN}sudo apt-get install -y build-essential${NC}"
  echo -e ""
  echo -e "  4. ${BOLD}Network issues${NC}: Package resolution requires internet access."
  echo -e "     Retry with:  ${CYAN}pixi run --manifest-path environments/pixi_setup/pixi.toml setup${NC}"
  echo -e ""
  echo -e "  5. ${BOLD}Stale lock file${NC}: Delete and re-resolve."
  echo -e "     ${CYAN}rm environments/pixi_setup/pixi.lock && pixi run --manifest-path environments/pixi_setup/pixi.toml setup${NC}"
  echo ""
  exit $EXIT_CODE
fi

PIXI_FLAG="--manifest-path environments/pixi_setup/pixi.toml"

# ── 6. Success banner & usage guide ─────────────────────────────────────────
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  🎉 Setup Complete! The environment is ready.${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BOLD}How to run scripts (from project root):${NC}"
echo ""
echo -e "  ${CYAN}# Run the main inference script${NC}"
echo -e "  ${YELLOW}pixi run $PIXI_FLAG python src/main.py --help${NC}"
echo ""
echo -e "  ${CYAN}# Run any Python file inside the Pixi environment${NC}"
echo -e "  ${YELLOW}pixi run $PIXI_FLAG python src/<your_script>.py${NC}"
echo ""
echo -e "  ${CYAN}# Open an interactive shell with the environment activated${NC}"
echo -e "  ${YELLOW}pixi shell $PIXI_FLAG${NC}"
echo ""
echo -e "  ${CYAN}# Re-run the full build (extensions + weight download)${NC}"
echo -e "  ${YELLOW}pixi run $PIXI_FLAG setup${NC}"
echo ""
echo -e "  ${CYAN}# Re-build only the C++ CUDA extensions${NC}"
echo -e "  ${YELLOW}pixi run $PIXI_FLAG build-ext${NC}"
echo ""
echo -e "  ${CYAN}# Re-download model weights only${NC}"
echo -e "  ${YELLOW}pixi run $PIXI_FLAG download-weights${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

