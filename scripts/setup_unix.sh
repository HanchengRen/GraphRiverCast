#!/usr/bin/env bash
# GraphRiverCast — Setup script for Linux / macOS
# Usage: bash scripts/setup_unix.sh
set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║        GraphRiverCast Environment Setup              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Detect OS ──
OS="$(uname -s)"
case "$OS" in
    Linux*)  PLATFORM="Linux" ;;
    Darwin*) PLATFORM="macOS" ;;
    *)       PLATFORM="Unknown: $OS" ;;
esac
echo "[1/5] Platform detected: $PLATFORM"

# ── Check conda ──
if command -v conda &> /dev/null; then
    echo "[2/5] Conda found: $(conda --version)"
    echo "      Creating environment from environment.yml ..."
    conda env create -f environment.yml || conda env update -f environment.yml
    echo ""
    echo "      Activate with:  conda activate graphrivercast"
    INSTALL_METHOD="conda"
else
    echo "[2/5] Conda not found, using pip ..."
    if command -v python3 &> /dev/null; then
        PY=python3
    else
        PY=python
    fi
    echo "      Python: $($PY --version)"
    $PY -m pip install --upgrade pip
    $PY -m pip install -r requirements.txt
    INSTALL_METHOD="pip"
fi

# ── Check CUDA ──
echo ""
echo "[3/5] Checking CUDA availability ..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "      GPU: $GPU_INFO"
else
    echo "      No NVIDIA GPU detected (CPU-only mode available)"
fi

# ── Verify installation ──
echo ""
echo "[4/5] Verifying imports ..."
if [ "$INSTALL_METHOD" = "conda" ]; then
    # Try to activate the conda env for verification
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate graphrivercast 2>/dev/null || true
fi
${PY:-python3} -c "
import torch
import torch_geometric
print(f'      PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')
print(f'      PyG {torch_geometric.__version__}')
from src.model import GraphRiverCast
print('      GraphRiverCast model imported successfully')
" 2>/dev/null && VERIFY_OK=1 || VERIFY_OK=0

if [ "$VERIFY_OK" = "1" ]; then
    echo ""
    echo "[5/5] ✓ Setup complete!"
else
    echo ""
    echo "[5/5] Import verification skipped (activate the environment first)"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Quick start:"
echo "    python -m src.inference \\"
echo "      --checkpoint checkpoints/pretrain/GRC_ColdStart.ckpt \\"
echo "      --data-dir ./data/global \\"
echo "      --start-date 2015-01-01"
echo "═══════════════════════════════════════════════════════"
