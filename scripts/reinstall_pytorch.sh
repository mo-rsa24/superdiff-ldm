#!/bin/bash
# Safe PyTorch reinstallation script for CUDA compatibility issues
# Usage: ./scripts/reinstall_pytorch.sh [cuda_version]

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================================================="
echo "PyTorch Safe Reinstallation Script"
echo "======================================================================="

# Check if we're in a conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "❌ Error: No conda environment activated."
    echo "   Please activate your environment first:"
    echo "   conda activate superdiff"
    exit 1
fi

echo "📦 Current environment: $CONDA_DEFAULT_ENV"

# Backup current package list
BACKUP_FILE="$PROJECT_ROOT/logs/package_backup_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$PROJECT_ROOT/logs"

echo "💾 Backing up current package list to: $BACKUP_FILE"
pip list > "$BACKUP_FILE"

# Check current PyTorch installation
echo ""
echo "🔍 Current PyTorch installation:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}')" 2>/dev/null || echo "  PyTorch not installed"

# Ask for confirmation
echo ""
read -p "⚠️  Do you want to uninstall PyTorch packages? (yes/no): " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "❌ Aborted by user."
    exit 0
fi

# Uninstall PyTorch and related packages
echo ""
echo "🗑️  Uninstalling PyTorch packages..."
pip uninstall torch torchvision torchaudio triton -y || true

# Clean pip cache
echo ""
echo "🧹 Cleaning pip cache..."
pip cache purge

# Determine which requirements file to use
USE_CUDA_OVERRIDE=false
if [[ -f "$PROJECT_ROOT/requirements_cuda.txt" ]]; then
    echo ""
    echo "📋 Found requirements_cuda.txt (CUDA-specific overrides)"
    read -p "   Use CUDA overrides for RTX 5090/newer GPUs? (yes/no): " use_cuda
    if [[ "$use_cuda" == "yes" ]]; then
        USE_CUDA_OVERRIDE=true
    fi
fi

# Install PyTorch
echo ""
if [[ "$USE_CUDA_OVERRIDE" == "true" ]]; then
    echo "📥 Installing PyTorch from requirements_cuda.txt..."
    pip install -r "$PROJECT_ROOT/requirements_cuda.txt"
else
    echo "📥 Installing PyTorch from requirements_clean.txt..."
    # Only install PyTorch-related packages
    grep -E "^(torch|torchvision|torchaudio|triton)==" "$PROJECT_ROOT/requirements_clean.txt" | pip install -r /dev/stdin
fi

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "
import torch
print(f'✅ PyTorch {torch.__version__} installed successfully')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
"

# Run compatibility check
echo ""
echo "🔍 Running compatibility check..."
python "$PROJECT_ROOT/scripts/check_cuda_compatibility.py"

echo ""
echo "======================================================================="
echo "✅ Reinstallation complete!"
echo "   Backup saved to: $BACKUP_FILE"
echo "======================================================================="
