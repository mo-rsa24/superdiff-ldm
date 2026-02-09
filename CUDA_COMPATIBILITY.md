# CUDA Compatibility Guide for SuperDiff-LDM

## Overview

This project uses PyTorch with CUDA acceleration. Different GPUs require different PyTorch versions based on their compute capability.

## Current Setup

- **Base requirements**: `requirements_clean.txt` - Contains stable versions for most GPUs
- **CUDA overrides**: `requirements_cuda.txt` - Contains newer versions for RTX 5090 and other newer GPUs

## GPU Compute Capabilities

| GPU Family | Compute Capability | PyTorch Requirement | Notes |
|------------|-------------------|---------------------|-------|
| RTX 5090   | sm_120 (12.0)     | PyTorch 2.6+        | Requires requirements_cuda.txt |
| RTX 4090   | sm_89 (8.9)       | PyTorch 2.0+        | Use requirements_clean.txt |
| RTX 3090   | sm_86 (8.6)       | PyTorch 1.9+        | Use requirements_clean.txt |
| RTX 3080   | sm_86 (8.6)       | PyTorch 1.9+        | Use requirements_clean.txt |

## Common Issue: RTX 5090 Compatibility Warning

If you see this warning:
```
UserWarning: NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**Solution**: Your PyTorch version is too old for the RTX 5090. Follow the reinstallation steps below.

## Checking Compatibility

Run the compatibility checker:

```bash
conda activate superdiff
python scripts/check_cuda_compatibility.py
```

This will detect your GPU, CUDA version, and PyTorch installation, and report any compatibility issues.

## Safe Reinstallation Process

### Option 1: Automated Script (Recommended)

```bash
conda activate superdiff
chmod +x scripts/reinstall_pytorch.sh
./scripts/reinstall_pytorch.sh
```

The script will:
1. Backup your current package list
2. Safely uninstall PyTorch packages
3. Ask which requirements file to use
4. Install the correct PyTorch version
5. Verify the installation

### Option 2: Manual Reinstallation

For **RTX 5090** (sm_120) or newer GPUs:

```bash
conda activate superdiff

# 1. Backup current packages (optional but recommended)
pip list > package_backup_$(date +%Y%m%d).txt

# 2. Uninstall PyTorch packages
pip uninstall torch torchvision torchaudio triton -y

# 3. Clean cache
pip cache purge

# 4. Install newer PyTorch for RTX 5090
pip install -r requirements_cuda.txt

# 5. Verify
python scripts/check_cuda_compatibility.py
```

For **older GPUs** (RTX 3080, 3090, 4090, etc.):

```bash
conda activate superdiff

# Uninstall and reinstall from base requirements
pip uninstall torch torchvision torchaudio triton -y
pip cache purge
pip install -r requirements_clean.txt
python scripts/check_cuda_compatibility.py
```

## Monitoring CUDA Version Changes

### When CUDA/Driver Updates

If you update your NVIDIA driver or CUDA toolkit:

1. **Check compatibility first**:
   ```bash
   python scripts/check_cuda_compatibility.py
   ```

2. **If incompatible**, reinstall PyTorch:
   ```bash
   ./scripts/reinstall_pytorch.sh
   ```

### When Switching GPUs

If you move to a different machine or GPU:

1. Run the compatibility check on the new system
2. Use the appropriate requirements file
3. Keep `package_backup_*.txt` files for reference

## File Structure

```
superdiff-ldm/
├── requirements_clean.txt          # Base requirements (stable, for most GPUs)
├── requirements_cuda.txt           # CUDA overrides (for RTX 5090+)
├── CUDA_COMPATIBILITY.md           # This file
├── scripts/
│   ├── check_cuda_compatibility.py # Automated compatibility checker
│   └── reinstall_pytorch.sh        # Safe reinstallation script
└── logs/
    └── package_backup_*.txt        # Backup files created during reinstall
```

## Best Practices

1. **Always check compatibility** after:
   - GPU changes
   - CUDA/driver updates
   - Environment recreation
   - PyTorch updates

2. **Keep backups**: The reinstall script automatically backs up your packages to `logs/`

3. **Don't modify requirements_clean.txt**: Use `requirements_cuda.txt` for CUDA-specific overrides

4. **Test after installation**:
   ```python
   import torch
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

## Troubleshooting

### Issue: "CUDA out of memory"
- This is a different issue from compatibility
- Reduce batch size or model size
- Check `nvidia-smi` for memory usage

### Issue: "No CUDA GPUs are available"
- Check `nvidia-smi` works
- Verify driver installation
- Check if another process is using the GPU

### Issue: "ImportError: cannot import name 'PILLOW_VERSION'"
- Update Pillow: `pip install --upgrade pillow`

### Issue: PyTorch nightly builds causing instability
- Switch back to stable releases in `requirements_cuda.txt`
- Comment out the nightly section, uncomment the stable CUDA 12.4 section

## Getting Help

If you encounter issues not covered here:

1. Run diagnostic: `python scripts/check_cuda_compatibility.py`
2. Check PyTorch forums: https://discuss.pytorch.org/
3. Check PyTorch installation guide: https://pytorch.org/get-started/locally/
4. For RTX 5090 specific issues, check if PyTorch 2.6+ is released, or use nightly builds

## Version History

| Date | Event | Action Taken |
|------|-------|--------------|
| 2026-02-09 | RTX 5090 compatibility issue detected | Created CUDA override system |

---

**Last Updated**: 2026-02-09
