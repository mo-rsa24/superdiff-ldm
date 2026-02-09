#!/usr/bin/env python3
"""
CUDA Compatibility Checker for SuperDiff-LDM

This script checks if your current PyTorch installation is compatible with your GPU.
It detects compute capability mismatches and provides recommendations.
"""

import subprocess
import sys
import warnings
from typing import Optional, Tuple


def get_gpu_info() -> Optional[Tuple[str, str]]:
    """Get GPU name and compute capability using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        if lines:
            gpu_name, compute_cap = lines[0].split(', ')
            return gpu_name.strip(), compute_cap.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None
    return None


def get_cuda_version() -> Optional[str]:
    """Get CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse CUDA version from nvidia-smi output
        for line in result.stdout.split('\n'):
            if 'CUDA Version:' in line:
                cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                return cuda_version
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return None


def get_pytorch_info() -> Optional[dict]:
    """Get PyTorch version and supported compute capabilities."""
    try:
        import torch

        info = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        }

        # Try to get supported architectures
        if hasattr(torch.cuda, 'get_arch_list'):
            info['supported_archs'] = torch.cuda.get_arch_list()

        return info
    except ImportError:
        return None


def compute_cap_to_sm(compute_cap: str) -> str:
    """Convert compute capability (e.g., '12.0') to sm notation (e.g., 'sm_120')."""
    major, minor = compute_cap.split('.')
    return f"sm_{major}{minor}"


def check_compatibility() -> bool:
    """Check if PyTorch is compatible with the GPU. Returns True if compatible."""
    print("=" * 70)
    print("CUDA Compatibility Check for SuperDiff-LDM")
    print("=" * 70)

    # Get GPU info
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("❌ Could not detect GPU. Make sure nvidia-smi is available.")
        return False

    gpu_name, compute_cap = gpu_info
    sm_version = compute_cap_to_sm(compute_cap)

    print(f"\n📊 GPU Information:")
    print(f"   Name: {gpu_name}")
    print(f"   Compute Capability: {compute_cap} ({sm_version})")

    # Get CUDA version
    cuda_version = get_cuda_version()
    if cuda_version:
        print(f"\n🔧 System CUDA Version: {cuda_version}")

    # Get PyTorch info
    pytorch_info = get_pytorch_info()
    if not pytorch_info:
        print("\n❌ PyTorch is not installed.")
        return False

    print(f"\n🔥 PyTorch Information:")
    print(f"   Version: {pytorch_info['version']}")
    print(f"   CUDA Available: {pytorch_info['cuda_available']}")
    if pytorch_info['cuda_version']:
        print(f"   Compiled for CUDA: {pytorch_info['cuda_version']}")

    # Check compatibility
    is_compatible = True
    if pytorch_info['cuda_available']:
        try:
            import torch
            # Try to create a tensor on GPU
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                test_tensor = torch.randn(1, device='cuda:0')

                # Check if there were any warnings about compute capability
                for warning in w:
                    if 'not compatible' in str(warning.message).lower():
                        is_compatible = False
                        print(f"\n⚠️  Warning: {warning.message}")
                        break
        except Exception as e:
            print(f"\n❌ Error testing GPU: {e}")
            is_compatible = False

    print("\n" + "=" * 70)
    if is_compatible:
        print("✅ Your PyTorch installation is compatible with your GPU!")
    else:
        print("❌ Compatibility Issue Detected!")
        print("\n📋 Recommended Actions:")
        print("   1. Uninstall current PyTorch:")
        print("      pip uninstall torch torchvision torchaudio triton -y")
        print("\n   2. Install compatible PyTorch version:")

        # Recommend based on compute capability
        if compute_cap >= "12.0":
            print("      # For RTX 5090 (sm_120+), you need PyTorch 2.6+:")
            print("      pip install -r requirements_cuda.txt")
        else:
            print("      pip install -r requirements_clean.txt")

        print("\n   3. Verify installation:")
        print("      python scripts/check_cuda_compatibility.py")

    print("=" * 70)
    return is_compatible


if __name__ == "__main__":
    is_compatible = check_compatibility()
    sys.exit(0 if is_compatible else 1)
