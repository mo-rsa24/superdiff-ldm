"""
Verify pre-encoded COCO latents and display example statistics.

Usage
-----
    python scripts/verify_latents.py
    python scripts/verify_latents.py --latent_dir preencoded_latents/coco_common_pairs
    python scripts/verify_latents.py --show_examples 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LATENT_DIR = PROJECT_ROOT / "preencoded_latents" / "coco_common_pairs"
EXPECTED_SHAPE = (4, 64, 64)


def parse_args():
    p = argparse.ArgumentParser(description="Verify pre-encoded latent files.")
    p.add_argument("--latent_dir", type=Path, default=DEFAULT_LATENT_DIR)
    p.add_argument("--show_examples", type=int, default=5,
                   help="Number of individual latent examples to print.")
    return p.parse_args()


def verify_all_npy(latents_dir: Path):
    """Check every .npy file for shape, dtype, NaN, Inf."""
    npy_files = sorted(latents_dir.glob("*.npy"))
    n = len(npy_files)

    bad_shape, bad_dtype, nan_count, inf_count = 0, 0, 0, 0
    mins, maxs, means, stds = [], [], [], []

    for f in npy_files:
        z = np.load(f)
        if z.shape != EXPECTED_SHAPE:
            bad_shape += 1
        if z.dtype != np.float32:
            bad_dtype += 1
        if np.any(np.isnan(z)):
            nan_count += 1
            continue
        if np.any(np.isinf(z)):
            inf_count += 1
            continue
        mins.append(z.min())
        maxs.append(z.max())
        means.append(z.mean())
        stds.append(z.std())

    return {
        "total": n,
        "bad_shape": bad_shape,
        "bad_dtype": bad_dtype,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "mins": np.array(mins),
        "maxs": np.array(maxs),
        "means": np.array(means),
        "stds": np.array(stds),
    }


def print_report(stats):
    n = stats["total"]
    valid = n - stats["nan_count"] - stats["inf_count"]

    print("=" * 60)
    print("  LATENT VERIFICATION REPORT")
    print("=" * 60)

    print(f"\n  Total .npy files    : {n}")
    print(f"  Expected shape      : {EXPECTED_SHAPE}")
    print(f"  Bad shape           : {stats['bad_shape']}")
    print(f"  Bad dtype (!=fp32)  : {stats['bad_dtype']}")
    print(f"  Files with NaN      : {stats['nan_count']}")
    print(f"  Files with Inf      : {stats['inf_count']}")
    print(f"  Valid files         : {valid}")

    if valid == 0:
        print("\n  No valid files to compute statistics on.")
        return False

    print(f"\n  --- Value statistics (over {valid} valid files) ---")
    print(f"  Global min          : {stats['mins'].min():.4f}")
    print(f"  Global max          : {stats['maxs'].max():.4f}")
    print(f"  Mean of per-file means : {stats['means'].mean():.4f}")
    print(f"  Std of per-file means  : {stats['means'].std():.4f}")
    print(f"  Mean of per-file stds  : {stats['stds'].mean():.4f}")
    print(f"  Min per-file std       : {stats['stds'].min():.4f}")
    print(f"  Max per-file std       : {stats['stds'].max():.4f}")

    ok = (stats["bad_shape"] == 0 and stats["bad_dtype"] == 0
          and stats["nan_count"] == 0 and stats["inf_count"] == 0)
    status = "PASS" if ok else "FAIL"
    print(f"\n  Overall: {status}")
    print("=" * 60)
    return ok


def show_examples(latents_dir: Path, manifest_path: Path, num: int):
    """Print per-channel stats for a few example latents."""
    records = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    indices = np.linspace(0, len(records) - 1, num, dtype=int)

    print(f"\n{'=' * 60}")
    print(f"  EXAMPLE LATENTS ({num} samples)")
    print(f"{'=' * 60}")

    for i in indices:
        rec = records[i]
        z = np.load(latents_dir.parent / rec["latent_path"])
        print(f"\n  [{i}] {rec['pair']}  (image_id={rec['image_id']})")
        print(f"       shape={z.shape}  dtype={z.dtype}")
        for ch in range(z.shape[0]):
            c = z[ch]
            print(f"       ch{ch}: min={c.min():+7.3f}  max={c.max():+7.3f}"
                  f"  mean={c.mean():+7.3f}  std={c.std():.3f}")


def test_dataset_class(latent_dir: Path):
    """Try loading via PreencodedLatentDataset."""
    print(f"\n{'=' * 60}")
    print("  PreencodedLatentDataset COMPATIBILITY")
    print(f"{'=' * 60}")

    try:
        from datasets.Latents import PreencodedLatentDataset
    except ImportError as e:
        print(f"\n  Could not import PreencodedLatentDataset: {e}")
        return False

    try:
        ds = PreencodedLatentDataset(str(latent_dir))
        print(f"\n  Dataset loaded: {len(ds)} samples")

        for idx in [0, len(ds) // 2, len(ds) - 1]:
            z, label = ds[idx]
            print(f"  ds[{idx:>5d}] -> z.shape={tuple(z.shape)}, "
                  f"z.dtype={z.dtype}, label={label}")

        # Verify labels span expected range
        all_labels = set()
        for rec in ds.records:
            all_labels.add(int(rec.get("label", 0)))
        print(f"\n  Unique labels: {len(all_labels)} ({min(all_labels)}..{max(all_labels)})")
        print(f"\n  Result: PASS")
        return True

    except Exception as e:
        print(f"\n  Error: {e}")
        print(f"\n  Result: FAIL")
        return False


def main():
    args = parse_args()
    latent_dir = args.latent_dir
    latents_subdir = latent_dir / "latents"
    manifest_path = latent_dir / "manifest.jsonl"

    # Check paths exist
    if not latent_dir.exists():
        print(f"Latent directory not found: {latent_dir}")
        sys.exit(1)
    if not latents_subdir.exists():
        print(f"Latents subdirectory not found: {latents_subdir}")
        sys.exit(1)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    # 1. Verify all .npy files
    print("Scanning all .npy files (this may take a moment)...\n")
    stats = verify_all_npy(latents_subdir)
    npy_ok = print_report(stats)

    # 2. Show example latents
    show_examples(latents_subdir, manifest_path, args.show_examples)

    # 3. Test PreencodedLatentDataset
    ds_ok = test_dataset_class(latent_dir)

    # Final summary
    print(f"\n{'=' * 60}")
    if npy_ok and ds_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — review output above")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
