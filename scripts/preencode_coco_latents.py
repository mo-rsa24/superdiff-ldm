"""
Pre-encode the COCO Common Pairs subset through the SDXL VAE.

Reads   : datasets/coco_common_pairs.json   (image list)
Writes  : preencoded_latents/coco_common_pairs/
            ├── latent_meta.json
            ├── manifest.jsonl
            └── latents/*.npy              (4×64×64, float32)

Usage
-----
    python scripts/preencode_coco_latents.py          # defaults
    python scripts/preencode_coco_latents.py --batch_size 16 --img_size 512
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ── defaults ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS_JSON = PROJECT_ROOT / "datasets" / "coco_common_pairs.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "preencoded_latents" / "coco_common_pairs"
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def parse_args():
    p = argparse.ArgumentParser(description="Encode COCO common-pairs through SDXL VAE.")
    p.add_argument("--pairs_json", type=Path, default=DEFAULT_PAIRS_JSON)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID,
                   help="HuggingFace model with a VAE subfolder.")
    p.add_argument("--img_size", type=int, default=512,
                   help="Resize+centre-crop target (512 → 64×64 latents).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    return p.parse_args()


def load_pairs_json(path: Path):
    with open(path) as f:
        data = json.load(f)
    pair_names = sorted(data["pairs"].keys())
    pair_to_label = {name: i for i, name in enumerate(pair_names)}
    return data, pair_to_label


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),            # [0, 1]
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])


def load_image_batch(records, image_dir: Path, transform):
    """Load a list of image records, return (tensor [B,3,H,W], records)."""
    tensors = []
    valid = []
    for rec in records:
        path = image_dir / rec["file_name"]
        try:
            img = Image.open(path).convert("RGB")
            tensors.append(transform(img))
            valid.append(rec)
        except Exception as e:
            print(f"  skip {rec['file_name']}: {e}")
    if not tensors:
        return None, []
    return torch.stack(tensors), valid


@torch.no_grad()
def encode_batch(vae, batch: torch.Tensor, generator: torch.Generator):
    """Encode a pixel batch → latent batch (float32 numpy, CHW)."""
    posterior = vae.encode(batch).latent_dist
    z = posterior.sample(generator=generator)
    z = z * vae.config.scaling_factor
    return z.float().cpu().numpy()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # ── load image list ──────────────────────────────────────────────
    data, pair_to_label = load_pairs_json(args.pairs_json)
    image_dir = Path(data["metadata"]["image_dir"])
    images = data["images"]
    print(f"Images to encode : {len(images)}")
    print(f"Image directory  : {image_dir}")
    print(f"Output directory : {args.output_dir}")

    # ── load VAE ─────────────────────────────────────────────────────
    print(f"\nLoading VAE from {args.model_id} …")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device).eval()
    scaling_factor = vae.config.scaling_factor
    print(f"  scaling_factor = {scaling_factor}")
    print(f"  dtype          = {dtype}")

    # ── prepare output ───────────────────────────────────────────────
    latents_dir = args.output_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    transform = make_transform(args.img_size)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ── write provenance ─────────────────────────────────────────────
    meta = {
        "description": "SDXL-VAE latents for COCO common-pairs subset",
        "model_id": args.model_id,
        "scaling_factor": scaling_factor,
        "img_size": args.img_size,
        "latent_shape": [4, args.img_size // 8, args.img_size // 8],
        "dtype_encoding": str(dtype),
        "dtype_saved": "float32",
        "num_images": len(images),
        "num_pairs": len(pair_to_label),
        "pair_labels": pair_to_label,
        "seed": args.seed,
        "source_json": str(args.pairs_json),
        "image_dir": str(image_dir),
    }
    (args.output_dir / "latent_meta.json").write_text(json.dumps(meta, indent=2))

    # ── encode ───────────────────────────────────────────────────────
    n = len(images)
    bs = args.batch_size
    batches = [images[i : i + bs] for i in range(0, n, bs)]

    t0 = time.time()
    idx = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for batch_records in tqdm(batches, desc="Encoding"):
            pixel_batch, valid_records = load_image_batch(
                batch_records, image_dir, transform
            )
            if pixel_batch is None:
                continue
            pixel_batch = pixel_batch.to(device=device, dtype=dtype)
            latents_np = encode_batch(vae, pixel_batch, generator)

            for latent, rec in zip(latents_np, valid_records):
                img_id = rec["image_id"]
                latent_name = f"{img_id:012d}.npy"
                np.save(latents_dir / latent_name, latent.astype(np.float32))
                mf.write(json.dumps({
                    "index": idx,
                    "label": pair_to_label[rec["pair"]],
                    "latent_path": f"latents/{latent_name}",
                    "source_path": str(image_dir / rec["file_name"]),
                    "image_id": img_id,
                    "pair": rec["pair"],
                    "categories": rec["categories"],
                }) + "\n")
                idx += 1

    elapsed = time.time() - t0
    print(f"\nEncoded {idx} images in {elapsed:.1f}s ({idx / elapsed:.1f} img/s)")
    print(f"Latents saved to {args.output_dir}")


if __name__ == "__main__":
    main()
