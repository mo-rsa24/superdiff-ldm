"""
Extract CLIP embeddings for the curated COCO Common Pairs subset.

Produces:
  1. Image embeddings  — one per COCO image  (N×D, L2-normalised)
  2. Text embeddings   — compositional ("a person and a car")
                         + individual   ("a person", "a car")
  3. Manifest linking each embedding row to its image/pair metadata

Uses OpenAI CLIP ViT-L/14, the same encoder used in Stable Diffusion 1.x/2.x
and as one of SDXL's two text encoders.  This shared image–text embedding
space is what lets us compare text-conditioned directions to the real
image distribution on the manifold.

Usage
-----
    python scripts/extract_clip_embeddings.py
    python scripts/extract_clip_embeddings.py --model_id openai/clip-vit-large-patch14
    python scripts/extract_clip_embeddings.py --batch_size 64
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ── paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS_JSON = PROJECT_ROOT / "datasets" / "coco_common_pairs.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "clip_embeddings" / "coco_common_pairs"

# The CLIP encoder used by Stable Diffusion (ViT-L/14)
DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract CLIP image & text embeddings for COCO common pairs."
    )
    p.add_argument("--pairs_json", type=Path, default=DEFAULT_PAIRS_JSON)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── image embeddings ─────────────────────────────────────────────────

def extract_image_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_records: list,
    image_dir: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Return (N, D) float32 array of L2-normalised CLIP image embeddings."""

    all_embeds = []
    n = len(image_records)

    for start in tqdm(range(0, n, batch_size), desc="Image embeddings"):
        batch_records = image_records[start : start + batch_size]
        images = []
        for rec in batch_records:
            img_path = Path(image_dir) / rec["file_name"]
            img = Image.open(img_path).convert("RGB")
            images.append(img)

        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)  # (B, D)
            # L2 normalise (CLIP convention)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)

        all_embeds.append(outputs.cpu().float().numpy())

    return np.concatenate(all_embeds, axis=0)


# ── text embeddings ──────────────────────────────────────────────────

def build_text_prompts(pairs_info: dict) -> dict:
    """
    For each pair, create three prompts:
      - compositional : "a {cat_a} and a {cat_b}"
      - individual_a  : "a {cat_a}"
      - individual_b  : "a {cat_b}"

    Returns dict mapping pair_key → {compositional, individual_a, individual_b}.
    """
    prompts = {}
    for pair_key, info in pairs_info.items():
        cat_a, cat_b = info["category_names"]
        prompts[pair_key] = {
            "compositional": f"a {cat_a} and a {cat_b}",
            "individual_a": f"a {cat_a}",
            "individual_b": f"a {cat_b}",
        }
    return prompts


def extract_text_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    text_prompts: dict,
    device: str,
) -> dict:
    """
    Returns dict with keys:
      - pair_keys        : list of pair names (ordered)
      - compositional    : (P, D) array — "a person and a car"
      - individual_a     : (P, D) array — "a person"
      - individual_b     : (P, D) array — "a car"
      - prompts          : dict of pair_key → {compositional, individual_a, individual_b}
    """
    pair_keys = list(text_prompts.keys())

    result = {"pair_keys": pair_keys, "prompts": text_prompts}

    for embed_type in ("compositional", "individual_a", "individual_b"):
        texts = [text_prompts[pk][embed_type] for pk in pair_keys]
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)  # (P, D)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)

        result[embed_type] = outputs.cpu().float().numpy()

    return result


# ── main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load curated dataset
    with open(args.pairs_json, "r") as f:
        data = json.load(f)

    image_dir = data["metadata"]["image_dir"]
    pairs_info = data["pairs"]
    image_records = data["images"]
    print(f"Loaded {len(image_records)} images, {len(pairs_info)} pairs")
    print(f"Image directory: {image_dir}")

    # Load CLIP
    print(f"\nLoading CLIP model: {args.model_id}")
    model = CLIPModel.from_pretrained(args.model_id).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_id)

    embed_dim = model.config.projection_dim
    print(f"Embedding dimension: {embed_dim}")
    print(f"Device: {args.device}\n")

    # ── Extract image embeddings ──
    t0 = time.time()
    image_embeds = extract_image_embeddings(
        model, processor, image_records, image_dir, args.batch_size, args.device
    )
    img_time = time.time() - t0
    print(f"\nImage embeddings: {image_embeds.shape} in {img_time:.1f}s")

    # ── Extract text embeddings ──
    text_prompts = build_text_prompts(pairs_info)
    text_result = extract_text_embeddings(model, processor, text_prompts, args.device)
    print(f"Text embeddings:  {text_result['compositional'].shape} "
          f"× 3 (compositional, individual_a, individual_b)")

    # ── Save everything ──
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Image embeddings — single (N, D) matrix
    np.save(out / "image_embeddings.npy", image_embeds)

    # Image manifest — maps row index to image metadata
    manifest_path = out / "image_manifest.jsonl"
    with open(manifest_path, "w") as f:
        for i, rec in enumerate(image_records):
            entry = {
                "idx": i,
                "image_id": rec["image_id"],
                "file_name": rec["file_name"],
                "pair": rec["pair"],
                "categories": rec["categories"],
            }
            f.write(json.dumps(entry) + "\n")

    # Text embeddings
    text_dir = out / "text_embeddings"
    text_dir.mkdir(exist_ok=True)
    np.save(text_dir / "compositional.npy", text_result["compositional"])
    np.save(text_dir / "individual_a.npy", text_result["individual_a"])
    np.save(text_dir / "individual_b.npy", text_result["individual_b"])

    with open(text_dir / "prompts.json", "w") as f:
        json.dump(text_result["prompts"], f, indent=2)

    # Provenance metadata
    meta = {
        "model_id": args.model_id,
        "embed_dim": embed_dim,
        "num_images": len(image_records),
        "num_pairs": len(pairs_info),
        "pairs_json": str(args.pairs_json),
        "image_dir": image_dir,
        "device": args.device,
        "batch_size": args.batch_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Quick sanity stats ──
    print(f"\n{'=' * 60}")
    print("  QUICK SANITY CHECK")
    print(f"{'=' * 60}")

    # Norms should all be ~1.0
    norms = np.linalg.norm(image_embeds, axis=1)
    print(f"\n  Image embedding norms: min={norms.min():.6f}  max={norms.max():.6f}")

    # Per-pair: cosine similarity between compositional text and mean image embedding
    print(f"\n  {'Pair':<28s}  cos(text, img_mean)  cos(a, img)  cos(b, img)")
    print(f"  {'-'*80}")

    for pi, pair_key in enumerate(text_result["pair_keys"]):
        # Find image indices for this pair
        pair_mask = [r["pair"] == pair_key for r in image_records]
        pair_imgs = image_embeds[pair_mask]  # (K, D)
        if len(pair_imgs) == 0:
            continue
        mean_img = pair_imgs.mean(axis=0)
        mean_img = mean_img / np.linalg.norm(mean_img)

        cos_comp = float(text_result["compositional"][pi] @ mean_img)
        cos_a = float(text_result["individual_a"][pi] @ mean_img)
        cos_b = float(text_result["individual_b"][pi] @ mean_img)

        print(f"  {pair_key:<28s}  {cos_comp:+.4f}               "
              f"{cos_a:+.4f}       {cos_b:+.4f}")

    print(f"\n  Saved to: {out}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
