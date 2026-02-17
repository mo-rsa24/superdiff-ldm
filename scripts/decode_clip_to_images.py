"""
Decode CLIP embeddings into images via IP-Adapter + SDXL.

Takes CLIP ViT-L/14 image embeddings (768-dim) — from the score SDE,
from text prompts, or from real images — and generates pixel-space images
by injecting them into SDXL's UNet through the IP-Adapter mechanism.

This lets us visually compare:
  - Images generated from CLIP_text("a person and a car")  → monolithic AND
  - Images generated from score SDE samples of p(x|person∧car) → learned real density
  - Images generated from composed scores p(x|person)+p(x|car)  → Riemannian composition

Usage:
    # From score SDE samples (after generate_riemannian_samples.py)
    python scripts/decode_clip_to_images.py \
        --embeddings results/clip_s767/samples.npy \
        --label "score_sde_samples" --n_images 8

    # From CLIP text embeddings
    python scripts/decode_clip_to_images.py \
        --embeddings clip_embeddings/coco_common_pairs/text_embeddings/compositional.npy \
        --label "compositional_text" --n_images 8 --index 0

    # Compare all three sources for a pair
    python scripts/decode_clip_to_images.py --compare "person+car" \
        --sde_samples results/clip_s767/samples.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIP_DIR = PROJECT_ROOT / "clip_embeddings" / "coco_common_pairs"


def parse_args():
    p = argparse.ArgumentParser(description="Decode CLIP embeddings to images via IP-Adapter")
    p.add_argument("--embeddings", type=Path, default=None,
                   help="Path to .npy embeddings to decode")
    p.add_argument("--index", type=int, default=None,
                   help="If embeddings has multiple rows, decode only this index")
    p.add_argument("--n_images", type=int, default=4,
                   help="Number of images to generate per embedding")
    p.add_argument("--label", type=str, default="decoded",
                   help="Label for output filenames")
    p.add_argument("--compare", type=str, default=None,
                   help="Pair name (e.g. 'person+car') — generates comparison grid")
    p.add_argument("--sde_samples", type=Path, default=None,
                   help="Path to score SDE samples .npy (for --compare mode)")
    p.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "results" / "decoded_images")
    p.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--ip_adapter_id", type=str, default="h94/IP-Adapter")
    p.add_argument("--ip_adapter_weight", type=str, default="ip-adapter_sdxl.bin")
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--ip_scale", type=float, default=0.7)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_pipeline(model_id, ip_adapter_id, ip_adapter_weight, device):
    """Load SDXL + IP-Adapter pipeline."""
    from diffusers import StableDiffusionXLPipeline
    from transformers import CLIPVisionModelWithProjection

    print(f"Loading SDXL pipeline: {model_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    print(f"Loading IP-Adapter: {ip_adapter_id}/{ip_adapter_weight}")
    pipe.load_ip_adapter(ip_adapter_id, subfolder="sdxl_models", weight_name=ip_adapter_weight)
    pipe.set_ip_adapter_scale(0.7)

    return pipe


def get_ip_adapter_embed_dim(pipe):
    """Get the expected embedding dimension from IP-Adapter's projection layer."""
    proj = pipe.unet.encoder_hid_proj
    if hasattr(proj, "image_projection_layers"):
        layer = proj.image_projection_layers[0]
    else:
        layer = proj
    if hasattr(layer, "image_embeds") and hasattr(layer.image_embeds, "in_features"):
        return layer.image_embeds.in_features
    return None


def embedding_to_ip_adapter_input(embedding, device, expected_dim=None):
    """Convert a CLIP embedding to IP-Adapter image_embeds format.

    When do_classifier_free_guidance is True (guidance_scale > 1), the
    pipeline calls .chunk(2) on each embed tensor, expecting:
      [negative_embeds, positive_embeds] concatenated along dim 0.
    So we provide (2, num_tokens, dim): zeros for negative, real for positive.

    If ``expected_dim`` differs from the embedding dimension the vector is
    zero-padded (or truncated) to match.  This lets 768-dim ViT-L/14
    embeddings be fed to an IP-Adapter trained on 1024-dim ViT-H/14 output.
    """
    if embedding.ndim == 1:
        embedding = embedding[np.newaxis, :]  # (1, D)

    if expected_dim is not None and embedding.shape[-1] != expected_dim:
        cur = embedding.shape[-1]
        if cur < expected_dim:
            pad = np.zeros((*embedding.shape[:-1], expected_dim - cur), dtype=embedding.dtype)
            embedding = np.concatenate([embedding, pad], axis=-1)
        else:
            embedding = embedding[..., :expected_dim]

    t = torch.from_numpy(embedding.astype(np.float32)).to(device).to(torch.float16)
    # (1, D) → (1, 1, D)  with 1 token for pooled embed
    pos = t.unsqueeze(1)
    neg = torch.zeros_like(pos)
    # Concat negative + positive along batch dim → (2, 1, D)
    combined = torch.cat([neg, pos], dim=0)
    return [combined]


def generate_from_embedding(pipe, embedding, n_images, guidance_scale, steps, seed, device,
                            expected_dim=None):
    """Generate images conditioned on a CLIP embedding via IP-Adapter."""
    ip_embeds = embedding_to_ip_adapter_input(embedding, device, expected_dim=expected_dim)

    generator = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        prompt="",
        negative_prompt="blurry, low quality",
        ip_adapter_image_embeds=ip_embeds,
        num_images_per_prompt=n_images,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    ).images

    return images


def make_grid(images_dict, output_path, title=""):
    """Create a comparison grid from multiple sources."""
    n_sources = len(images_dict)
    n_cols = max(len(imgs) for imgs in images_dict.values())

    fig, axes = plt.subplots(n_sources, n_cols, figsize=(4 * n_cols, 4.5 * n_sources))
    if n_sources == 1:
        axes = [axes]

    for row, (label, images) in enumerate(images_dict.items()):
        for col in range(n_cols):
            ax = axes[row][col] if n_cols > 1 else axes[row]
            if col < len(images):
                ax.imshow(images[col])
            ax.axis("off")
            if col == 0:
                ax.set_title(label, fontsize=10, fontweight="bold", loc="left")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved grid: {output_path}")


def run_comparison(args, pipe):
    """Generate comparison grid for a specific pair."""
    pair = args.compare
    output_dir = args.output_dir / pair.replace("+", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load text embeddings
    text_dir = CLIP_DIR / "text_embeddings"
    with open(text_dir / "prompts.json") as f:
        prompts = json.load(f)

    pair_keys = list(prompts.keys())
    if pair not in pair_keys:
        print(f"Pair '{pair}' not found. Available: {pair_keys}")
        return

    idx = pair_keys.index(pair)
    comp_embed = np.load(text_dir / "compositional.npy")[idx]
    ind_a_embed = np.load(text_dir / "individual_a.npy")[idx]
    ind_b_embed = np.load(text_dir / "individual_b.npy")[idx]
    p = prompts[pair]

    # Detect expected embedding dimension from IP-Adapter weights
    expected_dim = get_ip_adapter_embed_dim(pipe)
    embed_dim = comp_embed.shape[-1]
    if expected_dim is not None and embed_dim != expected_dim:
        print(f"  NOTE: CLIP embeddings are {embed_dim}-dim but IP-Adapter expects "
              f"{expected_dim}-dim — zero-padding to match")

    images_dict = {}

    # 1. Compositional text embedding
    print(f'\n  Generating from: "{p["compositional"]}"')
    images_dict[f'CLIP text: "{p["compositional"]}"'] = generate_from_embedding(
        pipe, comp_embed, args.n_images, args.guidance_scale, args.steps, args.seed, args.device,
        expected_dim=expected_dim,
    )

    # 2. Individual concept embeddings
    print(f'\n  Generating from: "{p["individual_a"]}"')
    images_dict[f'CLIP text: "{p["individual_a"]}"'] = generate_from_embedding(
        pipe, ind_a_embed, args.n_images, args.guidance_scale, args.steps, args.seed + 1, args.device,
        expected_dim=expected_dim,
    )

    print(f'\n  Generating from: "{p["individual_b"]}"')
    images_dict[f'CLIP text: "{p["individual_b"]}"'] = generate_from_embedding(
        pipe, ind_b_embed, args.n_images, args.guidance_scale, args.steps, args.seed + 2, args.device,
        expected_dim=expected_dim,
    )

    # 3. Score SDE samples (if provided)
    if args.sde_samples is not None and args.sde_samples.exists():
        print(f"\n  Generating from score SDE samples...")
        sde_embeds = np.load(args.sde_samples)
        # Use first n_images samples
        for i in range(min(args.n_images, len(sde_embeds))):
            imgs = generate_from_embedding(
                pipe, sde_embeds[i], 1, args.guidance_scale, args.steps,
                args.seed + 100 + i, args.device, expected_dim=expected_dim,
            )
            if f"Score SDE: p(x|{pair})" not in images_dict:
                images_dict[f"Score SDE: p(x|{pair})"] = []
            images_dict[f"Score SDE: p(x|{pair})"].extend(imgs)

    # 4. Midpoint of individual embeddings (linear interpolation baseline)
    midpoint = (ind_a_embed + ind_b_embed)
    midpoint = midpoint / np.linalg.norm(midpoint)  # Re-normalise to sphere
    print(f"\n  Generating from midpoint (linear interpolation)...")
    images_dict["Linear midpoint (a+b)/||a+b||"] = generate_from_embedding(
        pipe, midpoint, args.n_images, args.guidance_scale, args.steps, args.seed + 3, args.device,
        expected_dim=expected_dim,
    )

    make_grid(
        images_dict,
        output_dir / "comparison_grid.png",
        title=f"CLIP Embedding → Image Decoding: {pair}"
    )


def run_single(args, pipe):
    """Decode a single set of embeddings."""
    output_dir = args.output_dir / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_dim = get_ip_adapter_embed_dim(pipe)

    embeddings = np.load(args.embeddings)
    if args.index is not None:
        embeddings = embeddings[args.index:args.index + 1]

    if expected_dim is not None and embeddings.shape[-1] != expected_dim:
        print(f"  NOTE: embeddings are {embeddings.shape[-1]}-dim but IP-Adapter expects "
              f"{expected_dim}-dim — zero-padding to match")

    all_images = {}
    for i, embed in enumerate(embeddings[:8]):  # Cap at 8 embeddings
        label = f"Embedding {args.index if args.index is not None else i}"
        print(f"  Generating from {label}...")
        images = generate_from_embedding(
            pipe, embed, args.n_images, args.guidance_scale, args.steps,
            args.seed + i, args.device, expected_dim=expected_dim,
        )
        all_images[label] = images

    make_grid(all_images, output_dir / f"{args.label}_grid.png", title=args.label)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args.model_id, args.ip_adapter_id, args.ip_adapter_weight, args.device)

    if args.compare:
        run_comparison(args, pipe)
    elif args.embeddings:
        run_single(args, pipe)
    else:
        print("Provide either --embeddings or --compare")


if __name__ == "__main__":
    main()
