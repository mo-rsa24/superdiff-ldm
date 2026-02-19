"""
Phase 1: Generate SD3.5 closed-loop training data for the inverter.

For each prompt in the curated set, this script:
  1. Encodes the prompt with SD3.5's triple text encoders → conditioning tensors
  2. Generates K images using SD3.5(prompt) with K different seeds
  3. Saves: images (PNG) + conditioning tensors (.pt) to disk

The inverter f_θ is then trained on (image, conditioning) pairs so that
  f_θ(SD3.5(p)) ≈ p   (in the SD3.5 conditioning space)

Usage
-----
conda run -n superdiff python scripts/generate_inversion_training_data.py \
    [--model-id stabilityai/stable-diffusion-3.5-medium] \
    [--output-dir experiments/inversion/training_data] \
    [--images-per-prompt 8] \
    [--steps 50] \
    [--guidance 4.5] \
    [--image-size 512]
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from notebooks.utils import get_sd3_models, get_sd3_text_embedding
from notebooks.composition_experiments import sample_sd3_with_trajectory_tracking

# ---------------------------------------------------------------------------
# Prompt set
# ---------------------------------------------------------------------------

# Tier A: single concepts
TIER_A = [
    "a cat",
    "a dog",
    "a car",
    "a truck",
    "a person",
    "an umbrella",
    "a horse",
    "a bird",
    "a bicycle",
    "a tree",
    "a boat",
    "a chair",
    "a glass bottle",
    "a red apple",
    "a wooden table",
]

# Tier B: simple conjunctions
TIER_B = [
    "a person standing next to a bicycle",
    "a cat sitting on a chair",
    "a dog running in a field",
    "a bird perched on a tree branch",
    "a boat on the water",
    "a horse grazing in a meadow",
    "a glass bottle on a wooden table",
    "a red apple on a white plate",
    "a person walking with a dog",
    "a car parked next to a tree",
    "a bicycle leaning against a wall",
    "a cat and a bird in a garden",
    "a truck on a highway",
    "a person reading a book",
    "a horse and a dog in a field",
    "a boat near a wooden dock",
    "a chair next to a table",
    "a person with a bicycle",
    "a bird flying over a lake",
    "a cat sleeping on a chair",
    "a dog sitting next to a bicycle",
    "a truck parked near a tree",
    "a glass bottle next to a red apple",
    "a person standing under an umbrella",
    "a horse running in a field",
    "a bird sitting on a boat",
    "a car driving on a road",
    "a dog chasing a bicycle",
    "a cat looking out a window",
    "a person feeding a horse",
    "a tree growing next to a lake",
    "a wooden table with a glass bottle",
    "a bicycle near a red car",
    "a bird eating an apple",
    "a person walking near a lake",
    "a truck on a dirt road",
    "a chair beside a wooden table",
    "a boat sailing on a river",
    "a cat on top of a car",
    "a dog beneath a tree",
]

# Tier C: attribute-rich single concepts
TIER_C = [
    "a red car",
    "a blue truck",
    "a white horse",
    "a black cat",
    "a green bicycle",
    "a yellow umbrella",
    "a brown wooden chair",
    "a crystal glass bottle",
    "a shiny red apple",
    "a tall oak tree",
    "a small orange boat",
    "a fluffy white dog",
    "a grey stone table",
    "a colourful parrot",
    "a rusty old bicycle",
    "a silver sports car",
    "a striped umbrella",
    "a golden horse",
    "a black dog with white spots",
    "a vintage wooden boat",
    "a ceramic white cup",
    "a purple bicycle",
    "a spotted cat",
    "a dark green truck",
    "a tiny red bird",
    "a large grey elephant",
    "a yellow school bus",
    "a pink umbrella",
    "a maroon leather chair",
    "a clear glass vase",
]

# Tier D: held-out test pairs (DO NOT include in training)
# These are the exact pairs we will use for SuperDiff-AND at test time.
TEST_PAIRS_HELD_OUT = [
    ("a cat", "a dog"),
    ("a person", "an umbrella"),
    ("a person", "a car"),
    ("a car", "a truck"),
]
_HELD_OUT_SLUGS = {
    f"{c1.replace(' ', '_')}_{c2.replace(' ', '_')}"
    for c1, c2 in TEST_PAIRS_HELD_OUT
}

ALL_PROMPTS = TIER_A + TIER_B + TIER_C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(prompt: str) -> str:
    """Convert a prompt to a safe directory name."""
    s = prompt.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s[:80]


def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents (B, C, H, W) → images (B, 3, H, W) in [0, 1]."""
    latents = latents.to(dtype=vae.dtype)
    images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    return images.float()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate SD3.5 closed-loop training data")
    p.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--output-dir", default="experiments/inversion/training_data")
    p.add_argument("--images-per-prompt", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--image-size", type=int, default=512,
                   help="Output image size in pixels (square). Latent = size // 8.")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip prompts whose output directory already exists")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.dtype == "float16" else torch.bfloat16

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    latent_size = args.image_size // 8  # e.g. 512 → 64

    print(f"Loading SD3.5 from {args.model_id} ...")
    models = get_sd3_models(model_id=args.model_id, dtype=dtype, device=device)

    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    dataset_index = []

    for prompt in ALL_PROMPTS:
        slug = slugify(prompt)
        prompt_dir = out_root / slug

        if args.skip_existing and (prompt_dir / "conditioning.pt").exists():
            print(f"  [skip] {prompt}")
            dataset_index.append({"prompt": prompt, "slug": slug, "path": str(prompt_dir)})
            continue

        prompt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing: {prompt}")

        # Encode prompt once — shared across all seeds
        with torch.no_grad():
            seq_embeds, pooled_embeds = get_sd3_text_embedding(
                [prompt],
                models["tokenizer"],   models["text_encoder"],
                models["tokenizer_2"], models["text_encoder_2"],
                models["tokenizer_3"], models["text_encoder_3"],
                device=device,
            )

        # Save conditioning (float32 for precision)
        torch.save(
            {
                "seq_embeds":    seq_embeds.float().cpu(),    # (1, 410, 4096)
                "pooled_embeds": pooled_embeds.float().cpu(), # (1, 2048)
                "prompt":        prompt,
            },
            prompt_dir / "conditioning.pt",
        )

        # Generate K images with different seeds
        for seed in range(args.images_per_prompt):
            generator = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn(
                1, 16, latent_size, latent_size,
                device=device, dtype=dtype, generator=generator,
            )

            with torch.no_grad():
                final_latents, _ = sample_sd3_with_trajectory_tracking(
                    latents=latents,
                    prompt=prompt,
                    scheduler=scheduler,
                    transformer=models["transformer"],
                    tokenizer=models["tokenizer"],
                    text_encoder=models["text_encoder"],
                    tokenizer_2=models["tokenizer_2"],
                    text_encoder_2=models["text_encoder_2"],
                    tokenizer_3=models["tokenizer_3"],
                    text_encoder_3=models["text_encoder_3"],
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                    batch_size=1,
                    device=device,
                    dtype=dtype,
                )
                image = decode_latents(models["vae"], final_latents)  # (1, 3, H, W)

            # Resize to target image_size if needed
            if image.shape[-1] != args.image_size:
                import torch.nn.functional as F
                image = F.interpolate(image, size=args.image_size, mode="bilinear",
                                      align_corners=False)

            save_image(image, prompt_dir / f"img_{seed:04d}.png")

        dataset_index.append({"prompt": prompt, "slug": slug, "path": str(prompt_dir)})
        print(f"  → saved {args.images_per_prompt} images + conditioning.pt")

    # Write index
    index_path = out_root / "dataset_index.json"
    with open(index_path, "w") as f:
        json.dump(dataset_index, f, indent=2)

    print(f"\nDone. {len(dataset_index)} prompts processed.")
    print(f"Dataset index: {index_path}")
    print(f"Total images:  {len(dataset_index) * args.images_per_prompt}")


if __name__ == "__main__":
    main()
