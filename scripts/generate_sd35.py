"""
Simple SD3.5 image generation script.

Usage
-----
# Medium (default, already downloaded):
conda run -n superdiff python scripts/generate_sd35.py \
    --prompts "a photo of a cat" "a photo of a dog" \
    [--output-dir outputs/sd35] \
    [--steps 28] \
    [--guidance 4.5] \
    [--seed 42] \
    [--width 1024] \
    [--height 1024]

# Large (8B model, ~30 GB download):
conda run -n superdiff python scripts/generate_sd35.py \
    --large \
    --prompts "a photo of a cat"
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

MODEL_MEDIUM = "stabilityai/stable-diffusion-3.5-medium"
MODEL_LARGE  = "stabilityai/stable-diffusion-3.5-large"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with SD3.5")
    parser.add_argument(
        "--prompts", nargs="+", required=True,
        help="One or more text prompts",
    )
    parser.add_argument(
        "--large", action="store_true",
        help="Use SD3.5 Large (8B) instead of Medium (2B). "
             "Requires ~30 GB disk space and has not been downloaded yet — "
             "see the download instructions printed when this flag is used.",
    )
    parser.add_argument(
        "--model-id", default=None,
        help="Override model ID entirely (takes precedence over --large).",
    )
    parser.add_argument("--output-dir", default="outputs/sd35")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16"], default="bfloat16",
        help="Model dtype (float16 is faster; bfloat16 is more numerically stable)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve model ID: explicit --model-id > --large > default medium
    if args.model_id:
        model_id = args.model_id
    elif args.large:
        model_id = MODEL_LARGE
        print(
            "\n[SD3.5 Large] Before running, download the model with:\n"
            f"  huggingface-cli download {MODEL_LARGE} --local-dir-use-symlinks False\n"
            "  (~30 GB; requires ~30 GB free disk space)\n"
        )
    else:
        model_id = MODEL_MEDIUM

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id} on {device} ({args.dtype})...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(desc="Sampling")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    for i, prompt in enumerate(args.prompts):
        print(f"[{i+1}/{len(args.prompts)}] {prompt!r}")
        result = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=generator,
        )
        image = result.images[0]

        # Build a safe filename from the prompt
        slug = prompt[:60].replace(" ", "_").replace("/", "-")
        seed_tag = f"_s{args.seed}" if args.seed is not None else ""
        out_path = output_dir / f"{i:03d}_{slug}{seed_tag}.png"
        image.save(out_path)
        print(f"  Saved → {out_path}")

    print(f"\nDone. {len(args.prompts)} image(s) written to {output_dir}/")


if __name__ == "__main__":
    main()
