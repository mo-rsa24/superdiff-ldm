#!/usr/bin/env python3
"""
Generate a rows x columns comparison grid for one compositional prompt.

Rows    : model IDs
Columns : repeated runs with different seeds

Example:
    python scripts/generate_compositional_model_grid.py \
        --prompt "A person and a car" \
        --num_runs 5
"""

import argparse
import gc
import importlib.util
import inspect
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Smaller-footprint profile: 6 models.
SMALL_MODEL_IDS = [
    "CompVis/stable-diffusion-v1-4",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "SG161222/Realistic_Vision_V5.1_noVAE",
    "prompthero/openjourney-v4",
    "dreamlike-art/dreamlike-diffusion-1.0",
    "stabilityai/sd-turbo",
]

MEDIUM_MODEL_IDS = [
    "stabilityai/stable-diffusion-3.5-medium"
]

# Medium-memory profile: 6 models (no SD3.x).
MEDIUM_MODEL_IDS_ = [
    "CompVis/stable-diffusion-v1-4",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "SG161222/Realistic_Vision_V5.1_noVAE",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/sd-turbo",
    "stabilityai/sdxl-turbo",
]

# Large-capacity profile: 6 models.
LARGE_MODEL_IDS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-3.5-large-turbo",
]


def parse_args() -> argparse.Namespace:
    default_device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            default_device = "cuda"
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description="Generate prompt images across multiple diffusion models in a single grid."
    )
    p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Prompt text, e.g. "A person and a car".',
    )
    p.add_argument(
        "--profile",
        type=str,
        choices=["small", "medium", "mediam", "large"],
        default="small",
        help="Preset model list to use when --models is not provided. 'mediam' is accepted as an alias for 'medium'.",
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Explicit model IDs to compare. Overrides --profile.",
    )
    p.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs (columns) per model.",
    )
    p.add_argument(
        "--base_seed",
        type=int,
        default=1234,
        help="Base seed. Run i uses seed = base_seed + i.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Global inference step override for all models. If omitted, model-family defaults are used.",
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Global CFG override for all models. If omitted, model-family defaults are used.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (passed only if supported by the pipeline).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (passed only if supported by the pipeline).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=default_device,
        help='Torch device, e.g. "cuda", "cuda:0", "cpu".',
    )
    p.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Optional Hugging Face token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "model_prompt_grid",
        help="Output directory for per-image files and final grid.",
    )
    p.add_argument(
        "--filename_prefix",
        type=str,
        default="compositional_gap",
        help="Prefix for output filenames.",
    )
    return p.parse_args()


def model_family(model_id: str) -> str:
    m = model_id.lower()
    if "stable-diffusion-3" in m:
        return "sd3"
    if "xl" in m:
        return "sdxl"
    return "sd"


def model_defaults(model_id: str) -> Tuple[int, float]:
    m = model_id.lower()
    if "3.5-large-turbo" in m:
        return 8, 0.0
    if "stable-diffusion-3" in m:
        return 28, 4.5
    if "xl" in m:
        return 30, 6.5
    return 50, 7.5


def normalize_profile_name(profile: str) -> str:
    if profile == "mediam":
        return "medium"
    return profile


def resolve_model_ids(profile: str, explicit_models: Optional[List[str]]) -> List[str]:
    if explicit_models:
        return explicit_models
    profile = normalize_profile_name(profile)
    if profile == "large":
        return LARGE_MODEL_IDS
    if profile == "medium":
        return MEDIUM_MODEL_IDS
    return SMALL_MODEL_IDS


def validate_profile_sizes():
    if len(SMALL_MODEL_IDS) != 6:
        raise ValueError(f"SMALL_MODEL_IDS must contain exactly 6 models (found {len(SMALL_MODEL_IDS)}).")
    if len(MEDIUM_MODEL_IDS) != 6:
        raise ValueError(f"MEDIUM_MODEL_IDS must contain exactly 6 models (found {len(MEDIUM_MODEL_IDS)}).")
    if len(LARGE_MODEL_IDS) != 6:
        raise ValueError(f"LARGE_MODEL_IDS must contain exactly 6 models (found {len(LARGE_MODEL_IDS)}).")


def missing_runtime_deps(model_id: str) -> List[str]:
    """
    Return a list of missing Python packages required by this model family.
    """
    missing = []
    fam = model_family(model_id)
    # SD3/3.5 tokenizers rely on sentencepiece.
    if fam == "sd3":
        if importlib.util.find_spec("sentencepiece") is None:
            missing.append("sentencepiece")
        # Helpful hint: some tokenizer stacks also need protobuf available.
        if importlib.util.find_spec("google.protobuf") is None:
            missing.append("protobuf")
    return missing


def unsupported_model_reason(model_id: str) -> Optional[str]:
    """
    Return a user-facing reason when a model ID is incompatible with this
    script's single-stage text-to-image flow.
    """
    m = model_id.lower()
    if "stable-diffusion-xl-refiner" in m or "sdxl-refiner" in m:
        return (
            "SDXL Refiner is an img2img refinement model and is not supported "
            "as a standalone text-to-image row in this script. Use "
            "'stabilityai/sdxl-turbo' or run a two-stage base+refiner pipeline."
        )
    return None


def canonical_model_id(model_id: str) -> str:
    """
    Normalize known stale/legacy model IDs to currently valid alternatives.
    """
    aliases = {
        "stabilityai/stable-diffusion-2-1": "stabilityai/stable-diffusion-2-1-base",
        "runwayml/stable-diffusion-v1-5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    }
    return aliases.get(model_id, model_id)


def call_with_supported_kwargs(callable_obj: Callable, **kwargs):
    sig = inspect.signature(callable_obj)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters and v is not None:
            accepted[k] = v
    return callable_obj(**accepted)


def add_token_if_supported(from_pretrained: Callable, kwargs: Dict, token: Optional[str]) -> Dict:
    if not token:
        return kwargs
    sig = inspect.signature(from_pretrained)
    if "token" in sig.parameters:
        kwargs["token"] = token
    elif "use_auth_token" in sig.parameters:
        kwargs["use_auth_token"] = token
    return kwargs


def load_pipeline(model_id: str, device: str, dtype: Any, token: Optional[str]):
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    # Imported lazily for environments where SD3 class is unavailable.
    try:
        from diffusers import StableDiffusion3Pipeline
    except Exception:
        StableDiffusion3Pipeline = None  # type: ignore

    family = model_family(model_id)
    if family == "sd3":
        if StableDiffusion3Pipeline is None:
            raise RuntimeError(
                "StableDiffusion3Pipeline is unavailable in this diffusers version."
            )
        pipeline_cls = StableDiffusion3Pipeline
    elif family == "sdxl":
        pipeline_cls = StableDiffusionXLPipeline
    else:
        pipeline_cls = StableDiffusionPipeline

    base_kwargs = {"torch_dtype": dtype}
    base_kwargs = add_token_if_supported(pipeline_cls.from_pretrained, base_kwargs, token)

    # Try fp16+safetensors first when on CUDA, then fall back progressively.
    attempts = []
    if dtype == torch.float16:
        attempts.append(dict(base_kwargs, use_safetensors=True, variant="fp16"))
    attempts.append(dict(base_kwargs, use_safetensors=True))
    attempts.append(dict(base_kwargs))

    last_err = None
    for kwargs in attempts:
        try:
            pipe = pipeline_cls.from_pretrained(model_id, **kwargs)
            return pipe.to(device)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load model '{model_id}': {last_err}")


def make_generator(device: str, seed: int):
    import torch

    try:
        return torch.Generator(device=device).manual_seed(seed)
    except Exception:
        return torch.Generator().manual_seed(seed)


def make_error_image(width: int, height: int, text: str):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), color=(40, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((16, 16), text, fill=(255, 220, 220))
    return img


def sanitize_name(name: str) -> str:
    return name.replace("/", "__")


def save_grid(
    images_by_model: Dict[str, List[Any]],
    seeds: List[int],
    output_path: Path,
    prompt: str,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_ids = list(images_by_model.keys())
    rows = len(model_ids)
    cols = len(seeds)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)

    for r, model_id in enumerate(model_ids):
        row_images = images_by_model[model_id]
        for c in range(cols):
            ax = axes[r][c]
            ax.imshow(row_images[c])
            ax.axis("off")
            if r == 0:
                ax.set_title(f"Run {c + 1}\nseed={seeds[c]}", fontsize=10)
            if c == 0:
                # Keep exact model IDs visible on the left.
                ax.set_ylabel(
                    model_id,
                    fontsize=8.5,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=68,
                )

    fig.suptitle(f'Prompt: "{prompt}"', fontsize=13, y=1.0)
    max_label_len = max(len(m) for m in model_ids) if model_ids else 24
    left_margin = min(0.42, max(0.24, 0.14 + 0.0045 * max_label_len))
    fig.subplots_adjust(left=left_margin, right=0.98, top=0.92, bottom=0.04, wspace=0.03, hspace=0.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    # validate_profile_sizes()

    try:
        import torch
    except Exception as e:
        raise RuntimeError(
            "torch is not installed. Install requirements first (e.g., pip install torch)."
        ) from e

    try:
        import diffusers  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "diffusers is not installed. Install requirements first (e.g., pip install diffusers transformers accelerate)."
        ) from e

    if args.num_runs < 1:
        raise ValueError("--num_runs must be >= 1")

    resolved_profile = normalize_profile_name(args.profile)
    model_ids = resolve_model_ids(profile=resolved_profile, explicit_models=args.models)
    if not model_ids:
        raise ValueError("No models resolved. Use --profile or pass --models.")

    device = args.device
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    seeds = [args.base_seed + i for i in range(args.num_runs)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.filename_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Compositional model sweep")
    print(f'Prompt      : "{args.prompt}"')
    print(f"Profile     : {resolved_profile}")
    print(f"Models      : {len(model_ids)}")
    print(f"Runs/model  : {args.num_runs}")
    print(f"Seeds       : {seeds}")
    print(f"Device      : {device} (dtype={dtype})")
    print(f"Output dir  : {run_dir}")
    print("=" * 90)

    images_by_model: Dict[str, List[Any]] = {}

    for model_id in model_ids:
        requested_model_id = model_id
        model_id = canonical_model_id(model_id)
        if model_id != requested_model_id:
            print(f"\n[MODEL] {requested_model_id} -> {model_id} (canonicalized)")
        default_steps, default_guidance = model_defaults(model_id)
        steps = args.steps if args.steps is not None else default_steps
        guidance = args.guidance_scale if args.guidance_scale is not None else default_guidance

        print(f"\n[MODEL] {model_id}")
        print(f"  steps={steps} guidance={guidance}")
        model_out_dir = run_dir / sanitize_name(model_id)
        model_out_dir.mkdir(parents=True, exist_ok=True)

        row_images: List[Any] = []
        pipe = None
        try:
            reason = unsupported_model_reason(model_id)
            if reason:
                raise RuntimeError(reason)

            deps_missing = missing_runtime_deps(model_id)
            if deps_missing:
                raise RuntimeError(
                    "Missing runtime dependencies for this model: "
                    + ", ".join(deps_missing)
                    + ". Install with: python -m pip install "
                    + " ".join(deps_missing)
                )

            pipe = load_pipeline(
                model_id=model_id,
                device=device,
                dtype=dtype,
                token=args.hf_token,
            )

            # Keep memory footprint lower in GPU runs.
            if str(device).startswith("cuda"):
                try:
                    pipe.enable_attention_slicing()
                except Exception:
                    pass

            for col, seed in enumerate(seeds):
                print(f"  run {col + 1}/{len(seeds)} seed={seed}")
                gen = make_generator(device=device, seed=seed)
                call_kwargs = dict(
                    prompt=args.prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=args.height,
                    width=args.width,
                    generator=gen,
                )
                result = call_with_supported_kwargs(pipe.__call__, **call_kwargs)
                img = result.images[0]
                img_path = model_out_dir / f"run_{col + 1:02d}_seed_{seed}.png"
                img.save(img_path)
                row_images.append(img)

        except Exception as e:
            print(f"  [ERROR] {e}")
            err_text = f"{model_id}\nERROR\n{str(e)[:180]}"
            for _ in seeds:
                row_images.append(make_error_image(args.width, args.height, err_text))
        finally:
            if pipe is not None:
                del pipe
            gc.collect()
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        images_by_model[model_id] = row_images

    grid_path = run_dir / "model_grid.png"
    save_grid(
        images_by_model=images_by_model,
        seeds=seeds,
        output_path=grid_path,
        prompt=args.prompt,
    )
    print(f"\nSaved grid: {grid_path}")


if __name__ == "__main__":
    main()
