"""
Publication-quality comparison grids: real COCO vs generated prompt visuals.

For each category pair (e.g. "person+car"), this script builds rows:
  1. Real MS-COCO images (ground truth pair co-occurrence)
  2. Text-to-image from the compositional prompt ("a person and a car")
  3. CLIP-embedding decode of CLIP_text("a person and a car")
  4. CLIP-embedding decode of CLIP_text("a person")
  5. CLIP-embedding decode of CLIP_text("a car")
  6. (Optional) CLIP-embedding decode of Score-SDE samples

Default backend is SDXL + IP-Adapter, which is typically much stronger than
the legacy Karlo unCLIP decode path for visual clarity.

Usage:
    python scripts/make_comparison_figure.py --pair "person+car"
    python scripts/make_comparison_figure.py --pair "person+car" --mode compose
    python scripts/make_comparison_figure.py --pair "person+car" \
      --sde_samples results/clip_person_car/samples.npy
"""

import argparse
import gc
import inspect
import json
import random
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIP_DIR = PROJECT_ROOT / "clip_embeddings" / "coco_common_pairs"
DATASET_META = PROJECT_ROOT / "datasets" / "coco_common_pairs.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate comparison grids for CLIP compositional analysis"
    )
    p.add_argument(
        "--pair",
        type=str,
        nargs="+",
        required=True,
        help="Pair name(s), e.g. 'person+car'",
    )
    p.add_argument(
        "--mode",
        choices=["full", "generate", "compose"],
        default="full",
        help="full=generate+compose, generate=cache images only, compose=grid from cache",
    )
    p.add_argument(
        "--decoder_backend",
        choices=["ip_adapter", "karlo"],
        default="ip_adapter",
        help="Backend used for prompt generation and CLIP-embedding decoding",
    )
    p.add_argument(
        "--embedding_row_mode",
        choices=["retrieval", "decode"],
        default="retrieval",
        help="How to render CLIP-embedding rows: nearest-neighbor retrieval or image decode",
    )
    p.add_argument(
        "--n_images",
        type=int,
        default=4,
        help="Number of images per row",
    )
    p.add_argument(
        "--sde_samples",
        type=Path,
        default=None,
        help="Path to Score SDE samples .npy (for optional SDE row)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "comparison_figures",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--cell_size",
        type=float,
        default=2.5,
        help="Size of each image cell in inches",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument(
        "--save_pdf",
        action="store_true",
        help="Also save as vector PDF",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--no_l2_normalize_embeddings",
        action="store_true",
        help="Disable embedding L2 normalization before decoding",
    )

    # SDXL + IP-Adapter options
    p.add_argument(
        "--sdxl_model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    p.add_argument(
        "--ip_adapter_id",
        type=str,
        default="h94/IP-Adapter",
    )
    p.add_argument(
        "--ip_adapter_weight",
        type=str,
        default="ip-adapter_sdxl.bin",
    )
    p.add_argument(
        "--ip_scale",
        type=float,
        default=0.7,
        help="IP-Adapter conditioning scale for embedding decoding",
    )
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low quality, illustration, painting, cartoon",
    )
    return p.parse_args()


def backend_names(backend):
    if backend == "ip_adapter":
        return {
            "title_tag": "SDXL + IP-Adapter",
            "text_row": "SDXL text-to-image",
            "embed_row": "SDXL + IP-Adapter",
        }
    return {
        "title_tag": "Karlo",
        "text_row": "Karlo text-to-image",
        "embed_row": "Karlo unCLIP",
    }


def safe_empty_cache(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_generator(device, seed):
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except Exception:
        return torch.Generator().manual_seed(seed)


def call_with_supported_kwargs(callable_obj, **kwargs):
    sig = inspect.signature(callable_obj)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters and v is not None:
            accepted[k] = v
    return callable_obj(**accepted)


# ── CLIP retrieval index ─────────────────────────────────────────────


def load_clip_retrieval_index():
    """Load CLIP image embeddings + manifest for nearest-neighbor retrieval."""
    embed_path = CLIP_DIR / "image_embeddings.npy"
    manifest_path = CLIP_DIR / "image_manifest.jsonl"

    image_embeds = np.load(embed_path).astype(np.float32)
    norms = np.linalg.norm(image_embeds, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    image_embeds = image_embeds / norms

    manifest = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                manifest.append(json.loads(line))

    pair_to_indices = defaultdict(list)
    category_to_indices = defaultdict(list)
    for i, rec in enumerate(manifest):
        pair_to_indices[rec["pair"]].append(i)
        for cat in rec.get("categories", []):
            category_to_indices[cat].append(i)

    return {
        "embeddings": image_embeds,
        "manifest": manifest,
        "dim": image_embeds.shape[1],
        "pair_to_indices": dict(pair_to_indices),
        "category_to_indices": dict(category_to_indices),
        "image_dir": Path(get_image_dir()),
    }


def retrieve_images_for_embedding(
    retrieval_index,
    query_embedding,
    n_images,
    normalize_query=True,
    exact_pair=None,
    required_category=None,
):
    """Return top-K nearest real images in CLIP space for query embedding."""
    q = np.asarray(query_embedding, dtype=np.float32)
    if q.ndim == 2:
        q = q[0]
    q = match_embedding_dim(q[np.newaxis, :], retrieval_index["dim"])[0]

    if normalize_query:
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-8:
            q = q / q_norm

    if exact_pair is not None:
        candidate_idx = retrieval_index["pair_to_indices"].get(exact_pair, [])
    elif required_category is not None:
        candidate_idx = retrieval_index["category_to_indices"].get(required_category, [])
    else:
        candidate_idx = list(range(len(retrieval_index["manifest"])))

    if len(candidate_idx) == 0:
        candidate_idx = list(range(len(retrieval_index["manifest"])))

    cands = retrieval_index["embeddings"][candidate_idx]
    sims = cands @ q
    k = min(n_images, len(candidate_idx))
    if k == 0:
        return []

    top_local = np.argpartition(-sims, k - 1)[:k]
    top_local = top_local[np.argsort(-sims[top_local])]
    top_idx = [candidate_idx[i] for i in top_local]

    images = []
    for idx in top_idx:
        rec = retrieval_index["manifest"][idx]
        img_path = retrieval_index["image_dir"] / rec["file_name"]
        img = Image.open(img_path).convert("RGB")
        images.append(center_crop_resize(img))
    return images


# ── COCO image loading ───────────────────────────────────────────────


def get_image_dir():
    """Get the COCO image directory from the dataset metadata."""
    with open(DATASET_META) as f:
        meta = json.load(f)
    return meta["metadata"]["image_dir"]


def center_crop_resize(img, size=512):
    """Center-crop to square, then resize."""
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.LANCZOS)


def load_coco_images(pair, n_images=4, seed=42):
    """Load and randomly sample real COCO images for a specific pair."""
    image_dir = get_image_dir()
    manifest_path = CLIP_DIR / "image_manifest.jsonl"

    with open(manifest_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    matching = [r for r in records if r["pair"] == pair]
    if len(matching) == 0:
        print(f"  WARNING: No images found for pair '{pair}'")
        return []

    rng = random.Random(seed)
    sampled = rng.sample(matching, min(n_images, len(matching)))

    images = []
    for rec in sampled:
        img_path = Path(image_dir) / rec["file_name"]
        img = Image.open(img_path).convert("RGB")
        images.append(center_crop_resize(img))

    return images


# ── Embedding helpers ────────────────────────────────────────────────


def l2_normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[np.newaxis, :]
    norms = np.linalg.norm(emb, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return emb / norms


def match_embedding_dim(embedding, expected_dim):
    """Pad or truncate embedding to expected_dim if needed."""
    if expected_dim is None:
        return embedding
    if embedding.shape[-1] == expected_dim:
        return embedding
    cur_dim = embedding.shape[-1]
    if cur_dim < expected_dim:
        pad = np.zeros(
            (*embedding.shape[:-1], expected_dim - cur_dim), dtype=embedding.dtype
        )
        return np.concatenate([embedding, pad], axis=-1)
    return embedding[..., :expected_dim]


def get_ip_adapter_embed_dim(pipe):
    """Get expected embedding dimension from IP-Adapter projection."""
    proj = pipe.unet.encoder_hid_proj
    if hasattr(proj, "image_projection_layers"):
        layer = proj.image_projection_layers[0]
    else:
        layer = proj
    if hasattr(layer, "image_embeds") and hasattr(layer.image_embeds, "in_features"):
        return layer.image_embeds.in_features
    return None


def embedding_to_ip_adapter_input(embedding, device, expected_dim=None):
    """Convert CLIP embedding to IP-Adapter cfg-compatible format."""
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[np.newaxis, :]
    emb = match_embedding_dim(emb, expected_dim=expected_dim)

    tensor_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    t = torch.from_numpy(emb).to(device=device, dtype=tensor_dtype)
    pos = t.unsqueeze(1)  # (1, 1, D)
    neg = torch.zeros_like(pos)
    combined = torch.cat([neg, pos], dim=0)  # (2, 1, D)
    return [combined]


def zero_ip_adapter_input(device, expected_dim):
    """Build neutral IP-Adapter embeds so SDXL can run prompt-only generation."""
    if expected_dim is None:
        expected_dim = 1024
    tensor_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    pos = torch.zeros((1, 1, expected_dim), device=device, dtype=tensor_dtype)
    neg = torch.zeros_like(pos)
    combined = torch.cat([neg, pos], dim=0)  # (2, 1, D) for CFG
    return [combined]


# ── Pipeline loaders ─────────────────────────────────────────────────


def load_ip_adapter_bundle(args):
    from diffusers import StableDiffusionXLPipeline

    torch_dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    kwargs = {"torch_dtype": torch_dtype}
    if torch_dtype == torch.float16:
        kwargs["variant"] = "fp16"

    print(f"Loading SDXL pipeline: {args.sdxl_model_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(args.sdxl_model_id, **kwargs).to(
        args.device
    )

    print(f"Loading IP-Adapter: {args.ip_adapter_id}/{args.ip_adapter_weight}")
    pipe.load_ip_adapter(
        args.ip_adapter_id, subfolder="sdxl_models", weight_name=args.ip_adapter_weight
    )
    pipe.set_ip_adapter_scale(args.ip_scale)

    expected_dim = get_ip_adapter_embed_dim(pipe)
    if expected_dim is not None:
        print(f"  IP-Adapter expected embedding dim: {expected_dim}")

    return {
        "backend": "ip_adapter",
        "t2i_pipe": pipe,
        "decode_pipe": pipe,
        "expected_dim": expected_dim,
    }


def load_karlo_bundle(args):
    from diffusers import UnCLIPPipeline
    from diffusers.pipelines.unclip.pipeline_unclip_image_variation import (
        UnCLIPImageVariationPipeline,
    )

    print("Loading Karlo unCLIP pipeline...")
    torch_dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    t2i_pipe = UnCLIPPipeline.from_pretrained(
        "kakaobrain/karlo-v1-alpha",
        torch_dtype=torch_dtype,
    ).to(args.device)

    decode_pipe = UnCLIPImageVariationPipeline(
        decoder=t2i_pipe.decoder,
        text_encoder=t2i_pipe.text_encoder,
        tokenizer=t2i_pipe.tokenizer,
        text_proj=t2i_pipe.text_proj,
        feature_extractor=None,
        image_encoder=None,
        super_res_first=t2i_pipe.super_res_first,
        super_res_last=t2i_pipe.super_res_last,
        decoder_scheduler=t2i_pipe.decoder_scheduler,
        super_res_scheduler=t2i_pipe.super_res_scheduler,
    ).to(args.device)

    # diffusers path reads self.image_encoder.parameters() even when
    # image_embeddings are provided. Karlo t2i pipeline has no image_encoder.
    if getattr(decode_pipe, "image_encoder", None) is None:

        def _encode_image_from_embeddings_only(
            self, image, device, num_images_per_prompt, image_embeddings=None
        ):
            if image_embeddings is None:
                raise ValueError(
                    "Karlo embedding decode requires image_embeddings input."
                )
            return image_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        decode_pipe._encode_image = types.MethodType(
            _encode_image_from_embeddings_only, decode_pipe
        )

    return {
        "backend": "karlo",
        "t2i_pipe": t2i_pipe,
        "decode_pipe": decode_pipe,
        "expected_dim": 768,
    }


def load_generation_bundle(args):
    if args.decoder_backend == "ip_adapter":
        return load_ip_adapter_bundle(args)
    return load_karlo_bundle(args)


# ── Generation functions ─────────────────────────────────────────────


def generate_text_to_image(bundle, prompt, args, n_images=4, seed=42):
    images = []
    pipe = bundle["t2i_pipe"]
    zero_ip = None
    if bundle["backend"] == "ip_adapter":
        zero_ip = zero_ip_adapter_input(args.device, bundle.get("expected_dim"))

    for i in range(n_images):
        generator = build_generator(args.device, seed + i)
        if bundle["backend"] == "ip_adapter":
            result = call_with_supported_kwargs(
                pipe,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                ip_adapter_image_embeds=zero_ip,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                num_images_per_prompt=1,
                generator=generator,
            )
        else:
            result = call_with_supported_kwargs(
                pipe,
                prompt=prompt,
                num_images_per_prompt=1,
                generator=generator,
                prior_guidance_scale=args.guidance_scale,
                decoder_guidance_scale=args.guidance_scale,
                prior_num_inference_steps=args.steps,
                decoder_num_inference_steps=args.steps,
                super_res_num_inference_steps=args.steps,
            )

        img = result.images[0]
        images.append(img)
        safe_empty_cache(args.device)
    return images


def generate_from_clip_embedding(bundle, embedding, args, n_images=1, seed=42):
    images = []
    decode_pipe = bundle["decode_pipe"]
    expected_dim = bundle.get("expected_dim")
    np_emb = np.asarray(embedding, dtype=np.float32)

    if not args.no_l2_normalize_embeddings:
        np_emb = l2_normalize_embedding(np_emb)
    elif np_emb.ndim == 1:
        np_emb = np_emb[np.newaxis, :]

    np_emb = match_embedding_dim(np_emb, expected_dim=expected_dim)

    if bundle["backend"] == "ip_adapter":
        ip_embeds = embedding_to_ip_adapter_input(
            np_emb, args.device, expected_dim=expected_dim
        )

        for i in range(n_images):
            generator = build_generator(args.device, seed + i)
            result = call_with_supported_kwargs(
                decode_pipe,
                prompt="",
                negative_prompt=args.negative_prompt,
                ip_adapter_image_embeds=ip_embeds,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                num_images_per_prompt=1,
                generator=generator,
            )
            images.append(result.images[0])
            safe_empty_cache(args.device)
        return images

    # Karlo embedding decode path
    decoder_dtype = next(decode_pipe.decoder.parameters()).dtype
    emb = torch.from_numpy(np_emb).to(args.device, dtype=decoder_dtype)

    for i in range(n_images):
        generator = build_generator(args.device, seed + i)
        result = call_with_supported_kwargs(
            decode_pipe,
            image_embeddings=emb,
            generator=generator,
            num_images_per_prompt=1,
            decoder_num_inference_steps=args.steps,
            super_res_num_inference_steps=args.steps,
        )
        images.append(result.images[0])
        safe_empty_cache(args.device)
    return images


# ── Image caching ────────────────────────────────────────────────────


def cache_row(images, cache_dir, row_name):
    """Save a row's images to disk."""
    row_dir = cache_dir / row_name
    row_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        if isinstance(img, Image.Image):
            img.convert("RGB").save(row_dir / f"{i:02d}.png")
        else:
            Image.fromarray(np.array(img)).convert("RGB").save(row_dir / f"{i:02d}.png")


def load_cached_row(cache_dir, row_name):
    """Load a row's images from disk."""
    row_dir = cache_dir / row_name
    if not row_dir.exists():
        return None
    paths = sorted(row_dir.glob("*.png"))
    if not paths:
        return None
    return [Image.open(p).convert("RGB") for p in paths]


# ── Publication grid ─────────────────────────────────────────────────


def make_publication_grid(
    rows, output_path, title="", n_cols=4, cell_size=2.5, dpi=300, save_pdf=False
):
    """Create a publication-quality comparison grid with left-side row labels."""
    n_rows = len(rows)
    label_width = 2.0

    fig = plt.figure(figsize=(label_width + n_cols * cell_size, n_rows * cell_size))
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        width_ratios=[label_width / cell_size] + [1] * n_cols,
        wspace=0.03,
        hspace=0.06,
        left=0.01,
        right=0.99,
        top=0.94,
        bottom=0.01,
    )

    for row_idx, row in enumerate(rows):
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.axis("off")
        ax_label.text(
            0.95,
            0.5,
            row["label"],
            transform=ax_label.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            fontfamily="serif",
            linespacing=1.4,
        )

        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            if col_idx < len(row["images"]):
                ax.imshow(row["images"][col_idx])
            ax.axis("off")

        if row_idx == 0:
            line_y = 1.0 - (1.0 / n_rows) - 0.005
            fig.add_artist(
                plt.Line2D(
                    [0.02, 0.98],
                    [line_y, line_y],
                    transform=fig.transFigure,
                    color="gray",
                    linewidth=0.5,
                    linestyle="--",
                )
            )

    if title:
        fig.suptitle(title, fontsize=11, fontfamily="serif", fontweight="bold", y=0.98)

    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    if save_pdf:
        pdf_path = output_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"  Saved PDF: {pdf_path}")
    plt.close()
    print(f"  Saved grid: {output_path}")


# ── Row generation orchestrator ──────────────────────────────────────


def generate_rows_for_pair(bundle, pair, prompts, idx, args, cache_dir, retrieval_index):
    p = prompts[pair]
    names = backend_names(bundle["backend"])
    cat_a, cat_b = pair.split("+", 1)

    print(f"  Loading real COCO images for '{pair}'...")
    coco_images = load_coco_images(pair, n_images=args.n_images, seed=args.seed)
    cache_row(coco_images, cache_dir, "coco_ground_truth")

    prompt_text = p["compositional"]
    print(f'  Generating {names["text_row"]}: "{prompt_text}"...')
    text_images = generate_text_to_image(
        bundle, prompt_text, args, n_images=args.n_images, seed=args.seed
    )
    cache_row(text_images, cache_dir, "text_to_image")

    text_dir = CLIP_DIR / "text_embeddings"
    comp_embed = np.load(text_dir / "compositional.npy")[idx]
    if args.embedding_row_mode == "retrieval":
        print(f'  Retrieving nearest CLIP images for CLIP("{prompt_text}")...')
        comp_images = retrieve_images_for_embedding(
            retrieval_index,
            comp_embed,
            args.n_images,
            normalize_query=not args.no_l2_normalize_embeddings,
            exact_pair=pair,
        )
    else:
        print(f'  Generating {names["embed_row"]} from CLIP("{prompt_text}")...')
        comp_images = generate_from_clip_embedding(
            bundle, comp_embed, args, n_images=args.n_images, seed=args.seed + 10
        )
    cache_row(comp_images, cache_dir, "embed_compositional")

    ind_a_embed = np.load(text_dir / "individual_a.npy")[idx]
    if args.embedding_row_mode == "retrieval":
        print(f'  Retrieving nearest CLIP images for CLIP("{p["individual_a"]}")...')
        a_images = retrieve_images_for_embedding(
            retrieval_index,
            ind_a_embed,
            args.n_images,
            normalize_query=not args.no_l2_normalize_embeddings,
            required_category=cat_a,
        )
    else:
        print(f'  Generating {names["embed_row"]} from CLIP("{p["individual_a"]}")...')
        a_images = generate_from_clip_embedding(
            bundle, ind_a_embed, args, n_images=args.n_images, seed=args.seed + 20
        )
    cache_row(a_images, cache_dir, "embed_individual_a")

    ind_b_embed = np.load(text_dir / "individual_b.npy")[idx]
    if args.embedding_row_mode == "retrieval":
        print(f'  Retrieving nearest CLIP images for CLIP("{p["individual_b"]}")...')
        b_images = retrieve_images_for_embedding(
            retrieval_index,
            ind_b_embed,
            args.n_images,
            normalize_query=not args.no_l2_normalize_embeddings,
            required_category=cat_b,
        )
    else:
        print(f'  Generating {names["embed_row"]} from CLIP("{p["individual_b"]}")...')
        b_images = generate_from_clip_embedding(
            bundle, ind_b_embed, args, n_images=args.n_images, seed=args.seed + 30
        )
    cache_row(b_images, cache_dir, "embed_individual_b")

    if args.sde_samples is not None and args.sde_samples.exists():
        sde_embeds = np.load(args.sde_samples)
        n = min(args.n_images, len(sde_embeds))
        sde_images = []
        if args.embedding_row_mode == "retrieval":
            print(f"  Retrieving nearest CLIP images for {n} Score SDE samples...")
            for i in range(n):
                imgs = retrieve_images_for_embedding(
                    retrieval_index,
                    sde_embeds[i],
                    1,
                    normalize_query=not args.no_l2_normalize_embeddings,
                    exact_pair=pair,
                )
                sde_images.extend(imgs)
        else:
            print(f"  Generating {names['embed_row']} from {n} Score SDE samples...")
            for i in range(n):
                imgs = generate_from_clip_embedding(
                    bundle, sde_embeds[i], args, n_images=1, seed=args.seed + 100 + i
                )
                sde_images.extend(imgs)
        cache_row(sde_images, cache_dir, "embed_sde_samples")


def compose_grid_for_pair(pair, prompts, cache_dir, output_path, args):
    p = prompts[pair]
    names = backend_names(args.decoder_backend)
    embed_label_prefix = (
        "CLIP retrieval" if args.embedding_row_mode == "retrieval" else names["embed_row"]
    )
    rows = []

    coco = load_cached_row(cache_dir, "coco_ground_truth")
    if coco:
        rows.append({"label": "Ground truth\n(MS-COCO)", "images": coco})

    t2i = load_cached_row(cache_dir, "text_to_image")
    if t2i:
        rows.append(
            {
                "label": f'{names["text_row"]}\n"{p["compositional"]}"',
                "images": t2i,
            }
        )

    comp = load_cached_row(cache_dir, "embed_compositional")
    if comp:
        rows.append(
            {
                "label": f'{embed_label_prefix}\nCLIP("{p["compositional"]}")',
                "images": comp,
            }
        )

    ind_a = load_cached_row(cache_dir, "embed_individual_a")
    if ind_a:
        rows.append(
            {
                "label": f'{embed_label_prefix}\nCLIP("{p["individual_a"]}")',
                "images": ind_a,
            }
        )

    ind_b = load_cached_row(cache_dir, "embed_individual_b")
    if ind_b:
        rows.append(
            {
                "label": f'{embed_label_prefix}\nCLIP("{p["individual_b"]}")',
                "images": ind_b,
            }
        )

    sde = load_cached_row(cache_dir, "embed_sde_samples")
    if sde:
        rows.append(
            {
                "label": f"{embed_label_prefix}\nScore SDE\np(x|{pair})",
                "images": sde,
            }
        )

    if not rows:
        print(f"  No cached images found in {cache_dir}")
        return

    title = f'Compositional Analysis: {pair.replace("+", " + ")}'
    if args.embedding_row_mode == "decode":
        title += f' ({backend_names(args.decoder_backend)["title_tag"]})'
    else:
        title += " (CLIP retrieval for embedding rows)"
    make_publication_grid(
        rows,
        output_path,
        title=title,
        n_cols=args.n_images,
        cell_size=args.cell_size,
        dpi=args.dpi,
        save_pdf=args.save_pdf,
    )


# ── Main ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text_dir = CLIP_DIR / "text_embeddings"
    with open(text_dir / "prompts.json") as f:
        prompts = json.load(f)
    pair_keys = list(prompts.keys())

    for pair in args.pair:
        if pair not in pair_keys:
            print(f"ERROR: Pair '{pair}' not found. Available: {pair_keys}")
            return

    bundle = None
    if args.mode in ("full", "generate"):
        bundle = load_generation_bundle(args)
    retrieval_index = None
    if args.mode in ("full", "generate") and args.embedding_row_mode == "retrieval":
        print("Loading CLIP retrieval index...")
        retrieval_index = load_clip_retrieval_index()

    for pair in args.pair:
        idx = pair_keys.index(pair)
        pair_dir = args.output_dir / pair.replace("+", "_")
        cache_dir = pair_dir / "cache" / f"{args.decoder_backend}_{args.embedding_row_mode}"
        output_path = pair_dir / "comparison_grid.png"

        print(f"\n{'=' * 60}")
        print(f"  Processing pair: {pair}")
        print(f"  Backend: {args.decoder_backend}")
        print(f"{'=' * 60}")

        if args.mode in ("full", "generate"):
            generate_rows_for_pair(
                bundle, pair, prompts, idx, args, cache_dir, retrieval_index
            )

        if args.mode in ("full", "compose"):
            compose_grid_for_pair(pair, prompts, cache_dir, output_path, args)

    if bundle is not None:
        for key in ("t2i_pipe", "decode_pipe"):
            if key in bundle and bundle[key] is not None:
                del bundle[key]
        del bundle
        gc.collect()
        safe_empty_cache(args.device)

    meta = {
        "pairs": args.pair,
        "seed": args.seed,
        "n_images": args.n_images,
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
        "sde_samples": str(args.sde_samples) if args.sde_samples else None,
        "decoder_backend": args.decoder_backend,
        "embedding_row_mode": args.embedding_row_mode,
        "l2_normalize_embeddings": not args.no_l2_normalize_embeddings,
        "sdxl_model_id": args.sdxl_model_id,
        "ip_adapter_id": args.ip_adapter_id,
        "ip_adapter_weight": args.ip_adapter_weight,
        "ip_scale": args.ip_scale,
    }
    meta_path = args.output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
