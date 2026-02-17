#!/usr/bin/env python3
"""
Manifold Visualization: Co-occurrence vs Hybrid vs Individual Concepts

Embeds all of the following into a shared CLIP ViT-L/14 space and visualizes
their geometric relationships on S^{767}:

  1. Individual text anchors:  "a car", "a truck"
  2. Compositional text:       "a car and a truck"
  3. COCO ground-truth images  containing both objects (real co-occurrence)
  4. SuperDIFF FM-ODE output   decoded → CLIP image embedding  (hybrid/chimera)
  5. Standard SD3 "AND" output decoded → CLIP image embedding  (monolithic AND)
  6. Geodesic midpoint:        normalize(e_a + e_b)

Two plots are produced:
  A. Manifold projection (PCA / UMAP) — all points in a shared 2D space
  B. Geodesic decomposition (t, r) — tangential vs residual components
     relative to the great-circle arc between the two concept anchors.
     Co-occurrence → low r; hybrids → high r.

Usage:
    # Default: car+truck (best non-person pair, 224 COCO images)
    python scripts/manifold_cooccurrence_vs_hybrid.py

    # Specify pair and SuperDIFF experiment dir
    python scripts/manifold_cooccurrence_vs_hybrid.py \
        --pair car+truck \
        --experiment-dir experiments/trajectory_dynamics/YYYYMMDD_HHMMSS

    # With UMAP instead of PCA
    python scripts/manifold_cooccurrence_vs_hybrid.py --projection umap

    # Generate fresh SD3 outputs (requires GPU + SD3.5 model)
    python scripts/manifold_cooccurrence_vs_hybrid.py --generate
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CLIP_DIR = PROJECT_ROOT / "clip_embeddings" / "coco_common_pairs"
PROMPTS_JSON = CLIP_DIR / "text_embeddings" / "prompts.json"
MANIFEST = CLIP_DIR / "image_manifest.jsonl"
IMAGE_EMBEDS = CLIP_DIR / "image_embeddings.npy"
TEXT_EMBED_DIR = CLIP_DIR / "text_embeddings"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pair_list() -> List[str]:
    """Return list of available pair names from prompts.json."""
    with open(PROMPTS_JSON) as f:
        return list(json.load(f).keys())


def load_text_embeddings(pair: str) -> Dict[str, np.ndarray]:
    """Load the three text embeddings for a given pair.

    Returns dict with keys: 'individual_a', 'individual_b', 'compositional',
    each of shape (768,).
    """
    with open(PROMPTS_JSON) as f:
        prompts = json.load(f)
    pair_names = list(prompts.keys())
    idx = pair_names.index(pair)

    individual_a = np.load(TEXT_EMBED_DIR / "individual_a.npy")[idx]  # (768,)
    individual_b = np.load(TEXT_EMBED_DIR / "individual_b.npy")[idx]  # (768,)
    compositional = np.load(TEXT_EMBED_DIR / "compositional.npy")[idx]  # (768,)

    return {
        "individual_a": individual_a,
        "individual_b": individual_b,
        "compositional": compositional,
        "prompt_a": prompts[pair]["individual_a"],
        "prompt_b": prompts[pair]["individual_b"],
        "prompt_comp": prompts[pair]["compositional"],
    }


def load_coco_image_embeddings(pair: str) -> np.ndarray:
    """Load COCO image embeddings for a specific pair. Returns (N, 768)."""
    all_embeds = np.load(IMAGE_EMBEDS)  # (5722, 768)

    with open(MANIFEST) as f:
        records = [json.loads(line) for line in f if line.strip()]

    mask = np.array([r["pair"] == pair for r in records])
    return all_embeds[mask]


def compute_geodesic_midpoint(e_a: np.ndarray, e_b: np.ndarray) -> np.ndarray:
    """Geodesic midpoint on S^{d-1}: normalize(e_a + e_b)."""
    mid = e_a + e_b
    return mid / np.linalg.norm(mid)


# ---------------------------------------------------------------------------
# Geodesic decomposition
# ---------------------------------------------------------------------------

def geodesic_decomposition(
    points: np.ndarray,
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose points relative to the great-circle arc between two anchors.

    For each point p on S^{d-1}:
      - t (tangential): signed projection along the arc direction
        t > 0 means closer to anchor_b, t < 0 closer to anchor_a
      - r (residual): perpendicular distance from the arc
        r ~ 0 for co-occurrence, r >> 0 for hybrids

    All inputs should be L2-normalized (on the unit sphere).

    Returns:
        t_vals: (N,) tangential coordinates
        r_vals: (N,) residual magnitudes (always >= 0)
    """
    # Arc direction: component of b orthogonal to a, normalized
    # This gives us a local 2D coordinate frame on the tangent plane at midpoint
    mid = anchor_a + anchor_b
    mid = mid / np.linalg.norm(mid)

    # Tangent vector along the arc at the midpoint
    arc_dir = anchor_b - anchor_a
    # Project out the radial component (component along mid)
    arc_dir = arc_dir - np.dot(arc_dir, mid) * mid
    arc_norm = np.linalg.norm(arc_dir)
    if arc_norm < 1e-10:
        # Degenerate: anchors are identical
        return np.zeros(len(points)), np.zeros(len(points))
    arc_dir = arc_dir / arc_norm

    # For each point, project onto tangent plane at mid, then decompose
    t_vals = []
    r_vals = []
    for p in points:
        # Project p onto tangent plane at mid (remove radial component)
        p_tang = p - np.dot(p, mid) * mid
        # t = projection along arc_dir
        t = np.dot(p_tang, arc_dir)
        # r = magnitude of component perpendicular to arc_dir (in tangent plane)
        perp = p_tang - t * arc_dir
        r = np.linalg.norm(perp)
        t_vals.append(t)
        r_vals.append(r)

    return np.array(t_vals), np.array(r_vals)


# ---------------------------------------------------------------------------
# CLIP encoding of generated images (from latents or PIL)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_images_with_clip(
    images,  # list of PIL images
    clip_model_id: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
) -> np.ndarray:
    """Encode PIL images with CLIP ViT-L/14. Returns (N, 768) L2-normalized."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_model_id)

    all_embeds = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = model.get_image_features(**inputs)
        emb = F.normalize(emb, dim=-1)
        all_embeds.append(emb.cpu().numpy())

    del model, processor
    torch.cuda.empty_cache()

    return np.concatenate(all_embeds, axis=0)  # (N, 768)


def load_experiment_images(experiment_dir: Path) -> Dict[str, "PIL.Image"]:
    """Load decoded images from a trajectory experiment output directory."""
    from PIL import Image

    images = {}
    decoded_dir = experiment_dir
    # Look for decoded PNGs — the trajectory experiment saves individual condition images
    for png in sorted(decoded_dir.glob("*.png")):
        name = png.stem
        if "decoded" in name or "trajectory" in name or "clip_probe" in name:
            continue  # skip composite plots
        images[name] = Image.open(png).convert("RGB")

    return images


# ---------------------------------------------------------------------------
# Generation (optional — runs SD3 for fresh outputs)
# ---------------------------------------------------------------------------

def generate_outputs(
    prompt_a: str,
    prompt_b: str,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.5,
) -> Dict[str, "PIL.Image"]:
    """Generate images for all conditions using SD3.5 Medium.

    Returns dict: condition_name → PIL.Image
    """
    from notebooks.utils import get_sd3_models, get_image
    from notebooks.composition_experiments import (
        _get_sd3_conditioning,
        sample_sd3_with_trajectory_tracking,
        superdiff_sd3_with_trajectory_tracking,
        LatentTrajectoryCollector,
    )
    from diffusers import FlowMatchEulerDiscreteScheduler

    device = torch.device("cuda")
    dtype = torch.float16
    monolithic_prompt = f"{prompt_a} and {prompt_b}"

    print("Loading SD3.5 Medium...")
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    models = get_sd3_models(model_id)
    vae = models["vae"]
    transformer = models["transformer"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    tokenizer_2 = models["tokenizer_2"]
    text_encoder_2 = models["text_encoder_2"]
    tokenizer_3 = models["tokenizer_3"]
    text_encoder_3 = models["text_encoder_3"]

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )

    # Shared initial noise
    gen = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(1, 16, 128, 128, device=device, dtype=dtype, generator=gen)

    results = {}

    # 1. Individual A
    print(f"  Generating: {prompt_a}")
    final_a, tracker_a = sample_sd3_with_trajectory_tracking(
        latents.clone(), prompt_a, scheduler, transformer,
        tokenizer, text_encoder, tokenizer_2, text_encoder_2,
        tokenizer_3, text_encoder_3,
        guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
    )
    results["individual_a"] = get_image(vae, final_a, 1, 1)

    # 2. Individual B
    print(f"  Generating: {prompt_b}")
    final_b, tracker_b = sample_sd3_with_trajectory_tracking(
        latents.clone(), prompt_b, scheduler, transformer,
        tokenizer, text_encoder, tokenizer_2, text_encoder_2,
        tokenizer_3, text_encoder_3,
        guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
    )
    results["individual_b"] = get_image(vae, final_b, 1, 1)

    # 3. Monolithic AND
    print(f"  Generating: {monolithic_prompt}")
    final_m, tracker_m = sample_sd3_with_trajectory_tracking(
        latents.clone(), monolithic_prompt, scheduler, transformer,
        tokenizer, text_encoder, tokenizer_2, text_encoder_2,
        tokenizer_3, text_encoder_3,
        guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
    )
    results["monolithic_and"] = get_image(vae, final_m, 1, 1)

    # 4. SuperDIFF FM-ODE
    print(f"  Generating: SuperDIFF FM-ODE ({prompt_a} ∧ {prompt_b})")
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "trajectory_dynamics_experiment",
        PROJECT_ROOT / "scripts" / "trajectory_dynamics_experiment.py",
    )
    _tde = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tde)
    superdiff_fm_ode_sd3 = _tde.superdiff_fm_ode_sd3
    final_sd, _, kappa_sd, ll_obj, ll_bg = superdiff_fm_ode_sd3(
        latents.clone(), prompt_a, prompt_b, scheduler, transformer,
        tokenizer, text_encoder, tokenizer_2, text_encoder_2,
        tokenizer_3, text_encoder_3,
        guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
    )
    results["superdiff_fm_ode"] = get_image(vae, final_sd, 1, 1)

    # Cleanup
    del vae, transformer, scheduler
    del tokenizer, text_encoder, tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Color palette
COLORS = {
    "individual_a": "#e63946",      # red
    "individual_b": "#457b9d",      # blue
    "compositional_text": "#2a9d8f",# teal
    "monolithic_and": "#2a9d8f",    # teal (same as compositional text)
    "superdiff_fm_ode": "#e9c46a",  # gold
    "geodesic_midpoint": "#6c757d", # gray
    "coco_images": "#adb5bd",       # light gray
}


def plot_manifold_projection(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, str],
    coco_embeds: np.ndarray,
    output_path: str,
    projection: str = "pca",
    pair_name: str = "",
):
    """Plot A: Joint 2D projection of all embeddings.

    Args:
        embeddings: name → (768,) array for each named point
        labels: name → display label
        coco_embeds: (N, 768) COCO ground-truth image embeddings
        output_path: where to save
        projection: "pca" or "umap"
        pair_name: for the title
    """
    # Collect all points for fitting the projection
    named_points = list(embeddings.keys())
    all_points = np.stack([embeddings[k] for k in named_points])  # (M, 768)
    all_data = np.vstack([all_points, coco_embeds])  # (M+N, 768)

    if projection == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        proj = reducer.fit_transform(all_data)
        axis_label = "UMAP"
        var_info = ""
    else:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(all_data)
        axis_label = "PC"
        var_info = (f" ({pca.explained_variance_ratio_[0]:.1%})",
                    f" ({pca.explained_variance_ratio_[1]:.1%})")

    proj_named = proj[:len(named_points)]
    proj_coco = proj[len(named_points):]

    fig, ax = plt.subplots(figsize=(10, 8))

    # COCO cloud
    ax.scatter(
        proj_coco[:, 0], proj_coco[:, 1],
        c=COLORS["coco_images"], s=12, alpha=0.35, edgecolors="none",
        label=f"COCO images ({len(coco_embeds)})", zorder=2,
    )

    # Named points
    markers = {
        "individual_a": "D", "individual_b": "D",
        "compositional_text": "^",
        "monolithic_and": "s", "superdiff_fm_ode": "*",
        "geodesic_midpoint": "P",
    }
    sizes = {
        "individual_a": 150, "individual_b": 150,
        "compositional_text": 150,
        "monolithic_and": 180, "superdiff_fm_ode": 300,
        "geodesic_midpoint": 150,
    }

    for name in named_points:
        idx = named_points.index(name)
        color = COLORS.get(name, "#333333")
        marker = markers.get(name, "o")
        size = sizes.get(name, 120)
        ax.scatter(
            proj_named[idx, 0], proj_named[idx, 1],
            c=color, s=size, marker=marker,
            edgecolors="black", linewidths=1.0, zorder=10,
            label=labels[name],
        )

    # Draw geodesic arc (line between anchors through midpoint)
    if "individual_a" in embeddings and "individual_b" in embeddings:
        a_idx = named_points.index("individual_a")
        b_idx = named_points.index("individual_b")
        ax.plot(
            [proj_named[a_idx, 0], proj_named[b_idx, 0]],
            [proj_named[a_idx, 1], proj_named[b_idx, 1]],
            "k--", alpha=0.3, linewidth=1, zorder=3,
        )

    xlabel = f"{axis_label}1{var_info[0] if var_info else ''}"
    ylabel = f"{axis_label}2{var_info[1] if var_info else ''}"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"CLIP Embedding Space: {pair_name}\n"
        "Co-occurrence (COCO) vs Hybrid (SuperDIFF) vs Individual Concepts",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_geodesic_decomposition(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, str],
    coco_embeds: np.ndarray,
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
    output_path: str,
    pair_name: str = "",
):
    """Plot B: Geodesic decomposition — tangential (t) vs residual (r).

    t measures position along the concept arc (a ↔ b)
    r measures deviation from the arc (co-occurrence vs hybrid)
    """
    # Decompose COCO images
    t_coco, r_coco = geodesic_decomposition(coco_embeds, anchor_a, anchor_b)

    # Decompose named points
    named_points = list(embeddings.keys())
    named_arr = np.stack([embeddings[k] for k in named_points])
    t_named, r_named = geodesic_decomposition(named_arr, anchor_a, anchor_b)

    fig, ax = plt.subplots(figsize=(10, 7))

    # COCO cloud
    ax.scatter(
        t_coco, r_coco,
        c=COLORS["coco_images"], s=12, alpha=0.35, edgecolors="none",
        label=f"COCO co-occurrence ({len(coco_embeds)})", zorder=2,
    )

    # Named points
    markers = {
        "individual_a": "D", "individual_b": "D",
        "compositional_text": "^",
        "monolithic_and": "s", "superdiff_fm_ode": "*",
        "geodesic_midpoint": "P",
    }
    sizes = {
        "individual_a": 150, "individual_b": 150,
        "compositional_text": 150,
        "monolithic_and": 180, "superdiff_fm_ode": 300,
        "geodesic_midpoint": 150,
    }

    for i, name in enumerate(named_points):
        color = COLORS.get(name, "#333333")
        marker = markers.get(name, "o")
        size = sizes.get(name, 120)
        ax.scatter(
            t_named[i], r_named[i],
            c=color, s=size, marker=marker,
            edgecolors="black", linewidths=1.0, zorder=10,
            label=labels[name],
        )

    # Reference lines
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.15, linewidth=0.8)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.15, linewidth=0.8)

    # Annotate regions
    ax.text(0.02, 0.98, "← closer to A", transform=ax.transAxes,
            fontsize=8, color="#e63946", alpha=0.6, va="top")
    ax.text(0.98, 0.98, "closer to B →", transform=ax.transAxes,
            fontsize=8, color="#457b9d", alpha=0.6, va="top", ha="right")
    ax.text(0.5, 0.98, "↑ more hybrid / off-geodesic", transform=ax.transAxes,
            fontsize=8, color="#666", alpha=0.6, va="top", ha="center")
    ax.text(0.5, 0.02, "↓ more co-occurrence / on-geodesic", transform=ax.transAxes,
            fontsize=8, color="#666", alpha=0.6, va="bottom", ha="center")

    ax.set_xlabel("t  (tangential: position along concept arc)", fontsize=11)
    ax.set_ylabel("r  (residual: deviation from concept arc)", fontsize=11)
    ax.set_title(
        f"Geodesic Decomposition: {pair_name}\n"
        "Co-occurrence clusters near arc (low r); hybrids deviate (high r)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_cosine_distance_bars(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, str],
    coco_embeds: np.ndarray,
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
    output_path: str,
    pair_name: str = "",
):
    """Plot C: Bar chart of cosine similarities to each anchor + midpoint."""
    midpoint = compute_geodesic_midpoint(anchor_a, anchor_b)

    refs = {
        "Concept A": anchor_a,
        "Concept B": anchor_b,
        "Geodesic Mid": midpoint,
    }

    # Compute similarities for named points + COCO mean
    point_names = list(embeddings.keys())
    sims = {}
    for pname in point_names:
        sims[labels[pname]] = {
            rname: float(np.dot(embeddings[pname], ref))
            for rname, ref in refs.items()
        }

    # COCO mean embedding
    coco_mean = coco_embeds.mean(axis=0)
    coco_mean = coco_mean / np.linalg.norm(coco_mean)
    sims[f"COCO mean (n={len(coco_embeds)})"] = {
        rname: float(np.dot(coco_mean, ref))
        for rname, ref in refs.items()
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    group_names = list(sims.keys())
    ref_names = list(refs.keys())
    n_groups = len(group_names)
    n_refs = len(ref_names)
    x = np.arange(n_groups)
    width = 0.25

    ref_colors = ["#e63946", "#457b9d", "#6c757d"]

    for j, (rname, color) in enumerate(zip(ref_names, ref_colors)):
        vals = [sims[g][rname] for g in group_names]
        bars = ax.bar(x + j * width, vals, width, label=rname, color=color, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(group_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title(
        f"Cosine Similarity to Concept Anchors: {pair_name}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def print_summary_table(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, str],
    coco_embeds: np.ndarray,
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
):
    """Print a summary table of cosine similarities and geodesic metrics."""
    midpoint = compute_geodesic_midpoint(anchor_a, anchor_b)

    # Geodesic distance = arccos(cosine_sim)
    print("\n" + "=" * 80)
    print("SUMMARY: Cosine Similarities & Geodesic Distances")
    print("=" * 80)
    header = f"{'Point':<35} {'cos(A)':>8} {'cos(B)':>8} {'cos(mid)':>8} {'t':>8} {'r':>8}"
    print(header)
    print("-" * 80)

    all_names = list(embeddings.keys())
    all_arr = np.stack([embeddings[k] for k in all_names])
    t_vals, r_vals = geodesic_decomposition(all_arr, anchor_a, anchor_b)

    for i, name in enumerate(all_names):
        e = embeddings[name]
        cos_a = np.dot(e, anchor_a)
        cos_b = np.dot(e, anchor_b)
        cos_mid = np.dot(e, midpoint)
        print(f"{labels[name]:<35} {cos_a:>8.4f} {cos_b:>8.4f} {cos_mid:>8.4f} "
              f"{t_vals[i]:>8.4f} {r_vals[i]:>8.4f}")

    # COCO stats
    coco_cos_a = coco_embeds @ anchor_a
    coco_cos_b = coco_embeds @ anchor_b
    coco_cos_mid = coco_embeds @ midpoint
    t_coco, r_coco = geodesic_decomposition(coco_embeds, anchor_a, anchor_b)

    print(f"{'COCO mean':<35} {coco_cos_a.mean():>8.4f} {coco_cos_b.mean():>8.4f} "
          f"{coco_cos_mid.mean():>8.4f} {t_coco.mean():>8.4f} {r_coco.mean():>8.4f}")
    print(f"{'COCO std':<35} {coco_cos_a.std():>8.4f} {coco_cos_b.std():>8.4f} "
          f"{coco_cos_mid.std():>8.4f} {t_coco.std():>8.4f} {r_coco.std():>8.4f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Manifold visualization: co-occurrence vs hybrid vs individual concepts"
    )
    parser.add_argument("--pair", default="car+truck",
                        help="COCO pair to analyze (default: car+truck)")
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca",
                        help="2D projection method")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: results/manifold_analysis/<pair>)")
    parser.add_argument("--generate", action="store_true",
                        help="Generate fresh SD3 outputs (requires GPU)")
    parser.add_argument("--experiment-dir", type=Path, default=None,
                        help="Load decoded images from existing trajectory experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=1,
                        help="Number of seeds to generate (averages embedding)")
    args = parser.parse_args()

    # Validate pair
    available_pairs = load_pair_list()
    if args.pair not in available_pairs:
        print(f"Pair '{args.pair}' not found. Available pairs:")
        for p in available_pairs:
            print(f"  {p}")
        sys.exit(1)

    # Output directory
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "results" / "manifold_analysis" / args.pair
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pair: {args.pair}")
    print(f"Output: {args.output_dir}")
    print()

    # ------------------------------------------------------------------
    # 1. Load text embeddings (anchors)
    # ------------------------------------------------------------------
    print("Loading text embeddings...")
    text_data = load_text_embeddings(args.pair)
    anchor_a = text_data["individual_a"]
    anchor_b = text_data["individual_b"]
    comp_text = text_data["compositional"]
    prompt_a = text_data["prompt_a"]
    prompt_b = text_data["prompt_b"]
    prompt_comp = text_data["prompt_comp"]

    print(f"  Concept A: {prompt_a}")
    print(f"  Concept B: {prompt_b}")
    print(f"  Compositional: {prompt_comp}")

    # ------------------------------------------------------------------
    # 2. Load COCO ground-truth image embeddings
    # ------------------------------------------------------------------
    print("Loading COCO image embeddings...")
    coco_embeds = load_coco_image_embeddings(args.pair)
    print(f"  {len(coco_embeds)} COCO images for pair '{args.pair}'")

    # ------------------------------------------------------------------
    # 3. Compute geodesic midpoint
    # ------------------------------------------------------------------
    midpoint = compute_geodesic_midpoint(anchor_a, anchor_b)

    # ------------------------------------------------------------------
    # 4. Collect all named embeddings
    # ------------------------------------------------------------------
    embeddings = {
        "individual_a": anchor_a,
        "individual_b": anchor_b,
        "compositional_text": comp_text,
        "geodesic_midpoint": midpoint,
    }
    labels = {
        "individual_a": f'Text: "{prompt_a}"',
        "individual_b": f'Text: "{prompt_b}"',
        "compositional_text": f'Text: "{prompt_comp}"',
        "geodesic_midpoint": "Geodesic midpoint",
    }

    # ------------------------------------------------------------------
    # 5. Optionally generate or load SD3 outputs and encode with CLIP
    # ------------------------------------------------------------------
    generated_images = {}

    if args.generate:
        print("\nGenerating SD3.5 outputs...")
        generated_images = generate_outputs(
            prompt_a, prompt_b, seed=args.seed,
        )
    elif args.experiment_dir is not None:
        print(f"\nLoading images from: {args.experiment_dir}")
        generated_images = load_experiment_images(args.experiment_dir)
        if not generated_images:
            print("  WARNING: No images found in experiment dir. "
                  "Proceeding without generated outputs.")

    # Track image-space embeddings separately for image-only analysis
    image_embeddings = {}  # only CLIP image embeddings (same modality as COCO)
    image_labels = {}

    if generated_images:
        print(f"\nEncoding {len(generated_images)} generated images with CLIP...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for name, img in generated_images.items():
            clip_emb = encode_images_with_clip([img], device=device)
            embeddings[name] = clip_emb[0]
            image_embeddings[name] = clip_emb[0]
            # Auto-label
            if name not in labels:
                pretty = name.replace("_", " ").title()
                labels[name] = pretty
            image_labels[name] = labels[name]

    # ------------------------------------------------------------------
    # 6. Determine image-space anchors for geodesic decomposition
    # ------------------------------------------------------------------
    # The CLIP modality gap means text and image embeddings live in
    # different subspaces. For geodesic decomposition to be meaningful,
    # anchors must share the same modality as the points being analyzed.
    #
    # Strategy:
    #   - Text-space analysis: uses text anchors (always available)
    #   - Image-space analysis: uses generated image anchors (when --generate)
    #     Falls back to COCO centroid-based anchors otherwise.
    # ------------------------------------------------------------------

    # Image-space anchors: use generated individual images if available,
    # otherwise approximate from COCO cluster centroids
    if "individual_a" in image_embeddings and "individual_b" in image_embeddings:
        img_anchor_a = image_embeddings["individual_a"]
        img_anchor_b = image_embeddings["individual_b"]
        print("\n  Using generated image embeddings as image-space anchors")
    else:
        # Approximate: use COCO mean as a single image-space reference
        coco_mean = coco_embeds.mean(axis=0)
        img_anchor_a = coco_mean / np.linalg.norm(coco_mean)
        img_anchor_b = img_anchor_a  # degenerate — will skip image-space plots
        print("\n  No generated images — image-space geodesic decomposition unavailable")

    has_image_anchors = not np.allclose(img_anchor_a, img_anchor_b)
    img_midpoint = compute_geodesic_midpoint(img_anchor_a, img_anchor_b) if has_image_anchors else None

    # ------------------------------------------------------------------
    # 7. Summary tables
    # ------------------------------------------------------------------
    print("\n--- TEXT-SPACE analysis (text anchors) ---")
    print_summary_table(embeddings, labels, coco_embeds, anchor_a, anchor_b)

    if has_image_anchors:
        # Build image-only embedding dict for image-space analysis
        img_analysis_embeds = dict(image_embeddings)
        img_analysis_labels = dict(image_labels)
        if img_midpoint is not None:
            img_analysis_embeds["img_geodesic_mid"] = img_midpoint
            img_analysis_labels["img_geodesic_mid"] = "Image geodesic midpoint"
            COLORS["img_geodesic_mid"] = "#6c757d"

        print("\n--- IMAGE-SPACE analysis (generated image anchors) ---")
        print_summary_table(img_analysis_embeds, img_analysis_labels,
                            coco_embeds, img_anchor_a, img_anchor_b)

    # ------------------------------------------------------------------
    # 8. Generate plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")

    pair_display = args.pair.replace("+", " + ")

    # Plot A: Manifold projection (all embeddings, both modalities)
    plot_manifold_projection(
        embeddings, labels, coco_embeds,
        str(args.output_dir / f"manifold_{args.projection}.png"),
        projection=args.projection,
        pair_name=pair_display,
    )

    # Plot B: Geodesic decomposition — IMAGE-SPACE (meaningful)
    if has_image_anchors:
        # Image-space geodesic decomposition: anchors and points in same modality
        plot_geodesic_decomposition(
            img_analysis_embeds, img_analysis_labels, coco_embeds,
            img_anchor_a, img_anchor_b,
            str(args.output_dir / "geodesic_decomposition_image_space.png"),
            pair_name=f"{pair_display} (image-space anchors)",
        )
    else:
        print("  Skipping image-space geodesic decomposition (no generated images)")

    # Plot B2: Geodesic decomposition — TEXT-SPACE (for reference)
    plot_geodesic_decomposition(
        embeddings, labels, coco_embeds,
        anchor_a, anchor_b,
        str(args.output_dir / "geodesic_decomposition_text_space.png"),
        pair_name=f"{pair_display} (text-space anchors)",
    )

    # Plot C: Cosine similarity bars — IMAGE-SPACE
    if has_image_anchors:
        plot_cosine_distance_bars(
            img_analysis_embeds, img_analysis_labels, coco_embeds,
            img_anchor_a, img_anchor_b,
            str(args.output_dir / "cosine_similarity_bars_image_space.png"),
            pair_name=f"{pair_display} (image-space)",
        )

    # Plot C2: Cosine similarity bars — TEXT-SPACE (for reference)
    plot_cosine_distance_bars(
        embeddings, labels, coco_embeds,
        anchor_a, anchor_b,
        str(args.output_dir / "cosine_similarity_bars_text_space.png"),
        pair_name=f"{pair_display} (text-space)",
    )

    # Save embeddings for reproducibility
    np.savez(
        args.output_dir / "embeddings.npz",
        **{f"named_{k}": v for k, v in embeddings.items()},
        coco_images=coco_embeds,
        anchor_a=anchor_a,
        anchor_b=anchor_b,
    )
    print(f"\n  Saved embeddings: {args.output_dir / 'embeddings.npz'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
