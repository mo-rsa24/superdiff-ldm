"""
Phase 3: Measure the composability gap between SuperDiff-AND and SD3.5(p*).

For each held-out test pair (c₁, c₂):
  1. Generate images from SuperDiff-AND(c₁, c₂) with trajectory tracking
  2. Invert each image via f_θ to get p* = (pooled_embeds_pred, seq_embeds_pred)
  3. Generate images from SD3.5(p*) with same initial noise + trajectory tracking
  4. Generate images from SD3.5("c₁ and c₂") as the naive monolithic baseline
  5. Compute gap metrics at three levels:
       - Image:      CLIP cosine similarity, LPIPS
       - Latent:     MSE in VAE latent space at the final step
       - Trajectory: step-wise MSE and cosine similarity

Results are saved as gap_metrics.json + image grids per pair.

Usage
-----
conda run -n superdiff python scripts/measure_composability_gap.py \
    [--ckpt ckpt/inverter/best.pt] \
    [--model-id stabilityai/stable-diffusion-3.5-medium] \
    [--output-dir experiments/inversion/gap_analysis] \
    [--seeds 0 1 2 3 4 5 6 7] \
    [--steps 50] \
    [--guidance 4.5]
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Trajectory MDS will be skipped.")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from notebooks.utils import get_sd3_models, get_sd3_text_embedding
from notebooks.composition_experiments import (
    LatentTrajectoryCollector,
    sample_sd3_with_trajectory_tracking,
    get_vel_sd3,
)
from scripts.trajectory_dynamics_experiment import superdiff_fm_ode_sd3
from models.sd35_inverter import load_inverter, make_clip_preprocessor

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. LPIPS metric will be skipped.")
    print("  Install with: pip install lpips")


# ---------------------------------------------------------------------------
# Test pairs (held out from training)
# ---------------------------------------------------------------------------

TEST_PAIRS = [
    ("a cat",    "a dog"),
    ("a person", "an umbrella"),
    ("a person", "a car"),
    ("a car",    "a truck"),
]


# ---------------------------------------------------------------------------
# SD3.5 sampling with pre-computed conditioning
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_sd3_with_precomputed_cond(
    latents: torch.Tensor,
    cond_embeds: torch.Tensor,
    cond_pooled: torch.Tensor,
    uncond_embeds: torch.Tensor,
    uncond_pooled: torch.Tensor,
    scheduler,
    transformer,
    guidance_scale: float = 4.5,
    num_inference_steps: int = 50,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> tuple:
    """
    CFG sampling using pre-computed conditioning embeddings.
    Returns (final_latents, LatentTrajectoryCollector).
    """
    B = latents.shape[0]
    tracker = LatentTrajectoryCollector(
        num_inference_steps, B,
        latents.shape[1], latents.shape[2], latents.shape[3],
    )

    # Expand pre-computed conditioning to batch size
    c_embeds = cond_embeds.expand(B, -1, -1).to(device=device, dtype=dtype)
    c_pooled = cond_pooled.expand(B, -1).to(device=device, dtype=dtype)
    u_embeds = uncond_embeds.expand(B, -1, -1).to(device=device, dtype=dtype)
    u_pooled = uncond_pooled.expand(B, -1).to(device=device, dtype=dtype)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]

        vel_cond  = get_vel_sd3(transformer, t, latents, c_embeds, c_pooled,
                                device=device, dtype=dtype)
        vel_uncond = get_vel_sd3(transformer, t, latents, u_embeds, u_pooled,
                                  device=device, dtype=dtype)

        vf = vel_uncond + guidance_scale * (vel_cond - vel_uncond)
        dt = scheduler.sigmas[i + 1] - sigma
        tracker.store_step(i, latents, vf, float(sigma), t.item())
        latents = latents + dt * vf

    tracker.store_final(latents)
    return latents, tracker


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.to(dtype=vae.dtype)
    images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return ((images / 2 + 0.5).clamp(0, 1)).float()


_CLIP_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                         [0.26862954, 0.26130258, 0.27577711]),
])


def images_to_clip_input(images_01: torch.Tensor) -> torch.Tensor:
    """Convert (B, 3, H, W) [0,1] float to CLIP-preprocessed (B, 3, 224, 224)."""
    result = []
    for img in images_01.cpu():
        pil = transforms.ToPILImage()(img)
        result.append(_CLIP_EVAL_TRANSFORM(pil))
    return torch.stack(result)


# ---------------------------------------------------------------------------
# Gap metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_image_gap(
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    clip_model,
    lpips_fn=None,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """
    images_a, images_b: (N, 3, H, W) float32 [0, 1]
    Returns dict with clip_cos, lpips (if available).
    """
    a_clip = images_to_clip_input(images_a).to(device)
    b_clip = images_to_clip_input(images_b).to(device)

    feat_a = clip_model.get_image_features(pixel_values=a_clip)
    feat_b = clip_model.get_image_features(pixel_values=b_clip)

    feat_a = F.normalize(feat_a, dim=-1)
    feat_b = F.normalize(feat_b, dim=-1)

    clip_cos = (feat_a * feat_b).sum(dim=-1).mean().item()

    result = {"clip_cos": clip_cos}

    if lpips_fn is not None:
        # lpips expects [-1, 1]; kept on CPU to avoid OOM alongside the transformer
        a_lp = images_a.cpu().float() * 2 - 1
        b_lp = images_b.cpu().float() * 2 - 1
        lp_vals = lpips_fn(a_lp, b_lp)
        result["lpips"] = lp_vals.mean().item()

    return result


@torch.no_grad()
def compute_latent_gap(
    vae,
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """MSE in VAE latent space between corresponding images."""
    a = images_a.to(device=device, dtype=dtype) * 2 - 1
    b = images_b.to(device=device, dtype=dtype) * 2 - 1

    lat_a = vae.encode(a).latent_dist.mean
    lat_b = vae.encode(b).latent_dist.mean

    lat_mse = F.mse_loss(lat_a, lat_b).item()
    return {"lat_mse": lat_mse}


def compute_trajectory_gap(
    tracker_a: LatentTrajectoryCollector,
    tracker_b: LatentTrajectoryCollector,
) -> dict:
    """
    Step-wise MSE and cosine similarity between two trajectory collectors.
    Both must have the same number of steps and batch size.
    Returns: {traj_mse_mean, traj_cos_mean, traj_mse_per_step, traj_cos_per_step}
    """
    T = tracker_a.trajectories.shape[0] - 1  # exclude final stored step
    mse_per_step = []
    cos_per_step = []

    for t in range(T):
        a_t = tracker_a.trajectories[t].flatten(1)  # (B, D)
        b_t = tracker_b.trajectories[t].flatten(1)

        mse = F.mse_loss(a_t, b_t).item()
        cos = F.cosine_similarity(a_t, b_t, dim=-1).mean().item()

        mse_per_step.append(mse)
        cos_per_step.append(cos)

    return {
        "traj_mse_mean": float(sum(mse_per_step) / len(mse_per_step)),
        "traj_cos_mean": float(sum(cos_per_step) / len(cos_per_step)),
        "traj_mse_per_step": [float(v) for v in mse_per_step],
        "traj_cos_per_step": [float(v) for v in cos_per_step],
    }


# ---------------------------------------------------------------------------
# Best-of-K candidate selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def select_best_p_star(
    candidates: list,
    x_T: torch.Tensor,
    target_img: torch.Tensor,
    transformer,
    scheduler,
    uncond_embeds: torch.Tensor,
    uncond_pooled: torch.Tensor,
    guidance_scale: float,
    vae,
    clip_eval,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """
    Pick the best p* from K candidates using a single-NFE x₀ prediction.

    Formula:  x₀_pred = x_T - σ_max · v_cfg(x_T, t_max, p*_k)

    This is the direct clean-image prediction at the first (noisiest)
    timestep — one transformer call per candidate, no ODE solve.
    The winner is the candidate whose predicted x₀ has the highest
    CLIP cosine similarity to the reference AND image.

    Scheduler must already have set_timesteps called (e.g. by the AND
    run that preceded this call) — we read sigmas[0] without resetting.

    Parameters
    ----------
    candidates  : list of k (pooled: Tensor(1,2048), seq: Tensor(1,410,4096))
    x_T         : (1, 16, H, W) — shared initial noise for this seed
    target_img  : (1, 3, H, W) float32 [0,1] — AND image for this seed
    ...

    Returns
    -------
    (best_pooled, best_seq, best_score)
    best_score is None when len(candidates)==1.
    """
    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1], None

    # Timestep / sigma at t_max — read from already-configured scheduler
    t_max     = scheduler.timesteps[0]
    sigma_max = float(scheduler.sigmas[0])   # ≈ 1.0 for flow matching

    # Unconditional velocity at x_T — same for all candidates, compute once
    vel_uncond = get_vel_sd3(
        transformer, t_max, x_T,
        uncond_embeds.expand(1, -1, -1),
        uncond_pooled.expand(1, -1),
        device=device, dtype=dtype,
    )

    # CLIP features of the AND target image — computed once
    target_clip_in = images_to_clip_input(target_img.cpu()).to(device)
    feat_target = F.normalize(
        clip_eval.get_image_features(pixel_values=target_clip_in), dim=-1
    )  # (1, D_clip)

    best_score  = -float("inf")
    best_pooled, best_seq = candidates[0]

    for pooled_k, seq_k in candidates:
        c_embeds = seq_k.to(device=device, dtype=dtype)
        c_pooled = pooled_k.to(device=device, dtype=dtype)

        vel_cond = get_vel_sd3(
            transformer, t_max, x_T, c_embeds, c_pooled,
            device=device, dtype=dtype,
        )
        vf = vel_uncond + guidance_scale * (vel_cond - vel_uncond)

        # Direct x₀ prediction (flow-matching: x₀ = x_T - σ · v)
        x0_pred = x_T - sigma_max * vf

        img_pred = decode_latents(vae, x0_pred)
        approx_clip = images_to_clip_input(img_pred.cpu()).to(device)
        feat_approx = F.normalize(
            clip_eval.get_image_features(pixel_values=approx_clip), dim=-1
        )

        score = (feat_target * feat_approx).sum().item()
        if score > best_score:
            best_score  = score
            best_pooled, best_seq = pooled_k, seq_k

    return best_pooled, best_seq, best_score


# ---------------------------------------------------------------------------
# 1. Text decoding: find nearest training prompt to predicted p*
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_pooled_to_text(
    pred_pooled: torch.Tensor,          # (2048,) — averaged across seeds
    vocab_prompts: List[str],
    models: dict,
    device: torch.device,
    top_k: int = 3,
) -> List[tuple]:
    """
    Find the top-k nearest training prompts to the predicted pooled embedding
    by cosine similarity in SD3.5's pooled conditioning space (2048-dim).

    Returns list of (prompt_str, cosine_similarity) sorted descending.
    """
    query = F.normalize(pred_pooled.float().unsqueeze(0), dim=-1)  # (1, 2048)

    sims = []
    for prompt in vocab_prompts:
        _, pooled = get_sd3_text_embedding(
            [prompt],
            models["tokenizer"],   models["text_encoder"],
            models["tokenizer_2"], models["text_encoder_2"],
            models["tokenizer_3"], models["text_encoder_3"],
            device=device,
        )
        pooled_n = F.normalize(pooled.float(), dim=-1)  # (1, 2048)
        sim = (query * pooled_n).sum().item()
        sims.append((prompt, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


# ---------------------------------------------------------------------------
# 2. Side-by-side comparison grid
# ---------------------------------------------------------------------------

def _add_text_label(img_tensor: torch.Tensor, label: str, font_size: int = 16) -> torch.Tensor:
    """Burn a text label into the top of a (3, H, W) [0,1] float tensor."""
    pil = transforms.ToPILImage()(img_tensor.clamp(0, 1))
    draw = ImageDraw.Draw(pil)
    # Use default font — no TTF dependency
    draw.rectangle([0, 0, pil.width, font_size + 6], fill=(20, 20, 20))
    draw.text((4, 3), label, fill=(255, 255, 255))
    return transforms.ToTensor()(pil)


def plot_comparison_grid(
    imgs_and:   torch.Tensor,   # (N, 3, H, W) [0,1]
    imgs_pstar: torch.Tensor,   # (N, 3, H, W)
    imgs_mono:  torch.Tensor,   # (N, 3, H, W)
    decoded_text: List[tuple],  # top-k (prompt, sim) from decode_pooled_to_text
    c1: str,
    c2: str,
    out_path: Path,
    n_display: int = 4,
):
    """
    Save a 3-row comparison grid:
      Row 0 — SuperDiff AND
      Row 1 — SD3.5(p*)  [with decoded nearest prompts annotated below]
      Row 2 — SD3.5 monolithic  "c1 and c2"

    Each row shows n_display images side by side.
    Decoded text is printed below the p* row as a caption block.
    """
    N = min(n_display, imgs_and.shape[0])

    def label_row(imgs, label):
        labelled = [_add_text_label(imgs[i], label) for i in range(N)]
        return torch.stack(labelled)   # (N, 3, H, W)

    row_and   = label_row(imgs_and,   f"SuperDiff AND  ({c1}  +  {c2})")
    row_pstar = label_row(imgs_pstar, "SD3.5 ( p* — inverted )")
    row_mono  = label_row(imgs_mono,  f"SD3.5 monolithic  \"{c1} and {c2}\"")

    combined = torch.cat([row_and, row_pstar, row_mono], dim=0)  # (3N, 3, H, W)
    grid = make_grid(combined, nrow=N, padding=4, normalize=False)

    # Convert to PIL to add a caption strip at the bottom
    grid_pil = transforms.ToPILImage()(grid)
    line_h = 20
    caption_lines = ["Nearest training prompts to p* (cosine similarity in SD3.5 pooled space):"]
    for rank, (prompt, sim) in enumerate(decoded_text, 1):
        caption_lines.append(f"  {rank}. \"{prompt}\"  —  {sim:.4f}")

    caption_h = line_h * len(caption_lines) + 10
    canvas = Image.new("RGB", (grid_pil.width, grid_pil.height + caption_h), color=(30, 30, 30))
    canvas.paste(grid_pil, (0, 0))

    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(caption_lines):
        draw.text((8, grid_pil.height + 5 + i * line_h), line, fill=(220, 220, 220))

    canvas.save(str(out_path))


# ---------------------------------------------------------------------------
# 3. Trajectory MDS / PCA visualisation
# ---------------------------------------------------------------------------

def _collect_trajectory_array(tracker: LatentTrajectoryCollector) -> np.ndarray:
    """Flatten trajectories to (T+1, D) numpy array."""
    traj = tracker.trajectories  # (T+1, B, C, H, W)
    T1, B, C, H, W = traj.shape
    return traj[:, 0].reshape(T1, -1).numpy().astype(np.float32)  # (T+1, D)


def plot_trajectory_mds(
    tracker_and:   LatentTrajectoryCollector,
    tracker_pstar: LatentTrajectoryCollector,
    tracker_mono:  LatentTrajectoryCollector,
    c1: str,
    c2: str,
    out_path: Path,
    method: str = "pca",
):
    """
    Jointly project the latent trajectories of:
      - SuperDiff AND
      - SD3.5(p*)
      - SD3.5 monolithic
    into 2D via PCA or MDS and plot time-coloured curves.

    All three start from the same x_T (shared noise), so the origin is common.
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        print("  [skip] trajectory MDS — matplotlib or scikit-learn not available")
        return

    traj_and   = _collect_trajectory_array(tracker_and)    # (T+1, D)
    traj_pstar = _collect_trajectory_array(tracker_pstar)  # (T+1, D)
    traj_mono  = _collect_trajectory_array(tracker_mono)   # (T+1, D)

    T1 = traj_and.shape[0]
    stacked = np.vstack([traj_and, traj_pstar, traj_mono])  # (3*(T+1), D)

    # Dimensionality reduction
    if method == "pca":
        proj = PCA(n_components=2).fit_transform(stacked)
        axis_label = "PC"
    else:  # mds
        from sklearn.metrics import pairwise_distances
        dist = pairwise_distances(stacked, metric="euclidean")
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, normalized_stress="auto")
        proj = mds.fit_transform(dist)
        axis_label = "MDS"

    p_and   = proj[:T1]
    p_pstar = proj[T1:2*T1]
    p_mono  = proj[2*T1:]

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=T1 - 1)

    def draw_curve(pts, label, color_offset=0.0):
        for t in range(len(pts) - 1):
            seg = np.array([[pts[t], pts[t + 1]]])
            lc = LineCollection(seg, colors=[cmap(norm(t + color_offset))],
                                linewidths=2, alpha=0.85)
            ax.add_collection(lc)
        ax.scatter(*pts[-1], s=120, zorder=5, edgecolors="k", linewidths=0.8,
                   color=cmap(norm(T1 - 1)))
        ax.annotate(label, pts[-1], xytext=(6, 4), textcoords="offset points", fontsize=10)

    draw_curve(p_and,   f"SuperDiff AND")
    draw_curve(p_pstar, f"SD3.5 (p*)")
    draw_curve(p_mono,  f"SD3.5 monolithic")

    # Shared origin marker (all start from same x_T)
    ax.plot(*p_and[0], "ko", markersize=9, zorder=6)
    ax.annotate("$x_T$ (shared)", p_and[0], xytext=(6, -14),
                textcoords="offset points", fontsize=9, color="black")

    # Colourbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Denoising step")

    ax.autoscale()
    ax.set_xlabel(f"{axis_label} 1", fontsize=12)
    ax.set_ylabel(f"{axis_label} 2", fontsize=12)
    ax.set_title(
        f"Latent Trajectory: SuperDiff AND vs SD3.5($p^*$) vs Monolithic\n"
        f"({c1}  +  {c2})",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=cmap(0.2), lw=2, label=f"SuperDiff AND ({c1} + {c2})"),
        Line2D([0], [0], color=cmap(0.5), lw=2, label="SD3.5 (p* — inverted)"),
        Line2D([0], [0], color=cmap(0.8), lw=2, label=f"SD3.5 monolithic \"{c1} and {c2}\""),
        Line2D([0], [0], marker="o", color="k", lw=0, markersize=8, label="Shared start $x_T$"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Measure composability gap")
    p.add_argument("--ckpt",       default="ckpt/inverter/best.pt")
    p.add_argument("--model-id",   default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--output-dir", default="experiments/inversion/gap_analysis")
    p.add_argument("--data-dir",   default="experiments/inversion/training_data",
                   help="Path to training data directory (for text-decoding vocabulary)")
    p.add_argument("--seeds",      type=int, nargs="+", default=list(range(8)))
    p.add_argument("--steps",      type=int,   default=50)
    p.add_argument("--guidance",   type=float, default=4.5)
    p.add_argument("--image-size", type=int,   default=512)
    p.add_argument("--dtype",      default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    p.add_argument("--projection", default="pca", choices=["pca", "mds"],
                   help="Dimensionality reduction method for trajectory visualisation")
    p.add_argument("--k-samples", type=int, default=1,
                   help="Best-of-K inversion: sample K p* candidates via MC dropout "
                        "and pick the best by single-NFE CLIP score. "
                        "k=1 (default) disables and uses a single deterministic forward pass.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.dtype == "float16" else torch.bfloat16

    out_root    = Path(args.output_dir)
    latent_size = args.image_size // 8

    # ---- Load SD3.5 ----
    print("Loading SD3.5 ...")
    models = get_sd3_models(model_id=args.model_id, dtype=dtype, device=device)

    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    # ---- Load inverter ----
    print(f"Loading inverter from {args.ckpt} ...")
    inverter   = load_inverter(args.ckpt, clip_model_id=args.clip_model_id, device=device)
    inverter   = inverter.eval().to(dtype=torch.float32)
    preprocess = make_clip_preprocessor(device)

    # ---- Load CLIP for evaluation (separate from inverter backbone) ----
    print("Loading CLIP for evaluation ...")
    from transformers import CLIPModel
    clip_eval = CLIPModel.from_pretrained(args.clip_model_id).to(device).eval()

    # ---- Load LPIPS if available ----
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net="vgg")  # kept on CPU; images moved to CPU at call time

    # ---- Pre-compute unconditional conditioning ----
    with torch.no_grad():
        uncond_embeds, uncond_pooled = get_sd3_text_embedding(
            [""],
            models["tokenizer"],   models["text_encoder"],
            models["tokenizer_2"], models["text_encoder_2"],
            models["tokenizer_3"], models["text_encoder_3"],
            device=device,
        )

    # ---- Load vocabulary for text decoding ----
    vocab_prompts = []
    index_path = Path(args.data_dir) / "dataset_index.json"
    if index_path.exists():
        with open(index_path) as f:
            vocab_prompts = [entry["prompt"] for entry in json.load(f)]
        print(f"Loaded {len(vocab_prompts)} vocab prompts from {index_path}")
    else:
        print(f"Warning: {index_path} not found — text decoding will be skipped")

    all_pair_results = []
    all_seed_records = []
    all_traj_records = []

    for c1, c2 in TEST_PAIRS:
        pair_slug = f"{c1.replace(' ', '_')}_{c2.replace(' ', '_')}"
        pair_dir  = out_root / pair_slug
        pair_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Pair: '{c1}' AND '{c2}'")
        print(f"{'='*60}")

        images_and   = []
        images_pstar = []
        images_mono  = []

        trackers_and   = []
        trackers_pstar = []
        trackers_mono  = []
        pred_pooled_list = []   # accumulate across seeds for text decoding

        # Monolithic prompt
        mono_prompt = f"{c1} and {c2}"

        images_c1        = []
        images_c2        = []
        trackers_c1      = []
        trackers_c2      = []
        per_seed_records = []

        for seed in args.seeds:
            gen = torch.Generator(device=device).manual_seed(seed)
            init_latents = torch.randn(
                1, 16, latent_size, latent_size,
                device=device, dtype=dtype, generator=gen,
            )

            # ---- Step 1: SuperDiff-AND ----
            with torch.no_grad():
                lat_and, tracker_and, *_ = superdiff_fm_ode_sd3(
                    latents=init_latents.clone(),
                    obj_prompt=c1,
                    bg_prompt=c2,
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
                img_and = decode_latents(models["vae"], lat_and)  # (1, 3, H, W)

            images_and.append(img_and.cpu())
            trackers_and.append(tracker_and)

            # ---- Step 2: Invert SuperDiff-AND image ----
            # Preprocess for CLIP (224×224, normalised)
            img_clip = preprocess(img_and)  # (1, 3, 224, 224), float32

            candidates = inverter.sample_k(img_clip, k=args.k_samples)

            pred_pooled, pred_seq, best_score = select_best_p_star(
                candidates,
                x_T=init_latents.clone(),
                target_img=img_and,
                transformer=models["transformer"],
                scheduler=scheduler,
                uncond_embeds=uncond_embeds,
                uncond_pooled=uncond_pooled,
                guidance_scale=args.guidance,
                vae=models["vae"],
                clip_eval=clip_eval,
                device=device,
                dtype=dtype,
            )
            if best_score is not None:
                print(f"    best-of-{args.k_samples} score: {best_score:.4f}")
            # pred_pooled: (1, 2048)
            # pred_seq:    (1, 410, 4096)  — T5 portion zeroed
            pred_pooled_list.append(pred_pooled.squeeze(0).float().cpu())

            # ---- Step 3: SD3.5(p*) ----
            with torch.no_grad():
                lat_pstar, tracker_pstar = sample_sd3_with_precomputed_cond(
                    latents=init_latents.clone(),
                    cond_embeds=pred_seq.to(dtype=dtype),
                    cond_pooled=pred_pooled.to(dtype=dtype),
                    uncond_embeds=uncond_embeds,
                    uncond_pooled=uncond_pooled,
                    scheduler=scheduler,
                    transformer=models["transformer"],
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                    device=device,
                    dtype=dtype,
                )
                img_pstar = decode_latents(models["vae"], lat_pstar)

            images_pstar.append(img_pstar.cpu())
            trackers_pstar.append(tracker_pstar)

            # ---- Step 4: Naive monolithic baseline ----
            with torch.no_grad():
                lat_mono, tracker_mono = sample_sd3_with_trajectory_tracking(
                    latents=init_latents.clone(),
                    prompt=mono_prompt,
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
                img_mono = decode_latents(models["vae"], lat_mono)

            images_mono.append(img_mono.cpu())
            trackers_mono.append(tracker_mono)

            # ---- c1-only and c2-only baselines (same initial noise) ----
            with torch.no_grad():
                lat_c1, tracker_c1 = sample_sd3_with_trajectory_tracking(
                    latents=init_latents.clone(),
                    prompt=c1,
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
                img_c1 = decode_latents(models["vae"], lat_c1)

                lat_c2, tracker_c2 = sample_sd3_with_trajectory_tracking(
                    latents=init_latents.clone(),
                    prompt=c2,
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
                img_c2 = decode_latents(models["vae"], lat_c2)

            images_c1.append(img_c1.cpu())
            images_c2.append(img_c2.cpu())
            trackers_c1.append(tracker_c1)
            trackers_c2.append(tracker_c2)

            # ---- Per-seed terminal latent distances (anchor = AND) ----
            # Use raw tracker latents — no VAE re-encode error.
            # Per-element MSE is dimension-normalised and scale-comparable.
            z_and     = tracker_and.trajectories[-1].float()   # (1, 16, H, W)
            z_mono    = tracker_mono.trajectories[-1].float()
            z_c1_t    = tracker_c1.trajectories[-1].float()
            z_c2_t    = tracker_c2.trajectories[-1].float()
            z_pstar_t = tracker_pstar.trajectories[-1].float()

            per_seed_records.append({
                "pair":      f"{c1} + {c2}",
                "c1":        c1,
                "c2":        c2,
                "seed":      seed,
                "d_T_mono":  float(((z_mono    - z_and) ** 2).mean()),
                "d_T_c1":    float(((z_c1_t    - z_and) ** 2).mean()),
                "d_T_c2":    float(((z_c2_t    - z_and) ** 2).mean()),
                "d_T_pstar": float(((z_pstar_t - z_and) ** 2).mean()),
            })

            print(f"  seed {seed} done")

        # ---- Per-step trajectory distances (CPU, no GPU needed) ----
        # trackers already hold trajectories as CPU tensors from store_step().
        # Shape: (num_steps+1, 1, C, H, W) — iterate over all T+1 states.
        print("  Computing per-step trajectory distances ...")
        for seed_idx, seed in enumerate(args.seeds):
            ta  = trackers_and[seed_idx]
            tm  = trackers_mono[seed_idx]
            tc1 = trackers_c1[seed_idx]
            tc2 = trackers_c2[seed_idx]
            tp  = trackers_pstar[seed_idx]
            n_steps = ta.trajectories.shape[0]   # T+1
            for step in range(n_steps):
                z_and     = ta.trajectories[step].float()
                z_mono    = tm.trajectories[step].float()
                z_c1_t    = tc1.trajectories[step].float()
                z_c2_t    = tc2.trajectories[step].float()
                z_pstar_s = tp.trajectories[step].float()
                sigma     = float(ta.sigmas[min(step, len(ta.sigmas) - 1)])
                all_traj_records.append({
                    "pair":      f"{c1} + {c2}",
                    "c1":        c1,
                    "c2":        c2,
                    "seed":      seed,
                    "step":      step,
                    "sigma":     sigma,
                    "d_t_mono":  float(((z_mono    - z_and) ** 2).mean()),
                    "d_t_c1":    float(((z_c1_t    - z_and) ** 2).mean()),
                    "d_t_c2":    float(((z_c2_t    - z_and) ** 2).mean()),
                    "d_t_pstar": float(((z_pstar_s - z_and) ** 2).mean()),
                })

        # Stack across seeds
        imgs_and   = torch.cat(images_and,   dim=0)  # (N, 3, H, W)
        imgs_pstar = torch.cat(images_pstar, dim=0)
        imgs_mono  = torch.cat(images_mono,  dim=0)
        imgs_c1    = torch.cat(images_c1,    dim=0)
        imgs_c2    = torch.cat(images_c2,    dim=0)

        # ---- Save image grids ----
        save_image(imgs_and,   pair_dir / "superdiff_and.png",   nrow=4, normalize=False)
        save_image(imgs_pstar, pair_dir / "sd35_pstar.png",      nrow=4, normalize=False)
        save_image(imgs_mono,  pair_dir / "sd35_monolithic.png", nrow=4, normalize=False)
        save_image(imgs_c1,    pair_dir / "sd35_c1_only.png",    nrow=4, normalize=False)
        save_image(imgs_c2,    pair_dir / "sd35_c2_only.png",    nrow=4, normalize=False)

        # ---- Step 5: Gap metrics ----
        print("  Computing image gap: AND vs p* ...")
        gap_and_pstar = compute_image_gap(imgs_and, imgs_pstar, clip_eval, lpips_fn, device)

        print("  Computing image gap: AND vs monolithic ...")
        gap_and_mono = compute_image_gap(imgs_and, imgs_mono, clip_eval, lpips_fn, device)

        print("  Computing trajectory gap: AND vs p* ...")
        # Aggregate trajectory gap across seeds by averaging
        traj_gap_list = [
            compute_trajectory_gap(ta, tp)
            for ta, tp in zip(trackers_and, trackers_pstar)
        ]
        traj_gap = {
            "traj_mse_mean": float(sum(t["traj_mse_mean"] for t in traj_gap_list) / len(traj_gap_list)),
            "traj_cos_mean": float(sum(t["traj_cos_mean"] for t in traj_gap_list) / len(traj_gap_list)),
        }

        print("  Computing VAE latent gap ...")
        lat_gap = compute_latent_gap(models["vae"], imgs_and, imgs_pstar, device, dtype)

        metrics = {
            "pair":       (c1, c2),
            "slug":       pair_slug,
            "n_seeds":    len(args.seeds),
            "gap_and_pstar": {**gap_and_pstar, **lat_gap, **traj_gap},
            "gap_and_mono":  gap_and_mono,
        }

        with open(pair_dir / "gap_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        all_pair_results.append(metrics)
        all_seed_records.extend(per_seed_records)

        # ---- Visualisation 1: text decoding of p* ----
        decoded_text = []
        if vocab_prompts:
            print("  Decoding p* to nearest training prompts ...")
            avg_pooled = torch.stack(pred_pooled_list).mean(dim=0).to(device)
            decoded_text = decode_pooled_to_text(
                avg_pooled, vocab_prompts, models, device, top_k=3
            )
            print(f"  Nearest prompts to p*:")
            for rank, (prompt, sim) in enumerate(decoded_text, 1):
                print(f"    {rank}. \"{prompt}\"  (cos={sim:.4f})")
            metrics["decoded_p_star"] = [
                {"prompt": p, "cos_sim": round(s, 4)} for p, s in decoded_text
            ]
            with open(pair_dir / "gap_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # ---- Visualisation 2: side-by-side comparison grid ----
        print("  Saving comparison grid ...")
        plot_comparison_grid(
            imgs_and, imgs_pstar, imgs_mono,
            decoded_text=decoded_text,
            c1=c1, c2=c2,
            out_path=pair_dir / "comparison_grid.png",
            n_display=4,
        )

        # ---- Visualisation 3: trajectory MDS/PCA (seed 0) ----
        print(f"  Plotting trajectory {args.projection.upper()} (seed 0) ...")
        plot_trajectory_mds(
            tracker_and=trackers_and[0],
            tracker_pstar=trackers_pstar[0],
            tracker_mono=trackers_mono[0],
            c1=c1, c2=c2,
            out_path=pair_dir / f"trajectory_{args.projection}.png",
            method=args.projection,
        )

        print(f"\n  Results for '{c1}' AND '{c2}':")
        print(f"    AND vs p*:        CLIP={gap_and_pstar['clip_cos']:.4f}  "
              f"lat_mse={lat_gap['lat_mse']:.4f}  "
              f"traj_mse={traj_gap['traj_mse_mean']:.4f}  "
              f"traj_cos={traj_gap['traj_cos_mean']:.4f}")
        print(f"    AND vs mono:      CLIP={gap_and_mono['clip_cos']:.4f}")
        if "lpips" in gap_and_pstar:
            print(f"    LPIPS (AND/p*):  {gap_and_pstar['lpips']:.4f}")

    # Save combined summary
    summary_path = out_root / "all_pairs_gap.json"
    with open(summary_path, "w") as f:
        json.dump(all_pair_results, f, indent=2)

    # Save flat per-seed distances for histogram plotting
    seed_records_path = out_root / "per_seed_distances.json"
    with open(seed_records_path, "w") as f:
        json.dump(all_seed_records, f, indent=2)

    traj_path = out_root / "trajectory_distances.json"
    with open(traj_path, "w") as f:
        json.dump(all_traj_records, f, indent=2)

    print(f"\nAll results saved to {out_root}")
    print(f"Per-seed distances  → {seed_records_path}")
    print(f"Trajectory distances → {traj_path}")


if __name__ == "__main__":
    main()
