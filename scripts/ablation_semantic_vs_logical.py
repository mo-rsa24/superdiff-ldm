"""
Mini-Ablation: Semantic (CLIP AND) vs Logical (SuperDIFF AND) Composition

Tests whether SuperDIFF AND provides any latent-space signal distinct from
SD3.5's monolithic prompt across diverse prompt pairs. For each pair we:

  1. Run from the same x_T: Prompt A, Prompt B, CLIP AND, SuperDIFF AND
  2. Record full trajectories and project jointly via PCA
  3. Measure cosine similarity between CLIP AND and SuperDIFF AND final latents
  4. Produce a summary figure with per-pair trajectory panels and a bar chart

This answers: "Do semantic and logical composition converge to the same
region of latent space, or does SuperDIFF AND reach a genuinely different
point?"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import FlowMatchEulerDiscreteScheduler
from notebooks.utils import get_sd3_models, get_image
from notebooks.composition_experiments import (
    LatentTrajectoryCollector,
    get_vel_sd3,
    _get_sd3_conditioning,
    sample_sd3_with_trajectory_tracking,
)
from notebooks.dynamics import get_latents
from scripts.trajectory_dynamics_experiment import superdiff_fm_ode_sd3


# ── Prompt pairs for ablation ────────────────────────────────────────────────

DEFAULT_PROMPT_PAIRS = [
    ("a dog", "a cat"),
    ("a flamingo", "a candy cane"),
    ("a cat on the left", "a dog on the right"),
    ("a portrait", "oil painting style"),
    ("a red car", "a snowy mountain road"),
    ("fireworks", "a dandelion"),
]

# Per-condition visual style
COND_STYLE = {
    "prompt_a":  {"cmap": "Reds",    "color": "#e63946"},
    "prompt_b":  {"cmap": "Blues",    "color": "#457b9d"},
    "clip_and":  {"cmap": "Greens",  "color": "#2a9d8f"},
    "superdiff": {"cmap": "Oranges", "color": "#e9c46a"},
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _trunc(text: str, n: int = 22) -> str:
    return text if len(text) <= n else text[: n - 1] + "\u2026"


def collect_pair(
    prompt_a: str,
    prompt_b: str,
    models: dict,
    seed: int = 42,
    num_steps: int = 50,
    guidance_scale: float = 4.5,
    model_id: str = "stabilityai/stable-diffusion-3.5-medium",
) -> Tuple[Dict[str, LatentTrajectoryCollector], Dict[str, torch.Tensor]]:
    """Run 4 conditions for one prompt pair from a shared x_T."""

    device = torch.device("cuda")
    dtype = torch.float16

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )
    x_T = get_latents(
        scheduler, z_channels=16, device=device, dtype=dtype,
        num_inference_steps=num_steps, batch_size=1,
        latent_width=128, latent_height=128, seed=seed,
    )

    common = dict(
        scheduler=scheduler,
        transformer=models["transformer"],
        tokenizer=models["tokenizer"],
        text_encoder=models["text_encoder"],
        tokenizer_2=models["tokenizer_2"],
        text_encoder_2=models["text_encoder_2"],
        tokenizer_3=models["tokenizer_3"],
        text_encoder_3=models["text_encoder_3"],
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        batch_size=1, device=device, dtype=dtype,
    )

    trackers, finals = {}, {}

    for name, fn_args in [
        ("prompt_a", {"prompt": prompt_a}),
        ("prompt_b", {"prompt": prompt_b}),
        ("clip_and", {"prompt": f"{prompt_a} and {prompt_b}"}),
    ]:
        lat, trk = sample_sd3_with_trajectory_tracking(
            x_T.clone(), fn_args["prompt"], **common
        )
        trackers[name] = trk
        finals[name] = lat

    # SuperDIFF AND (FM-ODE)
    lat_sd, trk_sd, _, _, _ = superdiff_fm_ode_sd3(
        x_T.clone(), prompt_a, prompt_b, **common, lift=0.0,
    )
    trackers["superdiff"] = trk_sd
    finals["superdiff"] = lat_sd

    torch.cuda.empty_cache()
    return trackers, finals


def compute_metrics(finals: Dict[str, torch.Tensor]) -> Dict[str, float]:
    flat = {k: v.flatten().float() for k, v in finals.items()}
    cos = lambda a, b: F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    l2 = lambda a, b: (a - b).norm().item()
    return {
        "cos_clip_vs_sd": cos(flat["clip_and"], flat["superdiff"]),
        "cos_clip_vs_a": cos(flat["clip_and"], flat["prompt_a"]),
        "cos_clip_vs_b": cos(flat["clip_and"], flat["prompt_b"]),
        "cos_sd_vs_a": cos(flat["superdiff"], flat["prompt_a"]),
        "cos_sd_vs_b": cos(flat["superdiff"], flat["prompt_b"]),
        "l2_clip_vs_sd": l2(flat["clip_and"], flat["superdiff"]),
        "l2_a_vs_b": l2(flat["prompt_a"], flat["prompt_b"]),
    }


def project_pair(trackers):
    all_flat, names = [], list(trackers.keys())
    n_steps = None
    for name in names:
        traj = trackers[name].trajectories
        T1 = traj.shape[0]
        n_steps = T1
        all_flat.append(traj[:, 0].reshape(T1, -1).numpy().astype(np.float32))
    stacked = np.vstack(all_flat)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(stacked)
    result, off = {}, 0
    for name in names:
        result[name] = proj[off:off + n_steps]
        off += n_steps
    return result, pca, n_steps


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_ablation_grid(all_results, prompt_pairs, output_path):
    """
    Summary figure: trajectory panels + cosine similarity bar chart.
    Labels are truncated to prevent overflow.
    """
    n = len(prompt_pairs)
    n_cols = min(n, 3)
    n_rows_traj = (n + n_cols - 1) // n_cols
    n_rows = n_rows_traj + 1

    fig = plt.figure(figsize=(5.5 * n_cols, 4.2 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.42, wspace=0.32)

    # ── Trajectory panels ─────────────────────────────────────────────
    for idx, (pair, res) in enumerate(zip(prompt_pairs, all_results)):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        proj = res["projected"]
        ns = res["n_steps"]
        norm = Normalize(vmin=0, vmax=ns - 1)

        for cond in ["prompt_a", "prompt_b", "clip_and", "superdiff"]:
            pts = proj[cond]
            sty = COND_STYLE[cond]
            cmap_obj = cm.get_cmap(sty["cmap"])
            points = pts.reshape(-1, 1, 2)
            segs = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap_obj, norm=norm,
                                linewidths=2.0, alpha=0.85)
            lc.set_array(np.arange(len(segs)))
            ax.add_collection(lc)
            ax.plot(pts[-1, 0], pts[-1, 1], "o", color=sty["color"],
                    markersize=5, markeredgecolor="k", markeredgewidth=0.4, zorder=5)

        # Shared origin
        origin = proj["prompt_a"][0]
        ax.plot(origin[0], origin[1], "ko", markersize=6, zorder=6)

        cos_val = res["metrics"]["cos_clip_vs_sd"]
        ax.set_title(
            f'"{_trunc(pair[0], 16)}" \u2227 "{_trunc(pair[1], 16)}"\n'
            f'cos(CLIP, SD) = {cos_val:.3f}',
            fontsize=8.5, fontweight="bold",
        )
        ax.autoscale()
        ev = res["pca_var"]
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=7.5)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.2)

    # ── Legend in empty cell or floating ──────────────────────────────
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COND_STYLE["prompt_a"]["color"], lw=2.5, label="Prompt A"),
        Line2D([0], [0], color=COND_STYLE["prompt_b"]["color"], lw=2.5, label="Prompt B"),
        Line2D([0], [0], color=COND_STYLE["clip_and"]["color"], lw=2.5, label="CLIP AND"),
        Line2D([0], [0], color=COND_STYLE["superdiff"]["color"], lw=2.5, label="SuperDIFF AND"),
        Line2D([0], [0], marker="o", color="k", lw=0, markersize=5, label=r"$x_T$"),
    ]
    remaining = n_cols * n_rows_traj - n
    if remaining > 0:
        ax_leg = fig.add_subplot(gs[n_rows_traj - 1, n_cols - 1])
        ax_leg.axis("off")
        ax_leg.legend(handles=handles, loc="center", fontsize=9, frameon=True)
    else:
        fig.legend(handles=handles, loc="upper right", fontsize=7.5,
                   bbox_to_anchor=(0.99, 0.99))

    # ── Bar chart ────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[n_rows - 1, :])
    pair_labels = [
        f'{_trunc(a, 12)}\n\u2227 {_trunc(b, 12)}' for a, b in prompt_pairs
    ]
    cos_vals = [r["metrics"]["cos_clip_vs_sd"] for r in all_results]
    x = np.arange(n)
    bars = ax_bar.bar(x, cos_vals, color="#264653", alpha=0.85, width=0.55)
    for bar, v in zip(bars, cos_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(pair_labels, fontsize=7.5)
    ax_bar.set_ylabel("Cosine Similarity", fontsize=10)
    ax_bar.set_title("Final Latent Similarity: CLIP AND vs SuperDIFF AND",
                     fontsize=11, fontweight="bold")
    ax_bar.set_ylim(0, min(1.05, max(cos_vals) + 0.08))
    ax_bar.axhline(1.0, color="gray", ls="--", alpha=0.4)
    ax_bar.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Ablation: Semantic (CLIP AND) vs Logical (SuperDIFF AND) Composition",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_decoded_grid(all_finals, prompt_pairs, vae, output_path):
    """Decoded images grid: rows = pairs, cols = [A, B, CLIP AND, SuperDIFF AND]."""
    n_pairs = len(prompt_pairs)
    conds = ["prompt_a", "prompt_b", "clip_and", "superdiff"]
    col_headers = ["Prompt A", "Prompt B", "CLIP AND", "SuperDIFF AND"]

    fig, axes = plt.subplots(n_pairs, 4, figsize=(14, 3.5 * n_pairs))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    for r, (pair, finals) in enumerate(zip(prompt_pairs, all_finals)):
        for c, cond in enumerate(conds):
            ax = axes[r, c]
            img = get_image(vae, finals[cond])
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_headers[c], fontsize=10, fontweight="bold")
        # Row label
        axes[r, 0].set_ylabel(
            f'{_trunc(pair[0], 14)}\n\u2227 {_trunc(pair[1], 14)}',
            fontsize=8, rotation=0, labelpad=55, va="center",
        )

    fig.suptitle("Decoded: Semantic vs Logical Composition",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation: Semantic (CLIP AND) vs Logical (SuperDIFF AND)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--model-id", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--pairs", type=str, nargs="+", default=None,
                        help='Custom pairs: --pairs "a dog::a cat" "fire::ice"')
    parser.add_argument("--no-decode", action="store_true",
                        help="Skip image decoding (faster)")
    args = parser.parse_args()

    # Parse prompt pairs
    if args.pairs:
        prompt_pairs = []
        for s in args.pairs:
            parts = s.split("::")
            if len(parts) != 2:
                raise ValueError(f"Pair format is 'A::B', got: {s}")
            prompt_pairs.append((parts[0].strip(), parts[1].strip()))
    else:
        prompt_pairs = DEFAULT_PROMPT_PAIRS

    if not args.output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(
            PROJECT_ROOT / "experiments" / "ablation_semantic_vs_logical" / ts
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  Ablation: Semantic (CLIP AND) vs Logical (SuperDIFF AND)")
    print("=" * 70)
    print(f"  {len(prompt_pairs)} prompt pairs  |  {args.steps} steps  |  "
          f"gs={args.guidance}  |  seed={args.seed}")
    for i, (a, b) in enumerate(prompt_pairs, 1):
        print(f"    {i}. \"{a}\" \u2227 \"{b}\"")
    print(f"  Output: {args.output_dir}\n")

    # Load model once
    print("Loading SD3.5 Medium ...")
    models = get_sd3_models(
        model_id=args.model_id, dtype=torch.float16, device=torch.device("cuda"),
    )
    print("  Loaded.\n")

    all_results, all_finals = [], []

    for i, (pa, pb) in enumerate(prompt_pairs, 1):
        print(f"[{i}/{len(prompt_pairs)}] \"{pa}\" \u2227 \"{pb}\"")
        trackers, finals = collect_pair(
            pa, pb, models, seed=args.seed, num_steps=args.steps,
            guidance_scale=args.guidance, model_id=args.model_id,
        )
        proj, pca, ns = project_pair(trackers)
        met = compute_metrics(finals)
        print(f"  cos(CLIP, SD)={met['cos_clip_vs_sd']:.4f}  "
              f"L2={met['l2_clip_vs_sd']:.1f}  "
              f"cos(CLIP,A)={met['cos_clip_vs_a']:.4f}  "
              f"cos(CLIP,B)={met['cos_clip_vs_b']:.4f}")
        all_results.append({"projected": proj, "n_steps": ns, "metrics": met,
                            "pca_var": pca.explained_variance_ratio_.tolist()})
        all_finals.append(finals)

    # Plots
    print("\nGenerating figures ...")
    plot_ablation_grid(
        all_results, prompt_pairs,
        os.path.join(args.output_dir, "ablation_semantic_vs_logical.png"),
    )
    if not args.no_decode:
        print("Decoding images ...")
        plot_decoded_grid(
            all_finals, prompt_pairs, models["vae"],
            os.path.join(args.output_dir, "ablation_decoded_images.png"),
        )

    # JSON
    summary = {
        "pairs": [{"a": a, "b": b} for a, b in prompt_pairs],
        "metrics": [r["metrics"] for r in all_results],
        "pca_var": [r["pca_var"] for r in all_results],
        "config": {"seed": args.seed, "steps": args.steps, "guidance": args.guidance},
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Summary table
    print("\n" + "=" * 70)
    print(f"  {'Pair':<40} {'cos(CLIP,SD)':>12} {'L2':>10}")
    print("  " + "-" * 62)
    for (a, b), res in zip(prompt_pairs, all_results):
        m = res["metrics"]
        print(f'  "{_trunc(a,15)}" \u2227 "{_trunc(b,15)}"'
              f'{"":<{40-len(_trunc(a,15))-len(_trunc(b,15))-8}}'
              f'{m["cos_clip_vs_sd"]:>12.4f}{m["l2_clip_vs_sd"]:>10.1f}')
    avg = np.mean([r["metrics"]["cos_clip_vs_sd"] for r in all_results])
    print("  " + "-" * 62)
    print(f"  {'AVERAGE':<40} {avg:>12.4f}")
    print("=" * 70)
    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    main()
