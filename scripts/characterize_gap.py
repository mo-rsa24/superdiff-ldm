"""
Phase 4: Characterize the composability gap taxonomy.

Loads all gap_metrics.json files produced by measure_composability_gap.py,
computes the CLIP geometric relationship between each pair's individual
conditioning vectors, and produces:

  1. Scatter plots: CLIP cosine similarity between (c₁, c₂) vs each gap metric
  2. Taxonomy table (CSV) with semantic relationship type annotations
  3. Trajectory gap curves per pair (step-wise MSE and cosine similarity)
  4. Summary JSON with regression stats

Usage
-----
conda run -n superdiff python scripts/characterize_gap.py \
    [--gap-dir experiments/inversion/gap_analysis] \
    [--output-dir experiments/inversion/gap_analysis] \
    [--model-id stabilityai/stable-diffusion-3.5-medium]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from notebooks.utils import get_sd3_models, get_sd3_text_embedding


# ---------------------------------------------------------------------------
# Semantic relationship type annotations
# Manually assigned based on linguistic/conceptual analysis of each pair.
# ---------------------------------------------------------------------------

PAIR_ANNOTATIONS = {
    ("a cat", "a dog"):          "complementary",   # same-category, semantically close
    ("a person", "an umbrella"): "hierarchical",    # object–accessory relation
    ("a person", "a car"):       "co-occurring",    # common real-world co-occurrence
    ("a car", "a truck"):        "same-category",   # near-synonym, high CLIP similarity
}


def get_semantic_type(c1: str, c2: str) -> str:
    return PAIR_ANNOTATIONS.get((c1, c2), PAIR_ANNOTATIONS.get((c2, c1), "unknown"))


# ---------------------------------------------------------------------------
# CLIP cosine similarity between two individual prompts
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_cosine_between_prompts(
    c1: str,
    c2: str,
    models: dict,
    device: torch.device,
) -> float:
    """
    Compute cosine similarity between pooled_embeds of c₁ and c₂ separately.
    Uses SD3.5's CLIP-L + CLIP-G pooled embeddings (2048-dim).
    """
    _, pooled1 = get_sd3_text_embedding(
        [c1],
        models["tokenizer"],   models["text_encoder"],
        models["tokenizer_2"], models["text_encoder_2"],
        models["tokenizer_3"], models["text_encoder_3"],
        device=device,
    )
    _, pooled2 = get_sd3_text_embedding(
        [c2],
        models["tokenizer"],   models["text_encoder"],
        models["tokenizer_2"], models["text_encoder_2"],
        models["tokenizer_3"], models["text_encoder_3"],
        device=device,
    )
    cos = F.cosine_similarity(pooled1, pooled2, dim=-1).item()
    return cos


# ---------------------------------------------------------------------------
# Load all gap results
# ---------------------------------------------------------------------------

def load_all_gaps(gap_dir: Path) -> list:
    """Load gap_metrics.json from every subdirectory of gap_dir."""
    results = []
    for json_path in sorted(gap_dir.glob("*/gap_metrics.json")):
        with open(json_path) as f:
            results.append(json.load(f))
    return results


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------

def make_scatter_plots(records: list, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping scatter plots")
        return

    metrics_to_plot = [
        ("gap_and_pstar", "clip_cos",      "Image gap: CLIP cosine similarity (AND vs p*)"),
        ("gap_and_pstar", "lat_mse",       "Latent gap: VAE MSE (AND vs p*)"),
        ("gap_and_pstar", "traj_mse_mean", "Trajectory gap: mean step-wise MSE (AND vs p*)"),
        ("gap_and_pstar", "traj_cos_mean", "Trajectory gap: mean step-wise cos sim (AND vs p*)"),
        ("gap_and_mono",  "clip_cos",      "Image gap: CLIP cosine similarity (AND vs monolithic)"),
    ]

    x_vals     = [r["clip_cos_c1c2"]          for r in records]
    pair_labels = [f"{r['pair'][0]}\n+ {r['pair'][1]}" for r in records]
    sem_types   = [r["semantic_type"]          for r in records]

    type_colors = {
        "complementary": "#2196F3",
        "hierarchical":  "#FF9800",
        "co-occurring":  "#4CAF50",
        "same-category": "#E91E63",
        "unknown":       "#9E9E9E",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax_idx, (gap_key, metric_key, title) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        y_vals = [r[gap_key].get(metric_key) for r in records]

        # Skip if metric not available
        if any(v is None for v in y_vals):
            ax.set_visible(False)
            continue

        for i, (x, y, label, stype) in enumerate(
            zip(x_vals, y_vals, pair_labels, sem_types)
        ):
            color = type_colors.get(stype, "#9E9E9E")
            ax.scatter(x, y, color=color, s=120, zorder=3)
            ax.annotate(
                label, (x, y),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7, color=color,
            )

        # Trend line (linear regression)
        xs = [v for v in x_vals if v is not None]
        ys = [y_vals[i] for i, x in enumerate(x_vals) if x is not None]
        if len(xs) >= 2:
            import numpy as np
            m, b = np.polyfit(xs, ys, 1)
            x_line = [min(xs) - 0.02, max(xs) + 0.02]
            ax.plot(x_line, [m * xi + b for xi in x_line],
                    "--", color="grey", linewidth=1, alpha=0.6, zorder=2)
            r2 = float(np.corrcoef(xs, ys)[0, 1] ** 2)
            ax.set_title(f"{title}\n(R²={r2:.3f})", fontsize=9)
        else:
            ax.set_title(title, fontsize=9)

        ax.set_xlabel("CLIP cos sim(c₁, c₂)", fontsize=9)
        ax.set_ylabel(metric_key, fontsize=9)
        ax.grid(True, alpha=0.3)

    # Legend for semantic types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=stype)
        for stype, color in type_colors.items()
        if stype != "unknown"
    ]
    axes[-1].legend(handles=legend_elements, loc="center", fontsize=10,
                    title="Semantic type", title_fontsize=11, frameon=True)
    axes[-1].axis("off")

    plt.tight_layout()
    out_path = out_dir / "scatter_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plots saved: {out_path}")


# ---------------------------------------------------------------------------
# Trajectory curves per pair
# ---------------------------------------------------------------------------

def plot_trajectory_curves(records: list, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    has_per_step = any(
        "traj_mse_per_step" in r.get("gap_and_pstar", {}) for r in records
    )
    if not has_per_step:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for r in records:
        g = r["gap_and_pstar"]
        label = f"{r['pair'][0]} + {r['pair'][1]}"
        if "traj_mse_per_step" in g:
            ax1.plot(g["traj_mse_per_step"], label=label)
        if "traj_cos_per_step" in g:
            ax2.plot(g["traj_cos_per_step"], label=label)

    ax1.set_title("Trajectory MSE (AND vs p*) per step")
    ax1.set_xlabel("Diffusion step")
    ax1.set_ylabel("MSE")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Trajectory cosine sim (AND vs p*) per step")
    ax2.set_xlabel("Diffusion step")
    ax2.set_ylabel("Cosine similarity")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "trajectory_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Trajectory curves saved: {out_path}")


# ---------------------------------------------------------------------------
# Taxonomy table (CSV)
# ---------------------------------------------------------------------------

def write_taxonomy_table(records: list, out_dir: Path):
    import csv

    fieldnames = [
        "pair_c1", "pair_c2", "semantic_type",
        "clip_cos_c1c2",
        "clip_cos_and_pstar", "lpips_and_pstar",
        "lat_mse_and_pstar",
        "traj_mse_and_pstar", "traj_cos_and_pstar",
        "clip_cos_and_mono",
    ]

    rows = []
    for r in records:
        g_ap = r["gap_and_pstar"]
        g_am = r["gap_and_mono"]
        rows.append({
            "pair_c1":           r["pair"][0],
            "pair_c2":           r["pair"][1],
            "semantic_type":     r["semantic_type"],
            "clip_cos_c1c2":     round(r["clip_cos_c1c2"], 4),
            "clip_cos_and_pstar": round(g_ap.get("clip_cos", float("nan")), 4),
            "lpips_and_pstar":   round(g_ap.get("lpips",    float("nan")), 4),
            "lat_mse_and_pstar": round(g_ap.get("lat_mse",  float("nan")), 4),
            "traj_mse_and_pstar": round(g_ap.get("traj_mse_mean", float("nan")), 4),
            "traj_cos_and_pstar": round(g_ap.get("traj_cos_mean", float("nan")), 4),
            "clip_cos_and_mono": round(g_am.get("clip_cos", float("nan")), 4),
        })

    out_path = out_dir / "taxonomy_table.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Taxonomy table saved: {out_path}")
    return rows


# ---------------------------------------------------------------------------
# Regression stats
# ---------------------------------------------------------------------------

def compute_regression_stats(records: list) -> dict:
    """
    For each gap metric, regress against CLIP cosine similarity between c₁ and c₂.
    Returns R² and slope for each metric.
    """
    try:
        import numpy as np
    except ImportError:
        return {}

    x = [r["clip_cos_c1c2"] for r in records]
    metrics = {
        "clip_cos_and_pstar": [r["gap_and_pstar"].get("clip_cos") for r in records],
        "lat_mse_and_pstar":  [r["gap_and_pstar"].get("lat_mse") for r in records],
        "traj_mse_and_pstar": [r["gap_and_pstar"].get("traj_mse_mean") for r in records],
        "traj_cos_and_pstar": [r["gap_and_pstar"].get("traj_cos_mean") for r in records],
        "clip_cos_and_mono":  [r["gap_and_mono"].get("clip_cos") for r in records],
    }

    stats = {}
    for metric_name, y in metrics.items():
        valid = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
        if len(valid) < 2:
            continue
        xs, ys = zip(*valid)
        xs, ys = np.array(xs), np.array(ys)
        m, b  = np.polyfit(xs, ys, 1)
        r2    = float(np.corrcoef(xs, ys)[0, 1] ** 2)
        stats[metric_name] = {"slope": round(float(m), 4), "intercept": round(float(b), 4), "r2": round(r2, 4)}

    return stats


# ---------------------------------------------------------------------------
# Print summary table to console
# ---------------------------------------------------------------------------

def print_summary(records: list, stats: dict):
    print("\n" + "=" * 72)
    print(f"{'Pair':<36} {'Sem. type':<16} {'cos(c₁,c₂)':<12} {'CLIP gap':<10} {'Traj MSE':<10}")
    print("=" * 72)
    for r in records:
        pair_str  = f"{r['pair'][0]} + {r['pair'][1]}"
        sem_type  = r["semantic_type"]
        cos_c1c2  = r["clip_cos_c1c2"]
        clip_gap  = r["gap_and_pstar"].get("clip_cos", float("nan"))
        traj_mse  = r["gap_and_pstar"].get("traj_mse_mean", float("nan"))
        print(f"{pair_str:<36} {sem_type:<16} {cos_c1c2:<12.4f} {clip_gap:<10.4f} {traj_mse:<10.4f}")
    print("=" * 72)

    if stats:
        print("\nRegression: gap metrics vs CLIP cosine similarity(c₁, c₂)")
        print(f"  {'Metric':<28} {'slope':<10} {'R²':<8}")
        print("  " + "-" * 48)
        for metric, s in stats.items():
            print(f"  {metric:<28} {s['slope']:<10.4f} {s['r2']:<8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Characterize composability gap taxonomy")
    p.add_argument("--gap-dir",    default="experiments/inversion/gap_analysis",
                   help="Directory containing per-pair subdirs with gap_metrics.json")
    p.add_argument("--output-dir", default="experiments/inversion/gap_analysis")
    p.add_argument("--model-id",   default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--dtype",      default="float16", choices=["float16", "bfloat16"])
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype   = torch.float16 if args.dtype == "float16" else torch.bfloat16
    gap_dir = Path(args.gap_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load gap results ----
    raw_results = load_all_gaps(gap_dir)
    if not raw_results:
        print(f"No gap_metrics.json files found in {gap_dir}. "
              f"Run measure_composability_gap.py first.")
        return
    print(f"Loaded {len(raw_results)} pair results from {gap_dir}")

    # ---- Load SD3.5 text encoders for CLIP cosine similarity computation ----
    print("Loading SD3.5 text encoders ...")
    models = get_sd3_models(model_id=args.model_id, dtype=dtype, device=device)

    # ---- Enrich each record with CLIP geometry and semantic type ----
    print("Computing CLIP cosine similarity for each pair ...")
    records = []
    for r in raw_results:
        c1, c2 = r["pair"]
        cos_c1c2   = clip_cosine_between_prompts(c1, c2, models, device)
        sem_type   = get_semantic_type(c1, c2)

        enriched = {
            **r,
            "clip_cos_c1c2":  round(cos_c1c2, 4),
            "semantic_type":  sem_type,
        }
        records.append(enriched)
        print(f"  '{c1}' vs '{c2}': cos={cos_c1c2:.4f}  type={sem_type}")

    # Sort by CLIP cosine similarity ascending (widest gap first)
    records.sort(key=lambda r: r["clip_cos_c1c2"])

    # ---- Compute regression stats ----
    stats = compute_regression_stats(records)

    # ---- Save enriched summary ----
    summary = {
        "n_pairs":   len(records),
        "records":   records,
        "regression_stats": stats,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {out_dir / 'summary.json'}")

    # ---- Outputs ----
    print("\nGenerating outputs ...")
    make_scatter_plots(records, out_dir)
    plot_trajectory_curves(records, out_dir)
    rows = write_taxonomy_table(records, out_dir)

    # ---- Console summary ----
    print_summary(records, stats)

    print(f"\nCharacterization complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
