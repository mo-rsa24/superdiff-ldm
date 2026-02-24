"""
scripts/plot_gap_clean.py
=========================
Clean scaffolding for composability-gap visualizations.

What is being measured
----------------------
For every (seed, concept-pair) combination we run FOUR diffusion chains
from the *same* initial noise z_0:

    1. SuperDiff-AND(c₁, c₂)   → z_T^AND    ← anchor / reference
    2. SD3.5("c₁ and c₂")      → z_T^mono
    3. SD3.5(c₁)               → z_T^c1
    4. SD3.5(c₂)               → z_T^c2

Distance metric (per-element MSE):
    d = mean over ALL elements of the latent tensor of (z_cond − z_AND)²

Intuition:
    d ≈ 0.00  →  nearly identical to AND  (same image)
    d ≈ 0.25  →  subtle differences
    d ≈ 0.60  →  noticeable difference in composition or content
    d ≈ 1.50  →  substantially different image
    d ≈ 2.00  →  theoretical ceiling: two completely unrelated
                 latents drawn from a unit-Gaussian prior

Figures
-------
  terminal (default)
      Strip chart: every (seed × concept-pair) run as one coloured dot.
      Bold horizontal tick = pooled mean across all seeds and pairs.
      Thin bar = ±1 SD.
      Dots coloured by concept pair so you can see pair-level
      heterogeneity alongside the pooled signal.

  --facet
      Same data, one subplot per concept pair.
      Useful when pair-level differences matter for the story.

  --temporal   [STUB — fill in after terminal is approved]
      Mean distance-from-AND as a function of denoising step.
      One line per condition + shaded 95 % CI band.
      Source: trajectory_distances.json.

Usage
-----
conda run -n superdiff python scripts/plot_gap_clean.py
conda run -n superdiff python scripts/plot_gap_clean.py --facet
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas required:  pip install pandas")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("plotly required:  pip install plotly")


# ─── Shared style constants ────────────────────────────────────────────────────

# Keys as they appear in per_seed_distances.json
TERM_KEYS = ["d_T_mono", "d_T_c1", "d_T_c2"]

# Two-line labels for the X-axis ticks
TERM_LABEL = {
    "d_T_mono": "Monolithic<br><sup>SD3.5(c₁ and c₂)</sup>",
    "d_T_c1":   "Solo c₁<br><sup>SD3.5(c₁ only)</sup>",
    "d_T_c2":   "Solo c₂<br><sup>SD3.5(c₂ only)</sup>",
}

# Short labels used inside subplots
TERM_SHORT = {
    "d_T_mono": "Monolithic",
    "d_T_c1":   "Solo c₁",
    "d_T_c2":   "Solo c₂",
}

# One bold colour per condition
TERM_COLOR = {
    "d_T_mono": "#C0392B",   # crimson
    "d_T_c1":   "#2471A3",   # sapphire
    "d_T_c2":   "#1E8449",   # forest green
}

# Keys as they appear in trajectory_distances.json (lowercase t)
TRAJ_KEYS = ["d_t_mono", "d_t_c1", "d_t_c2"]
TRAJ_LABEL = {
    "d_t_mono": "Monolithic  SD3.5(c₁ and c₂)",
    "d_t_c1":   "Solo c₁  SD3.5(c₁ only)",
    "d_t_c2":   "Solo c₂  SD3.5(c₂ only)",
}
TRAJ_COLOR = {
    "d_t_mono": "#C0392B",
    "d_t_c1":   "#2471A3",
    "d_t_c2":   "#1E8449",
}

# Four perceptually distinct hues for the 4 concept pairs
PAIR_PALETTE = ["#E67E22", "#8E44AD", "#1ABC9C", "#E74C3C"]

# Y-axis label shared across terminal figures
Y_LABEL = (
    "Distance from SuperDiff-AND  (per-element MSE)<br>"
    "<sup>lower = more similar to AND's output</sup>"
)


# ─── Data loaders ──────────────────────────────────────────────────────────────

def load_terminal(data_dir: Path) -> list:
    """Load per_seed_distances.json (64 records = 4 pairs × 16 seeds)."""
    path = data_dir / "per_seed_distances.json"
    if not path.exists():
        sys.exit(
            f"File not found: {path}\n"
            "Run  measure_composability_gap.py  first to generate it."
        )
    with open(path) as f:
        return json.load(f)


def load_trajectory(data_dir: Path) -> list:
    """Load trajectory_distances.json (n_pairs × n_seeds × n_steps records)."""
    path = data_dir / "trajectory_distances.json"
    if not path.exists():
        sys.exit(
            f"File not found: {path}\n"
            "Run  measure_composability_gap.py  first to generate it."
        )
    with open(path) as f:
        return json.load(f)


# ─── Figure 1 — terminal strip chart ──────────────────────────────────────────

def build_terminal_figure(records: list) -> go.Figure:
    """
    One dot per (seed × concept-pair) run per condition.

    What a single dot represents:
        "For seed s, concept pair (c₁, c₂):
         the [condition] diffusion chain — which started from the *same*
         initial noise as AND — ended at a latent that is [dot height] MSE
         units away from AND's terminal latent."

    Three conditions on the X-axis:
        Monolithic  →  SD3.5("c₁ and c₂")  anchored against AND
        Solo c₁     →  SD3.5(c₁ only)       anchored against AND
        Solo c₂     →  SD3.5(c₂ only)       anchored against AND
    """
    df    = pd.DataFrame(records)
    pairs = sorted(df["pair"].unique())
    pair_color = {p: PAIR_PALETTE[i % len(PAIR_PALETTE)] for i, p in enumerate(pairs)}

    fig = go.Figure()
    x_pos = {c: float(i) for i, c in enumerate(TERM_KEYS)}
    rng   = np.random.default_rng(42)

    # ── Individual dots (one per seed × pair) ─────────────────────────────────
    first_pair = True
    for pair in pairs:
        sub   = df[df["pair"] == pair].reset_index(drop=True)
        color = pair_color[pair]

        xs, ys, texts = [], [], []
        for cond in TERM_KEYS:
            n      = len(sub)
            jitter = rng.uniform(-0.13, 0.13, n)
            xs.extend((x_pos[cond] + jitter).tolist())
            ys.extend(sub[cond].tolist())
            texts.extend(
                f"<b>{pair}</b><br>"
                f"Seed {row['seed']}<br>"
                f"Distance from AND: {row[cond]:.4f}"
                for _, row in sub.iterrows()
            )

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name=pair,
            legendgroup="pairs",
            legendgrouptitle_text="Concept pair" if first_pair else None,
            marker=dict(
                color=color, size=9, opacity=0.70,
                line=dict(width=1.2, color="white"),
            ),
            text=texts,
            hovertemplate="%{text}<extra></extra>",
        ))
        first_pair = False

    # ── Mean ± 1 SD shown as cross-hair shapes ────────────────────────────────
    # (Plotly shapes are not in the legend, but mean annotations label them.)
    for cond in TERM_KEYS:
        vals  = df[cond].values
        mean_ = float(vals.mean())
        std_  = float(vals.std(ddof=1))
        x     = x_pos[cond]
        col   = TERM_COLOR[cond]

        lo = max(0.0, mean_ - std_)
        hi = mean_ + std_

        # Vertical SD bar
        fig.add_shape(type="line",
            x0=x, x1=x, y0=lo, y1=hi,
            line=dict(color=col, width=2))

        # SD cap ticks
        for y_cap in (lo, hi):
            fig.add_shape(type="line",
                x0=x - 0.09, x1=x + 0.09, y0=y_cap, y1=y_cap,
                line=dict(color=col, width=2))

        # Bold horizontal mean line
        fig.add_shape(type="line",
            x0=x - 0.24, x1=x + 0.24, y0=mean_, y1=mean_,
            line=dict(color=col, width=5))

        # Mean value annotation (right of the cross-hair)
        fig.add_annotation(
            x=x, y=mean_,
            text=f"<b>μ = {mean_:.3f}</b>",
            showarrow=False,
            xshift=40,
            xanchor="left",
            font=dict(size=11, color=col),
        )

    # ── Reference line: theoretical ceiling for unrelated latents ─────────────
    # For a unit-Gaussian latent prior, E[(z_a − z_b)²] = 2·Var = 2.0.
    # Real SD3.5 latents differ, but 2.0 gives a useful upper-bound reference.
    fig.add_hline(
        y=2.0,
        line=dict(color="#BBBBBB", width=1.2, dash="dot"),
        annotation_text="~ ceiling: two unrelated latents",
        annotation_position="top right",
        annotation_font=dict(size=9, color="#AAAAAA"),
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    y_max = max(df[k].max() for k in TERM_KEYS) * 1.08

    fig.update_layout(
        title=dict(
            text=(
                "Terminal Latent Distance from SuperDiff-AND<br>"
                "<sup>Each dot = one run (seed × concept pair).  "
                "Bold tick = mean.  Bar = ±1 SD.  "
                "Lower means the method ends up closer to AND.</sup>"
            ),
            font=dict(size=17),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Generation condition",
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=[TERM_LABEL[c] for c in TERM_KEYS],
            tickfont=dict(size=13),
            range=[-0.6, 2.6],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=Y_LABEL,
            tickformat=".2f",
            gridcolor="#EBEBEB",
            zeroline=True,
            zerolinecolor="#CCCCCC",
            range=[0, max(y_max, 2.15)],
        ),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            tracegroupgap=8,
            borderwidth=1,
            bordercolor="#CCCCCC",
            font=dict(size=12),
        ),
        margin=dict(l=90, r=250, t=125, b=100),
        width=860,
        height=580,
    )

    # ── Scale-interpretation annotation ───────────────────────────────────────
    fig.add_annotation(
        text=(
            "<b>Scale:</b>  0 = identical to AND  │  "
            "≈ 0.25 subtle differences  │  "
            "≈ 0.60 noticeable difference  │  "
            "≈ 2.0 unrelated latents (ceiling)"
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10, color="#777777"),
        xanchor="center",
    )

    # ── Dataset summary ────────────────────────────────────────────────────────
    n_pairs = df["pair"].nunique()
    n_seeds = df["seed"].nunique()
    fig.add_annotation(
        text=f"N = {len(df)} runs  ({n_pairs} concept pairs × {n_seeds} seeds per pair)",
        xref="paper", yref="paper",
        x=0.02, y=-0.11,
        showarrow=False,
        font=dict(size=10, color="#999999"),
        xanchor="left",
    )

    return fig


# ─── Figure 2 — terminal, faceted by concept pair ─────────────────────────────

def build_pair_figure(records: list) -> go.Figure:
    """
    Same data as build_terminal_figure() but one subplot per concept pair.
    Reveals pair-level heterogeneity: which pairs have larger gaps?
    """
    df    = pd.DataFrame(records)
    pairs = sorted(df["pair"].unique())
    n     = len(pairs)
    rng   = np.random.default_rng(42)

    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=pairs,
        shared_yaxes=True,
        horizontal_spacing=0.035,
    )

    legend_done: set[str] = set()

    for col, pair in enumerate(pairs, start=1):
        sub = df[df["pair"] == pair].reset_index(drop=True)

        for ci, cond in enumerate(TERM_KEYS):
            vals   = sub[cond].values
            jitter = rng.uniform(-0.16, 0.16, len(vals))
            color  = TERM_COLOR[cond]
            show   = cond not in legend_done
            legend_done.add(cond)

            # Dots
            fig.add_trace(
                go.Scatter(
                    x=ci + jitter,
                    y=vals,
                    mode="markers",
                    name=TERM_SHORT[cond],
                    legendgroup=cond,
                    showlegend=show,
                    marker=dict(
                        color=color, size=9, opacity=0.72,
                        line=dict(width=1.2, color="white"),
                    ),
                    hovertemplate=(
                        f"<b>{pair}</b><br>"
                        f"{TERM_SHORT[cond]}<br>"
                        "MSE: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )

            # Mean tick
            mean_ = float(vals.mean())
            fig.add_shape(
                type="line",
                x0=ci - 0.25, x1=ci + 0.25, y0=mean_, y1=mean_,
                line=dict(color=color, width=4),
                row=1, col=col,
            )

        fig.update_xaxes(
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=["Mono", "Solo c₁", "Solo c₂"],
            tickfont=dict(size=10),
            showgrid=False,
            zeroline=False,
            row=1, col=col,
        )

    fig.update_yaxes(
        title_text="Distance from AND  (per-element MSE)",
        tickformat=".2f",
        gridcolor="#EBEBEB",
        zeroline=True, zerolinecolor="#CCCCCC",
        rangemode="tozero",
        row=1, col=1,
    )
    for col in range(2, n + 1):
        fig.update_yaxes(
            gridcolor="#EBEBEB", zeroline=True, zerolinecolor="#CCCCCC",
            rangemode="tozero", row=1, col=col,
        )

    fig.update_layout(
        title=dict(
            text=(
                "Terminal Latent Distance from AND — per Concept Pair<br>"
                "<sup>Bold tick = mean across seeds. "
                "Each dot = one seed.</sup>"
            ),
            font=dict(size=16), x=0.5, xanchor="center",
        ),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            title_text="Condition",
            borderwidth=1, bordercolor="#CCCCCC",
        ),
        margin=dict(l=85, r=50, t=110, b=60),
        width=285 * n + 130,
        height=490,
    )

    return fig


# ─── Figure 3 — temporal line chart [STUB] ────────────────────────────────────

def build_temporal_figure(records: list, ci_method: str = "sem") -> go.Figure:
    """
    Mean distance-from-AND vs. denoising step (0 → T).

    One coloured line per condition.
    Shaded band = ±1 SEM across seeds (or bootstrap 95 % CI if ci_method="boot").

    Parameters
    ----------
    records    : list of dicts from trajectory_distances.json
    ci_method  : "sem"  → ±1 SEM across seeds
                 "std"  → ±1 SD
                 "boot" → bootstrap 95 % CI  [TODO]

    Reading:
        At step t, the Monolithic line height shows the mean (across all seeds
        and concept pairs) per-element MSE between z_t^mono and z_t^AND.
        The band shows uncertainty across seeds.
    """
    df = pd.DataFrame(records)
    df = df[df["step"] > 0].copy()   # drop step 0 (trivially 0, shared noise)

    fig = go.Figure()

    for cond in TRAJ_KEYS:
        # Group by step: mean and SEM across seeds × pairs
        grp = (
            df.groupby("step")[cond]
            .agg(mean_="mean", std_="std", n_="count")
            .reset_index()
        )
        grp["sem_"] = grp["std_"] / np.sqrt(grp["n_"])

        if ci_method == "std":
            hi = grp["mean_"] + grp["std_"]
            lo = (grp["mean_"] - grp["std_"]).clip(lower=0)
        else:  # sem
            hi = grp["mean_"] + grp["sem_"]
            lo = (grp["mean_"] - grp["sem_"]).clip(lower=0)

        color = TRAJ_COLOR[cond]

        # Shaded CI band
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=np.concatenate([grp["step"], grp["step"][::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.12)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=grp["step"],
            y=grp["mean_"],
            mode="lines",
            name=TRAJ_LABEL[cond],
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>{TRAJ_LABEL[cond]}</b><br>"
                "Step %{x}<br>"
                "Mean MSE: %{y:.4f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=(
                "Distance from SuperDiff-AND Over Denoising Steps<br>"
                "<sup>Mean per-element MSE across all seeds and concept pairs.  "
                f"Band = ±1 {'SEM' if ci_method == 'sem' else 'SD'}.</sup>"
            ),
            font=dict(size=17),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Denoising step  (0 = pure noise → T = clean image)",
            gridcolor="#EBEBEB",
        ),
        yaxis=dict(
            title=Y_LABEL,
            tickformat=".3f",
            gridcolor="#EBEBEB",
            zeroline=True,
            zerolinecolor="#CCCCCC",
            rangemode="tozero",
        ),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(borderwidth=1, bordercolor="#CCCCCC"),
        margin=dict(l=90, r=40, t=120, b=80),
        width=900,
        height=480,
    )

    # Annotation: step 0 excluded
    fig.add_annotation(
        text="Step 0 excluded: all conditions share the same initial noise (distance = 0)",
        xref="paper", yref="paper",
        x=0.5, y=-0.13,
        showarrow=False,
        font=dict(size=10, color="#888888"),
        xanchor="center",
    )

    return fig


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Clean composability-gap visualization scaffolding"
    )
    p.add_argument(
        "--data-dir", default="experiments/inversion/gap_analysis",
        help="Directory containing per_seed_distances.json (and trajectory_distances.json)",
    )
    p.add_argument(
        "--output-dir", default="experiments/inversion/gap_analysis",
        help="Directory to write output figures",
    )
    p.add_argument(
        "--facet", action="store_true",
        help="Also build per-pair faceted figure (terminal only)",
    )
    p.add_argument(
        "--temporal", action="store_true",
        help="Build temporal line chart instead (reads trajectory_distances.json)",
    )
    p.add_argument(
        "--ci", default="sem", choices=["sem", "std"],
        help="CI band for temporal chart: sem (default) or std",
    )
    return p.parse_args()


def _save(fig: go.Figure, stem: str, out_dir: Path) -> None:
    html_path = out_dir / f"{stem}.html"
    fig.write_html(str(html_path))
    print(f"  Saved {html_path}")

    png_path = out_dir / f"{stem}.png"
    try:
        fig.write_image(str(png_path), scale=2)
        print(f"  Saved {png_path}")
    except Exception as e:
        print(f"  PNG skipped ({type(e).__name__}: {e})")


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.temporal:
        print(f"Loading trajectory_distances.json from {data_dir} ...")
        records = load_trajectory(data_dir)
        df_info = pd.DataFrame(records)
        print(
            f"  {len(records):,} records — "
            f"{df_info['pair'].nunique()} pairs × "
            f"{df_info['seed'].nunique()} seeds × "
            f"{df_info['step'].nunique()} steps"
        )
        print(f"\nBuilding temporal line chart (CI = {args.ci}) ...")
        fig = build_temporal_figure(records, ci_method=args.ci)
        _save(fig, "temporal_gap_line", out_dir)
    else:
        print(f"Loading per_seed_distances.json from {data_dir} ...")
        records = load_terminal(data_dir)
        df_info = pd.DataFrame(records)
        print(
            f"  {len(records)} records — "
            f"{df_info['pair'].nunique()} pairs × "
            f"{df_info['seed'].nunique()} seeds per pair"
        )

        print("\nBuilding terminal strip chart ...")
        fig = build_terminal_figure(records)
        _save(fig, "terminal_gap", out_dir)

        if args.facet:
            print("\nBuilding per-pair faceted figure ...")
            fig_f = build_pair_figure(records)
            _save(fig_f, "terminal_gap_by_pair", out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
