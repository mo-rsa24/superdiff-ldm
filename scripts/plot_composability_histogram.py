"""
Plot terminal-distance histograms for the composability gap analysis.

Reads per_seed_distances.json produced by measure_composability_gap.py and
generates two Plotly figures:

  1. Pooled violin + strip  — all pairs combined, 3 conditions side by side.
     Points are coloured by concept pair; violin shows the KDE + box.

  2. Faceted violin          — one column per concept pair.
     Conditions are coloured; useful to inspect pair-level heterogeneity.

Usage
-----
conda run -n superdiff python scripts/plot_composability_histogram.py \\
    [--data-dir experiments/inversion/gap_analysis] \\
    [--output-dir experiments/inversion/gap_analysis] \\
    [--no-facet]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("plotly is required:  pip install plotly")


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# One colour per condition (Tableau 10 subset)
COND_LINE = {
    "d_T_mono": "#E15759",
    "d_T_c1":   "#4E79A7",
    "d_T_c2":   "#59A14F",
}
COND_FILL = {
    "d_T_mono": "rgba(225, 87, 89,  0.13)",
    "d_T_c1":   "rgba(78,  121, 167, 0.13)",
    "d_T_c2":   "rgba(89,  161, 79,  0.13)",
}
COND_LABEL = {
    "d_T_mono": "Monolithic",
    "d_T_c1":   "Solo c₁",
    "d_T_c2":   "Solo c₂",
}

# One colour per concept pair (four visually distinct hues)
PAIR_PALETTE = ["#F28E2B", "#76B7B2", "#B07AA1", "#9C755F"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(data_dir: Path) -> list:
    path = data_dir / "per_seed_distances.json"
    if not path.exists():
        sys.exit(
            f"File not found: {path}\n"
            "Run measure_composability_gap.py first to generate it."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: pooled violin + strip
# ---------------------------------------------------------------------------

def build_pooled_figure(records: list) -> go.Figure:
    """
    Three violins (one per condition) plotted side by side.

    go.Violin.marker.color does not accept per-point arrays, so the design
    separates concerns:
      • go.Violin  — KDE shape + box + mean line; no strip points.
      • go.Scatter — one trace per concept pair, placed at the matching
                     numeric x-position with reproducible jitter so that
                     pair membership is colour-coded across all three violins.
    """
    df = pd.DataFrame(records)
    pairs     = sorted(df["pair"].unique())
    pair_cmap = {p: PAIR_PALETTE[i % len(PAIR_PALETTE)] for i, p in enumerate(pairs)}

    fig = go.Figure()

    conditions = ["d_T_mono", "d_T_c1", "d_T_c2"]
    # Map each condition to a numeric x-position (0, 1, 2)
    x_pos = {cond: float(i) for i, cond in enumerate(conditions)}

    # ---- Violin outlines (KDE + box, no individual points) ----
    for i, cond in enumerate(conditions):
        fig.add_trace(go.Violin(
            x=[x_pos[cond]] * len(df),
            y=df[cond].values,
            name=COND_LABEL[cond],
            legendgroup="conditions",
            legendgrouptitle_text="Condition" if i == 0 else None,
            box_visible=True,
            meanline_visible=True,
            points=False,
            line_color=COND_LINE[cond],
            fillcolor=COND_FILL[cond],
            hoverinfo="skip",
            showlegend=True,
        ))

    # ---- Scatter strip: one trace per pair, all three conditions combined ----
    rng = np.random.default_rng(42)   # reproducible jitter
    for i, pair in enumerate(pairs):
        sub = df[df["pair"] == pair].reset_index(drop=True)

        x_jitter, y_vals, hover_texts = [], [], []
        for cond in conditions:
            n = len(sub)
            jitter = rng.uniform(-0.07, 0.07, n)
            x_jitter.extend((x_pos[cond] + jitter).tolist())
            y_vals.extend(sub[cond].tolist())
            hover_texts.extend([
                f"<b>{pair}</b><br>{COND_LABEL[cond]}<br>"
                f"Seed: {row.seed}<br>MSE: {row[cond]:.5f}"
                for _, row in sub.iterrows()
            ])

        fig.add_trace(go.Scatter(
            x=x_jitter,
            y=y_vals,
            mode="markers",
            name=pair,
            legendgroup="pairs",
            legendgrouptitle_text="Concept pair" if i == 0 else None,
            marker=dict(
                color=pair_cmap[pair],
                size=9,
                opacity=0.85,
                line=dict(width=1.2, color="white"),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text=(
                "Composability Gap — Terminal Latent Distance from SuperDiff-AND<br>"
                "<sup>Per-element MSE between z<sub>T</sub><sup>condition</sup> "
                "and z<sub>T</sub><sup>AND</sup> (shared noise seed)</sup>"
            ),
            font=dict(size=17),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Generation condition",
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=[COND_LABEL[c] for c in ["d_T_mono", "d_T_c1", "d_T_c2"]],
            tickfont=dict(size=13),
        ),
        yaxis=dict(
            title="Per-element MSE  (z<sub>T</sub><sup>cond</sup> − z<sub>T</sub><sup>AND</sup>)",
            tickformat=".3f",
            gridcolor="#ECECEC",
            zeroline=False,
            tickfont=dict(size=12),
        ),
        template="plotly_white",
        violinmode="group",
        violingap=0.25,
        violingroupgap=0.1,
        legend=dict(
            tracegroupgap=16,
            borderwidth=1,
            bordercolor="#CCCCCC",
            font=dict(size=12),
        ),
        font=dict(family="Arial, sans-serif", size=13),
        margin=dict(l=85, r=220, t=110, b=65),
        width=820,
        height=570,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Annotation: sample count
    n_pairs = len(pairs)
    n_seeds = df["seed"].nunique()
    fig.add_annotation(
        text=f"N = {len(df)} samples ({n_pairs} pairs × {n_seeds} seeds)",
        xref="paper", yref="paper",
        x=0.01, y=-0.11,
        showarrow=False,
        font=dict(size=11, color="#666666"),
        xanchor="left",
    )

    return fig


# ---------------------------------------------------------------------------
# Figure 2: faceted violin (one column per pair)
# ---------------------------------------------------------------------------

def build_faceted_figure(records: list) -> go.Figure:
    """
    One subplot per concept pair.  Within each subplot, three violins
    (one per condition) are shown side by side.  Condition colour is
    consistent across panels; legend shown only once (left-most panel).
    """
    df    = pd.DataFrame(records)
    pairs = sorted(df["pair"].unique())
    n     = len(pairs)

    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[p.replace(" + ", " +<br>") for p in pairs],
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )

    conditions = ["d_T_mono", "d_T_c1", "d_T_c2"]

    for col, pair in enumerate(pairs, start=1):
        sub = df[df["pair"] == pair]
        for cond in conditions:
            first_panel = col == 1
            fig.add_trace(
                go.Violin(
                    x=[COND_LABEL[cond]] * len(sub),
                    y=sub[cond].values,
                    name=COND_LABEL[cond],
                    legendgroup=cond,
                    showlegend=first_panel,
                    box_visible=True,
                    meanline_visible=True,
                    line_color=COND_LINE[cond],
                    fillcolor=COND_FILL[cond],
                    points="all",
                    jitter=0.4,
                    pointpos=0,
                    marker=dict(
                        color=COND_LINE[cond],
                        size=8,
                        opacity=0.8,
                        line=dict(width=1.2, color="white"),
                    ),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Seed : %{x}<br>"
                        "MSE  : %{y:.5f}<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )

    # Shared y-axis label on the first subplot only
    fig.update_yaxes(
        title_text="Per-element MSE",
        tickformat=".3f",
        gridcolor="#ECECEC",
        zeroline=False,
        row=1, col=1,
    )
    for col in range(2, n + 1):
        fig.update_yaxes(gridcolor="#ECECEC", zeroline=False, row=1, col=col)

    fig.update_layout(
        title=dict(
            text="Composability Gap by Concept Pair",
            font=dict(size=17),
            x=0.5,
            xanchor="center",
        ),
        violinmode="group",
        violingap=0.2,
        violingroupgap=0.1,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            title_text="Condition",
            borderwidth=1,
            bordercolor="#CCCCCC",
        ),
        margin=dict(l=80, r=40, t=100, b=60),
        width=300 * n + 100,
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot composability gap histograms")
    p.add_argument("--data-dir",   default="experiments/inversion/gap_analysis",
                   help="Directory containing per_seed_distances.json")
    p.add_argument("--output-dir", default="experiments/inversion/gap_analysis",
                   help="Directory to write output figures")
    p.add_argument("--no-facet",   action="store_true",
                   help="Skip the per-pair faceted figure")
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
        print(f"  PNG export skipped ({e}). Install kaleido:  pip install kaleido")


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading per_seed_distances.json from {data_dir} ...")
    records = load_records(data_dir)

    df      = pd.DataFrame(records)
    n_pairs = df["pair"].nunique()
    n_seeds = df["seed"].nunique()
    print(f"  {len(records)} records  ({n_pairs} pairs × {n_seeds} seeds per pair)")

    # ---- Figure 1: pooled ----
    print("\nBuilding pooled violin figure ...")
    fig_pooled = build_pooled_figure(records)
    _save(fig_pooled, "composability_gap_histogram", out_dir)

    # ---- Figure 2: faceted ----
    if not args.no_facet:
        print("\nBuilding faceted figure ...")
        fig_facet = build_faceted_figure(records)
        _save(fig_facet, "composability_gap_by_pair", out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
