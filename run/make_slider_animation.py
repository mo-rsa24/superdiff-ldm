
"""
make_slider_from_compose_sweep.py

Discovers images in:
  compose_sweep_20251015_131459/samples/

with names like:
  final_samples_grid_lambda_0.00.png
  final_samples_grid_lambda_0.10.png
  ...
  final_samples_grid_lambda_1.00.png

Any common image extension works (.png, .jpg, .jpeg, .webp). Files are sorted by the numeric lambda.

Outputs (written to the same "samples" folder by default):
  - slider_animation.html  (interactive Plotly slider)
  - slider_frames.pdf      (multi-page PDF flipbook)
  - slider_animation.gif   (optional; requires imageio)

Usage:
  python make_slider_from_compose_sweep.py
  # optional: python make_slider_from_compose_sweep.py --dir path/to/your/samples
  # optional: python make_slider_from_compose_sweep.py --no-gif
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# --- Imports ---
try:
    import plotly.graph_objects as go
except Exception:
    print("Plotly is required. Install via: pip install plotly", file=sys.stderr)
    raise

try:
    from PIL import Image
except Exception:
    print("Pillow is required. Install via: pip install pillow", file=sys.stderr)
    raise

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:
    print("Matplotlib is required for PDF export. Install via: pip install matplotlib", file=sys.stderr)
    raise

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


LAMBDA_RE = re.compile(r"final_samples_grid_lambda_([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def find_images(samples_dir: Path) -> List[Tuple[float, Path]]:
    """Return list of (lambda_value, path) sorted by lambda ascending."""
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    cand = []
    for p in samples_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            m = LAMBDA_RE.search(p.name)
            if m:
                lam = float(m.group(1))
                cand.append((lam, p))

    if not cand:
        raise FileNotFoundError(
            f"No matching images in {samples_dir}. Expected files like 'final_samples_grid_lambda_0.10.png'."
        )

    cand.sort(key=lambda t: t[0])
    return cand


def load_images(paths: List[Path]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGBA")
        imgs.append(im)
    return imgs


def make_plotly_slider(imgs: List[Image.Image], lambdas: List[float], html_out: Path):
    import base64
    from io import BytesIO

    encoded = []
    for im in imgs:
        buf = BytesIO()
        im.save(buf, format="PNG")
        encoded.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8"))

    fig = go.Figure()
    # initial image
    fig.add_layout_image(
        dict(source=encoded[0], xref="x", yref="y", x=0, y=0, sizex=1, sizey=1, sizing="stretch", layer="below")
    )

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1], scaleanchor="x", scaleratio=1)

    frames = []
    for i, src in enumerate(encoded):
        frames.append(
            go.Frame(
                name=f"frame{i}",
                layout=dict(
                    images=[dict(source=src, xref="x", yref="y", x=0, y=0, sizex=1, sizey=1, sizing="stretch", layer="below")]
                )
            )
        )
    fig.frames = frames

    steps = []
    for i, lam in enumerate(lambdas):
        steps.append(
            dict(
                method="animate",
                args=[[f"frame{i}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                label=f"{lam:.2f}",
            )
        )

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "λ = "},
        pad={"t": 40},
        steps=steps
    )]

    updatemenus = [dict(
        type="buttons",
        showactive=False,
        y=1.15,
        x=1.0,
        xanchor="right",
        yanchor="top",
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=250, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))])
        ]
    )]

    fig.update_layout(
        width=900, height=700, margin=dict(l=0, r=0, t=50, b=0),
        sliders=sliders, updatemenus=updatemenus,
        title="SuperDiff Samples — Slider over λ"
    )

    fig.write_html(str(html_out), include_plotlyjs="cdn")
    print(f"[OK] Interactive HTML written to: {html_out}")


def export_pdf_flipbook(imgs: List[Image.Image], lambdas: List[float], pdf_out: Path):
    with PdfPages(str(pdf_out)) as pdf:
        for im, lam in zip(imgs, lambdas):
            w, h = im.size
            aspect = w / h if h != 0 else 1.0
            fig_h = 8
            fig_w = fig_h * aspect

            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = plt.axes([0, 0, 1, 1])
            ax.axis("off")
            ax.imshow(im)
            # small header strip with lambda
            fig.text(0.01, 0.99, f"λ = {lam:.2f}", ha="left", va="top", fontsize=10)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
    print(f"[OK] Multi-page PDF written to: {pdf_out}")


def export_gif(imgs: List[Image.Image], gif_out: Path, duration_ms: int = 250):
    if not HAS_IMAGEIO:
        print("[Skip] imageio not installed; skipping GIF export. Install via: pip install imageio")
        return
    frames = [im.convert("P", palette=Image.ADAPTIVE) for im in imgs]
    import imageio.v2 as imageio
    imageio.mimsave(str(gif_out), frames, duration=duration_ms / 1000.0)
    print(f"[OK] Animated GIF written to: {gif_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="compose_sweep_20251015_131459/samples",
                        help="Directory containing final_samples_grid_lambda_*.{png,jpg,jpeg,webp}")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF export")
    args = parser.parse_args()

    samples_dir = Path(args.dir)
    pairs = find_images(samples_dir)
    lambdas = [lam for lam, _ in pairs]
    paths = [p for _, p in pairs]

    imgs = load_images(paths)

    # Outputs next to images
    html_out = samples_dir / "slider_animation.html"
    pdf_out  = samples_dir / "slider_frames.pdf"
    gif_out  = samples_dir / "slider_animation.gif"

    make_plotly_slider(imgs, lambdas, html_out)
    export_pdf_flipbook(imgs, lambdas, pdf_out)
    if not args.no_gif:
        export_gif(imgs, gif_out)


if __name__ == "__main__":
    main()
