
"""
make_slider_animation.py

Usage:
  python make_slider_animation.py

What it does:
- Loads 10 images (update `image_paths` to your own).
- Creates an interactive Plotly slider that toggles frames. Saves as: slider_animation.html
- Builds a multi-page PDF (one frame per page) as a PDF-friendly flipbook: slider_frames.pdf
- (Optional) Exports an animated GIF: slider_animation.gif

Notes:
- True interactive sliders do NOT work in PDFs. Use the HTML for interactivity.
- The PDF is a static, multi-page "flipbook" you can page through.
"""

from pathlib import Path
from typing import List
import sys

# ---- CONFIG: Point these to your 10 images ----
# Replace these placeholders with your actual file paths.
image_paths: List[str] = [
    "frame_01.png",
    "frame_02.png",
    "frame_03.png",
    "frame_04.png",
    "frame_05.png",
    "frame_06.png",
    "frame_07.png",
    "frame_08.png",
    "frame_09.png",
    "frame_10.png",
]

# Output files
HTML_OUT = "slider_animation.html"
PDF_OUT = "slider_frames.pdf"
GIF_OUT = "slider_animation.gif"  # optional (requires imageio and pillow)

# --- Imports (done after config so error messages are clearer) ---
try:
    import plotly.graph_objects as go
except Exception as e:
    print("Plotly is required for the HTML slider. Install via: pip install plotly", file=sys.stderr)
    raise

try:
    from PIL import Image
except Exception as e:
    print("Pillow is required for image handling. Install via: pip install pillow", file=sys.stderr)
    raise

# Matplotlib only for PDF export
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception as e:
    print("Matplotlib is required for PDF export. Install via: pip install matplotlib", file=sys.stderr)
    raise

# Optional GIF export
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


def load_images(paths: List[str]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img = Image.open(p).convert("RGBA")
        imgs.append(img)
    return imgs


def make_plotly_slider(imgs: List[Image.Image], html_out: str):
    # We will use images as layout images, and toggle visibility with frames + slider
    # Convert PIL images to base64 for Plotly
    import base64
    from io import BytesIO

    encoded = []
    for im in imgs:
        buf = BytesIO()
        im.save(buf, format="PNG")
        encoded.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8"))

    # Figure layout
    fig = go.Figure()

    # Add first image as initial
    fig.add_layout_image(
        dict(
            source=encoded[0],
            xref="x",
            yref="y",
            x=0, y=0,
            sizex=1, sizey=1,
            sizing="stretch",
            layer="below"
        )
    )

    # Axes off; use [0,1] coords for a simple canvas
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1], scaleanchor="x", scaleratio=1)

    # Build frames
    frames = []
    for i, src in enumerate(encoded):
        frames.append(
            go.Frame(
                name=f"frame{i}",
                layout=dict(
                    images=[
                        dict(
                            source=src,
                            xref="x",
                            yref="y",
                            x=0, y=0,
                            sizex=1, sizey=1,
                            sizing="stretch",
                            layer="below"
                        )
                    ]
                )
            )
        )
    fig.frames = frames

    # Slider to select frame
    steps = []
    for i in range(len(frames)):
        steps.append(
            dict(
                method="animate",
                args=[[f"frame{i}"],
                      dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                label=str(i+1),
            )
        )

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frame: "},
        pad={"t": 40},
        steps=steps
    )]

    # Play/Pause buttons
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            y=1.15,
            x=1.0,
            xanchor="right",
            yanchor="top",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True,
                                     transition=dict(duration=0))]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))]
                )
            ]
        )
    ]

    fig.update_layout(
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        sliders=sliders,
        updatemenus=updatemenus,
        title="Slider Animation (HTML)"
    )

    fig.write_html(html_out, include_plotlyjs="cdn")
    print(f"[OK] Interactive HTML written to: {html_out}")


def export_pdf_flipbook(imgs: List[Image.Image], pdf_out: str):
    # Create a multipage PDF; each page shows one image filling the page
    with PdfPages(pdf_out) as pdf:
        for i, im in enumerate(imgs):
            # Create a figure sized to the image's aspect ratio
            w, h = im.size
            aspect = w / h if h != 0 else 1.0
            # Set figure size in inches (8 inches tall as a baseline)
            fig_h = 8
            fig_w = fig_h * aspect

            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = plt.axes([0, 0, 1, 1])
            ax.axis("off")
            ax.imshow(im)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
    print(f"[OK] Multi-page PDF written to: {pdf_out}")


def export_gif(imgs: List[Image.Image], gif_out: str, duration_ms: int = 200):
    if not HAS_IMAGEIO:
        print("[Skip] imageio not installed; skipping GIF export. Install via: pip install imageio")
        return
    frames = [im.convert("P", palette=Image.ADAPTIVE) for im in imgs]
    imageio.mimsave(gif_out, frames, duration=duration_ms / 1000.0)
    print(f"[OK] Animated GIF written to: {gif_out}")


def main():
    imgs = load_images(image_paths)

    # 1) Interactive HTML with slider
    make_plotly_slider(imgs, HTML_OUT)

    # 2) PDF "flipbook" (one page per frame)
    export_pdf_flipbook(imgs, PDF_OUT)

    # 3) Optional GIF
    export_gif(imgs, GIF_OUT)


if __name__ == "__main__":
    main()
