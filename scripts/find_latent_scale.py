import sys
import os
import argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import torch
import tensorflow as tf
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# --- ADDED: Imports for visualization ---
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import plotly.express as px
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ChestXRay import ChestXrayDataset
from run.ldm import load_autoencoder


# --- ADDED: Visualization and helper functions from diagnose_ae.py ---
def _save_reconstruction_grid(ae_model, ae_params, loader, output_dir, n_samples=16):
    """Encodes and decodes a batch of images and saves the comparison."""
    x_real, _ = next(iter(loader))
    x_real = x_real[:n_samples]

    x_np = x_real.numpy().transpose(0, 2, 3, 1)
    x_jax = jnp.asarray((x_np + 1.0) / 2.0)

    posterior = ae_model.apply({'params': ae_params}, x_jax, method=ae_model.encode, train=False)
    z = posterior.mode()
    x_rec_jax = ae_model.apply({'params': ae_params}, z, method=ae_model.decode, train=False)
    x_rec_torch = torch.from_numpy(np.asarray(x_rec_jax)).permute(0, 3, 1, 2).clamp(0, 1)

    # Convert original images from [-1, 1] to [0, 1] for visualization
    x_real_norm = (x_real + 1.0) / 2.0

    comparison_grid = torch.cat([x_real_norm, x_rec_torch])
    grid_img = make_grid(comparison_grid, nrow=x_real.shape[0])
    save_image(grid_img, output_dir / "reconstruction_comparison.png")
    print(f"✅ Saved reconstruction grid to {output_dir}")


def _save_latent_histograms(latents, output_dir):
    """Saves histograms of the latent space distribution."""
    if not _HAS_PLOTLY:
        print("⚠️ Plotly not installed. Skipping histogram generation.")
        return

    fig = px.histogram(
        x=latents.flatten(),
        nbins=100,
        title="Latent Space Distribution (z)",
        labels={"x": "Latent Value"}
    )
    fig.update_layout(bargap=0.1)
    fig.write_image(output_dir / "latent_distribution.png")
    print(f"✅ Saved latent distribution plot to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Find latent scale factor and generate diagnostic visualizations.")
    parser.add_argument("--ae_config_path", type=str, required=True, help="Path to the AE run_meta.json config.")
    parser.add_argument("--ae_ckpt_path", type=str, required=True, help="Path to the AE checkpoint (.flax).")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root of the dataset.")
    parser.add_argument("--task", type=str, required=True, help="Dataset task (e.g., TB).")
    parser.add_argument("--img_size", type=int, default=256, help="Image size.")
    parser.add_argument("--class_filter", type=int, default=None, help="Class to filter for.")
    parser.add_argument("--num_samples_for_scale", type=int, default=2259,
                        help="Number of images to use for scale calculation.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    args = parser.parse_args()

    # Create output directory based on the checkpoint path
    run_dir = Path(args.ae_ckpt_path).parent.parent
    output_dir = run_dir / "diagnostics"
    output_dir.mkdir(exist_ok=True)

    print("--- Loading Autoencoder ---")
    ae_model, ae_params = load_autoencoder(args.ae_config_path, args.ae_ckpt_path)

    print("\n--- Loading Dataset ---")
    dataset = ChestXrayDataset(root_dir=args.data_root, task=args.task, split='train', img_size=args.img_size,
                               class_filter=args.class_filter)

    # Use a subset for faster processing
    num_samples = min(len(dataset), args.num_samples_for_scale)
    ds_subset = Subset(dataset, range(num_samples))
    loader = DataLoader(ds_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Dataset loaded. Using {num_samples} images for diagnostics.")

    # --- ADDED: Generate reconstruction grid ---
    print("\n--- Generating Reconstruction Grid ---")
    _save_reconstruction_grid(ae_model, ae_params, loader, output_dir, n_samples=8)

    print("\n--- Encoding Images to Calculate Scale Factor ---")
    all_latents = []
    for batch in tqdm(loader, desc="Encoding batches"):
        x_torch, _ = batch
        x_np = x_torch.numpy().transpose(0, 2, 3, 1)
        x_jax = jnp.asarray((x_np + 1.0) / 2.0)

        posterior = ae_model.apply({'params': ae_params}, x_jax, method=ae_model.encode, train=False)
        z = posterior.mode()
        all_latents.append(np.asarray(z))

    full_latents = np.concatenate(all_latents, axis=0)
    unscaled_std = full_latents.std()
    scale_factor = 1.0 / unscaled_std

    print("\n--- Results ---")
    print(f"Unscaled Latent Standard Deviation: {unscaled_std:.6f}")
    print(f"Calculated Scale Factor (1.0 / std): {scale_factor:.6f}")

    # Save the scale factor to a text file
    output_path = run_dir / "latent_scale_factor.txt"
    with open(output_path, 'w') as f:
        f.write(str(scale_factor))
    print(f"✅ Saved scale factor to {output_path}")

    # --- ADDED: Generate latent histograms ---
    print("\n--- Generating Latent Space Visualizations ---")
    _save_latent_histograms(full_latents, output_dir)
    print("\nDiagnostic script finished.")


if __name__ == "__main__":
    main()