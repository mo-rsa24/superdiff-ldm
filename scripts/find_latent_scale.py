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
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.ChestXRay import ChestXrayDataset
from run.ldm import load_autoencoder


def main():
    parser = argparse.ArgumentParser(description="Find the correct latent scale factor over multiple images.")
    parser.add_argument("--ae_config_path", type=str, required=True, help="Path to the AE run_meta.json config.")
    parser.add_argument("--ae_ckpt_path", type=str, required=True, help="Path to the AE checkpoint (.flax).")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root of the dataset.")
    parser.add_argument("--task", type=str, required=True, help="Dataset task (e.g., TB).")
    parser.add_argument("--img_size", type=int, default=256, help="Image size.")
    parser.add_argument("--class_filter", type=int, default=None, help="Class to filter for.")
    # Use a large default, but you can override it. For 1745 images, using all is best.
    parser.add_argument("--num_samples", type=int, default=1745, help="Number of samples to use for calculation.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    args = parser.parse_args()

    # Prevent TF from grabbing GPU memory
    tf.config.experimental.set_visible_devices([], "GPU")

    print("--- Loading Model and Dataset ---")
    ae_model, ae_params = load_autoencoder(args.ae_config_path, args.ae_ckpt_path)

    ds = ChestXrayDataset(
        root_dir=args.data_root,
        task=args.task,
        split="train",
        img_size=args.img_size,
        class_filter=args.class_filter
    )

    if len(ds) == 0:
        print("Error: Dataset is empty. Check data_root, task, and class_filter.")
        return

    # Use a subset of the dataset for efficiency
    num_to_process = min(args.num_samples, len(ds))
    subset_indices = list(range(num_to_process))
    ds_subset = Subset(ds, subset_indices)
    loader = DataLoader(ds_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Dataset loaded. Using {len(ds_subset)} images for scaling calculation.")

    all_latents = []
    print("\n--- Encoding Images in Batches---")
    for batch in tqdm(loader):
        x_torch, _ = batch
        # Convert to JAX array in [0, 1] range
        x_np = x_torch.numpy().transpose(0, 2, 3, 1)
        x_jax = jnp.asarray((x_np + 1.0) / 2.0)

        posterior = ae_model.apply({'params': ae_params}, x_jax, method=ae_model.encode, train=False)
        z = posterior.mode()
        all_latents.append(np.asarray(z))

    # Concatenate all latent vectors and calculate global std dev
    full_latents = np.concatenate(all_latents, axis=0)
    unscaled_std = full_latents.std()
    scale_factor = 1.0 / unscaled_std

    print("\n--- Results ---")
    print(f"Unscaled Latent Standard Deviation (over {len(ds_subset)} images): {unscaled_std:.6f}")
    print(f"Calculated Scale Factor (1.0 / std): {scale_factor:.6f}")

    scaled_z = full_latents * scale_factor
    print(f"Verification: Scaled Latent Std Dev: {np.std(scaled_z):.6f}")

    print("\n✅ Success! Use the 'Calculated Scale Factor' in your full training script.")


if __name__ == "__main__":
    main()