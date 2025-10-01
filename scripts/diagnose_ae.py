import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from tqdm import tqdm

# Add project root to path to allow importing local modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from datasets.ChestXRay import ChestXrayDataset
from models.ae_kl import AutoencoderKL


def load_autoencoder(config_path, ckpt_path):
    """Loads a trained autoencoder model and its parameters."""
    print(f"Loading AE from config: {config_path}")
    with open(config_path, 'r') as f:
        ae_args = json.load(f)

    ch_mults = tuple(int(c.strip()) for c in ae_args['ch_mults'].split(','))
    enc_cfg = dict(ch_mults=ch_mults, num_res_blocks=ae_args['num_res_blocks'], z_ch=ae_args['z_channels'],
                   double_z=True)
    dec_cfg = dict(ch_mults=ch_mults, num_res_blocks=ae_args['num_res_blocks'], out_ch=1)
    ae_model = AutoencoderKL(enc_cfg=enc_cfg, dec_cfg=dec_cfg, embed_dim=ae_args['z_channels'])

    print(f"Loading AE checkpoint from: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()
    # Create dummy states to restore into
    rng = jax.random.PRNGKey(0)
    fake_img = jnp.ones((1, ae_args['img_size'], ae_args['img_size'], 1))
    ae_variables = ae_model.init({'params': rng, 'dropout': rng}, fake_img, rng=rng)

    # We only need the generator (AE) part for diagnosis
    from flax.training.train_state import TrainState
    import optax

    # --- FIX: Recreate the full optimizer chain from training to ensure states match ---
    def get_tx(lr, grad_clip, weight_decay):
        """Replicates the optimizer creation from the training script."""
        return optax.chain(
            optax.clip_by_global_norm(grad_clip) if grad_clip > 0 else optax.identity(),
            optax.adamw(lr, weight_decay=weight_decay)
        )

    # Use the function to create a structurally identical optimizer
    tx = get_tx(
        lr=ae_args.get('lr', 1e-4),
        grad_clip=ae_args.get('grad_clip', 1.0),
        weight_decay=ae_args.get('weight_decay', 1e-4)
    )

    dummy_gen_state = TrainState.create(apply_fn=None, params={'ae': ae_variables['params']}, tx=tx)
    dummy_disc_state = TrainState.create(apply_fn=None, params={}, tx=tx)

    from flax.serialization import from_bytes
    restored_gen_state, _ = from_bytes((dummy_gen_state, dummy_disc_state), blob)
    ae_params = restored_gen_state.params['ae']

    print("Autoencoder loaded successfully.")
    return ae_model, ae_params, ae_args


def encode(model, params, x):
    return model.apply({'params': params}, x, method=model.encode, train=False)
encode = jax.jit(encode, static_argnums=(0,))


# @jax.jit(static_argnums=(0,))
def decode(model, params, z):
    """Decode a batch of latents into images."""
    return model.apply({'params': params}, z, method=model.decode, train=False)
decode = jax.jit(decode, static_argnums=(0,))

def check_1_reconstruction(ae_model, ae_params, loader, output_dir, n_images=8):
    """CHECK 1: Visual Reconstruction Test."""
    print("\n--- Running Check 1: Visual Reconstruction Test ---")
    batch, _ = next(iter(loader))
    x_orig = jnp.asarray(batch.numpy()).transpose(0, 2, 3, 1)  # to (N, H, W, C)
    x_orig = (x_orig + 1.0) * 0.5  # [-1, 1] -> [0, 1]

    # Use only a subset for the grid
    x_orig = x_orig[:n_images]

    posterior = encode(ae_model, ae_params, x_orig)
    z = posterior.mode()  # Use mode for deterministic reconstruction
    x_rec = decode(ae_model, ae_params, z)

    # Convert back to torch for grid saving
    x_orig_torch = torch.from_numpy(np.asarray(x_orig).transpose(0, 3, 1, 2))
    x_rec_torch = torch.from_numpy(np.asarray(x_rec).transpose(0, 3, 1, 2))

    # Interleave original and reconstructed images for comparison
    comparison_grid = torch.cat([x_orig_torch, x_rec_torch], dim=0)
    # Correctly interleave images: [orig1, rec1, orig2, rec2, ...]
    interleaved = torch.empty((2 * n_images, 1, x_orig.shape[1], x_orig.shape[2]), dtype=x_orig_torch.dtype)
    interleaved[0::2] = x_orig_torch
    interleaved[1::2] = x_rec_torch

    grid_img = make_grid(interleaved, nrow=2, padding=2, normalize=False)
    grid_img_np = (grid_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    output_path = Path(output_dir) / "reconstruction_comparison.png"
    Image.fromarray(grid_img_np).save(output_path)
    print(f"✅ Reconstruction grid saved to: {output_path}")
    print("   (Left column: Original, Right column: Reconstructed)")


def check_2_latent_stats(ae_model, ae_params, loader, n_samples=1024):
    """CHECK 2: Latent Statistics Analysis."""
    print("\n--- Running Check 2: Latent Statistics Analysis ---")
    all_latents = []

    # Limit number of samples for efficiency
    dataset = loader.dataset
    if len(dataset) > n_samples:
        dataset = Subset(dataset, list(range(n_samples)))

    stat_loader = DataLoader(dataset, batch_size=loader.batch_size, shuffle=False, num_workers=4)

    print(f"Encoding {len(dataset)} samples to calculate latent statistics...")
    for batch, _ in tqdm(stat_loader):
        x = jnp.asarray(batch.numpy()).transpose(0, 2, 3, 1)  # to (N, H, W, C)
        x = (x + 1.0) * 0.5  # [-1, 1] -> [0, 1]

        posterior = encode(ae_model, ae_params, x)
        z = posterior.mode()  # Use mode (mean) for stable statistics
        all_latents.append(np.asarray(z))

    full_latents = np.concatenate(all_latents, axis=0)

    mean = np.mean(full_latents)
    std = np.std(full_latents)
    min_val = np.min(full_latents)
    max_val = np.max(full_latents)

    print("\n--- Latent Space Statistics ---")
    print(f"  Mean:     {mean:.6f}")
    print(f"  Std Dev:  {std:.6f}")
    print(f"  Min:      {min_val:.6f}")
    print(f"  Max:      {max_val:.6f}")
    print("---------------------------------")

    # Provide feedback
    if abs(mean) < 0.1 and 0.85 < std < 1.15:
        print("✅ PASSED: Latent distribution appears well-normalized (mean≈0, std≈1).")
    else:
        print("⚠️ FAILED: Latent distribution is NOT well-normalized. The LDM may struggle.")
        print("   Consider adjusting the KL_WEIGHT during AE training.")


def main():
    parser = argparse.ArgumentParser(description="Run diagnostic checks on a trained AutoencoderKL model.")
    parser.add_argument("run_dir", type=str,
                        help="Path to the AE run directory (containing run_meta.json and ckpts/last.flax).")
    parser.add_argument("--data_root", type=str, default="../datasets/cleaned", help="Root directory of the dataset.")
    args = parser.parse_args()

    # Disable GPU usage by TensorFlow, which is often used for data loading.
    tf.config.experimental.set_visible_devices([], "GPU")

    run_path = Path(args.run_dir)
    config_path = run_path / "run_meta.json"
    ckpt_path = run_path / "ckpts" / "last.flax"

    if not all([config_path.exists(), ckpt_path.exists()]):
        print(f"Error: Could not find 'run_meta.json' or 'ckpts/last.flax' in {run_path}")
        sys.exit(1)

    ae_model, ae_params, ae_args = load_autoencoder(config_path, ckpt_path)

    # Setup dataset loader
    ds = ChestXrayDataset(
        root_dir=args.data_root, task=ae_args['task'], split="train",
        img_size=ae_args['img_size'], class_filter=ae_args.get('class_filter')
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    # Run Checks
    output_dir = run_path / "diagnostics"
    output_dir.mkdir(exist_ok=True)

    check_1_reconstruction(ae_model, ae_params, loader, output_dir)
    check_2_latent_stats(ae_model, ae_params, loader)

    print("\n--- Check 3: LDM Overfitting Test (Guidance) ---")
    print("To perform this check, you must now train the LDM on these verified latents.")
    print("1. Update 'ldm_tb_diagnostic_train.sh' with the path to this verified AE run.")
    print(
        "2. IMPORTANT: If your latent space is now normalized (std≈1), you may not need a 'latent_scale_factor' > 1.0 for the LDM.")
    print("   You can start by removing it or setting it to 1.0 in your LDM launch script.")
    print(
        "3. Run the LDM diagnostic script. If the loss decreases and samples are non-black, your pipeline is working.")
    print("✅ Diagnostics complete.")


if __name__ == "__main__":
    main()

