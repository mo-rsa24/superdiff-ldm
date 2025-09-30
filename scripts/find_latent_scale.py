import sys
import os
import json
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# Add project root to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.ChestXRay import ChestXrayDataset
from run.ldm import load_autoencoder  # Re-use the AE loading logic


def main(config_path, ckpt_path, data_root, task, n_samples=1024):
    """Calculates the standard deviation of the AE's latent space."""
    tf.config.experimental.set_visible_devices([], "GPU")  # Keep TF off GPU

    print("Loading Autoencoder...")
    ae_model, ae_params = load_autoencoder(config_path, ckpt_path)

    print("Loading Dataset...")
    with open(config_path, 'r') as f:
        ae_args = json.load(f)
    ds = ChestXrayDataset(root_dir=data_root, task=task, split="train", img_size=ae_args['img_size'],
                          class_filter=ae_args.get('class_filter'))
    # Use a subset for efficiency
    ds = Subset(ds, list(range(min(n_samples, len(ds)))))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    @jax.jit
    def encode_batch(x):
        posterior = ae_model.apply({'params': ae_params}, x, method=ae_model.encode, train=False)
        # We only care about the mean for calculating variance, sampling adds noise
        return posterior.mode()

    all_latents = []
    print(f"Encoding {len(ds)} samples to find latent distribution stats...")
    for batch in tqdm(loader):
        x, _ = batch
        x = jnp.asarray(x.numpy()).transpose(0, 2, 3, 1)
        x = (x + 1.0) / 2.0  # Assuming [-1, 1] -> [0, 1] normalization as in ldm.py
        latents = encode_batch(x)
        all_latents.append(np.asarray(latents))

    # Calculate global standard deviation
    full_latents = np.concatenate(all_latents, axis=0)
    std_dev = full_latents.std()

    # The scale factor is the reciprocal of the standard deviation
    scale_factor = 1.0 / std_dev

    print("\n--- Latent Space Statistics ---")
    print(f"  Mean: {full_latents.mean():.6f}")
    print(f"  Std Dev: {std_dev:.6f}")
    print(f"  Min: {full_latents.min():.6f}")
    print(f"  Max: {full_latents.max():.6f}")
    print("\n---------------------------------")
    print(f"✅ Recommended latent_scale_factor: {scale_factor:.8f}")
    print("---------------------------------")


if __name__ == "__main__":
    # Use the paths from your failed run
    AE_CONFIG = "runs/ae_full_tb_b4_20250924/20250924-041825/run_meta.json"
    AE_CKPT = "runs/ae_full_tb_b4_20250924/20250924-041825/ckpts/last.flax"
    DATA_ROOT = "../datasets/cleaned"
    TASK = "TB"
    main(AE_CONFIG, AE_CKPT, DATA_ROOT, TASK)