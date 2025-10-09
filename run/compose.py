import os
import sys
import json
import math
import functools
from datetime import datetime
from typing import Tuple, Dict
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf

# --- ensure repo root on sys.path ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.serialization import from_bytes
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import optax

# --- Use the project's actual modules, following ldm.py's structure ---
from diffusion.equations import get_kappa, score_function_hutchinson_estimator, marginal_prob_std_fn, diffusion_coeff_fn
from models.cxr_unet import ScoreNet
from run.ldm import load_autoencoder  # Directly import the trusted loader


def load_ldm(config_path: str, ckpt_path: str) -> Tuple[ScoreNet, Dict]:
    """
    FIXED: Loads a ScoreNet LDM by perfectly replicating the model and optimizer
    setup from ldm.py to prevent deserialization errors.
    """
    print(f"Loading LDM from config: {config_path}")
    with open(config_path, 'r') as f:
        # The training arguments are saved in the 'args' sub-dictionary
        meta = json.load(f)['args']

    # --- Load VAE metadata to get parameters required by the U-Net ---
    vae_config_path = meta['ae_config_path']
    with open(vae_config_path, 'r') as f:
        vae_meta = json.load(f)['args']

    embed_dim = vae_meta['embed_dim']
    z_channels = vae_meta['z_channels']

    # --- Construct U-Net parameters from LDM metadata ---
    ch_mults = tuple(int(c) for c in meta['ldm_ch_mults'].split(',') if c)
    ldm_chans = tuple([meta['ldm_base_ch']] + [meta['ldm_base_ch'] * m for m in ch_mults])
    attn_res = tuple(int(r) for r in meta['ldm_attn_res'].split(',') if r)
    num_res_blocks = meta['ldm_num_res_blocks']

    # --- Instantiate ScoreNet ---
    ldm_model = ScoreNet(
        embed_dim=embed_dim,
        channels=ldm_chans,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_res,
        num_heads=4
    )

    # --- Create a dummy state with a PERFECTLY MATCHING optimizer structure ---
    rng = jax.random.PRNGKey(0)
    latent_size = meta['img_size'] // 8

    fake_latents = jnp.ones((1, latent_size, latent_size, z_channels))
    fake_time = jnp.ones((1,))
    ldm_variables = ldm_model.init({'params': rng, 'dropout': rng}, fake_latents, fake_time)

    # This optimizer definition MUST match the one in ldm.py
    tx = optax.chain(
        optax.clip_by_global_norm(meta.get('grad_clip', 1.0)),
        optax.adamw(meta.get('lr', 3e-5), weight_decay=meta.get('weight_decay', 0.01))
    )
    dummy_state = TrainState.create(apply_fn=ldm_model.apply, params=ldm_variables['params'], tx=tx)

    print(f"Loading LDM checkpoint from: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()

    try:
        restored_state = from_bytes(dummy_state, blob)
    except KeyError as e:
        print("\n--- FLAX DESERIALIZATION ERROR ---")
        print(f"A `KeyError` occurred: {e}")
        print(
            "This almost always means the model architecture defined in the script does not match the one in the saved checkpoint.")
        print("Please check the following in your `run_meta.json` and `cxr_unet.py`:")
        print(f"  - `ldm_num_res_blocks`: {num_res_blocks}")
        print(f"  - `ldm_base_ch`: {meta['ldm_base_ch']}")
        print(f"  - `ldm_ch_mults`: {ch_mults}")
        print("----------------------------------\n")
        raise e

    print("✅ LDM loaded successfully.")
    return ldm_model, restored_state.params


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Compose two pretrained LDM models using Superdiffusion.")
    parser.add_argument("--run_tb", type=str, required=True, help="Path to the TB model run directory.")
    parser.add_argument("--run_normal", type=str, required=True, help="Path to the Normal model run directory.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples to generate.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of reverse diffusion steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="composed_output",
                        help="Directory to save generated samples.")
    args = parser.parse_args()

    # --- Setup ---
    key = jax.random.PRNGKey(args.seed)
    output_dir = os.path.join(args.output_dir, f"compose_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    print(f"🚀 Starting composition run. Outputs will be saved to: {output_dir}")

    # --- Load Models using the robust, consistent loaders ---
    tb_config_path = os.path.join(args.run_tb, "run_meta.json")
    tb_ckpt_path = os.path.join(args.run_tb, "ckpts/last.flax")
    normal_config_path = os.path.join(args.run_normal, "run_meta.json")
    normal_ckpt_path = os.path.join(args.run_normal, "ckpts/last.flax")

    ldm_tb_model, ldm_tb_params = load_ldm(tb_config_path, tb_ckpt_path)
    ldm_normal_model, ldm_normal_params = load_ldm(normal_config_path, normal_ckpt_path)

    state_tb = TrainState.create(apply_fn=ldm_tb_model.apply, params=ldm_tb_params, tx=optax.identity())
    state_normal = TrainState.create(apply_fn=ldm_normal_model.apply, params=ldm_normal_params, tx=optax.identity())

    with open(tb_config_path, 'r') as f:
        tb_meta = json.load(f)['args']
    ae_config_path = tb_meta['ae_config_path']
    ae_ckpt_path = tb_meta['ae_ckpt_path']
    vae_def, vae_params = load_autoencoder(ae_config_path, ae_ckpt_path)
    latent_scale_factor = tb_meta['latent_scale_factor']

    @jax.jit
    def vae_decode_fn(params, latents):
        return vae_def.apply({'params': params}, latents, method=vae_def.decode)

    # --- Prepare for Composition Sampling ---
    latent_size = tb_meta['img_size'] // 8
    z_channels = vae_def.enc_cfg['z_ch']
    sample_shape = (args.batch_size, latent_size, latent_size, z_channels)

    @functools.partial(jax.jit, static_argnums=(3,))
    def composed_score_fn(x, t, key, batch_size):
        eps_tb = state_tb.apply_fn({'params': state_tb.params}, x, t)
        eps_normal = state_normal.apply_fn({'params': state_normal.params}, x, t)
        sigma_t = marginal_prob_std_fn(t)[:, None, None, None]
        score_tb = -eps_tb / sigma_t
        score_normal = -eps_normal / sigma_t
        key, div_key_tb, div_key_normal = jax.random.split(key, 3)
        dlog_p_tb = score_function_hutchinson_estimator(x, t, state_tb.apply_fn, state_tb.params, div_key_tb)[0]
        dlog_p_normal = \
        score_function_hutchinson_estimator(x, t, state_normal.apply_fn, state_normal.params, div_key_normal)[0]
        kappa = get_kappa(t, (dlog_p_tb, dlog_p_normal), (score_tb, score_normal))
        kappa = jnp.clip(kappa, 0.0, 1.0)
        composed_score = (1 - kappa) * score_normal + kappa * score_tb
        diagnostics = {
            'score_norm_tb': jnp.linalg.norm(score_tb.reshape(batch_size, -1), axis=-1),
            'score_norm_normal': jnp.linalg.norm(score_normal.reshape(batch_size, -1), axis=-1),
            'score_norm_composed': jnp.linalg.norm(composed_score.reshape(batch_size, -1), axis=-1),
            'dlog_tb': dlog_p_tb, 'dlog_normal': dlog_p_normal, 'kappa': kappa,
        }
        return composed_score, diagnostics

    # --- Run Sampling ---
    print(f"Starting Euler-Maruyama sampler for {args.steps} steps...")
    key, sample_key = jax.random.split(key)
    time_steps = jnp.linspace(1., 1e-3, args.steps)
    dt = time_steps[0] - time_steps[1]
    x = jax.random.normal(sample_key, sample_shape) * marginal_prob_std_fn(jnp.ones((args.batch_size,)))[
        :, None, None, None]
    diagnostics_history = []
    for t in tqdm(time_steps, desc="Sampling"):
        key, step_key, noise_key = jax.random.split(key, 3)
        t_batch = jnp.ones(args.batch_size) * t
        score, diagnostics = composed_score_fn(x, t_batch, step_key, args.batch_size)
        diagnostics_history.append(jax.device_get(diagnostics))
        g_t = diffusion_coeff_fn(t_batch)[:, None, None, None]
        noise = jax.random.normal(noise_key, x.shape)
        x = x + (g_t ** 2 * score * dt) + g_t * jnp.sqrt(dt) * noise
    print("✅ Sampling complete.")

    # --- Decode and Save ---
    decoded_samples = vae_decode_fn(vae_params, x / latent_scale_factor)
    decoded_samples = jnp.clip(decoded_samples, 0.0, 1.0)
    samples_torch = torch.tensor(np.asarray(decoded_samples).transpose(0, 3, 1, 2))
    grid_path = os.path.join(samples_dir, "final_samples_grid.png")
    save_image(samples_torch, grid_path, nrow=int(math.sqrt(args.batch_size)))
    print(f"Saved final sample grid to {grid_path}")

    # --- Diagnostics ---
    if diagnostics_history:
        print("📊 Processing and saving diagnostics...")
        diag_agg = {k: np.array([d[k] for d in diagnostics_history]) for k in diagnostics_history[0]}
        fig, axs = plt.subplots(1, 4, figsize=(24, 5))
        steps_axis = np.arange(len(diagnostics_history))
        axs[0].plot(steps_axis, diag_agg['score_norm_tb'].mean(axis=-1), label="TB")
        axs[0].plot(steps_axis, diag_agg['score_norm_normal'].mean(axis=-1), label="Normal")
        axs[0].plot(steps_axis, diag_agg['score_norm_composed'].mean(axis=-1), label="Composed", linestyle='--',
                    alpha=0.8)
        axs[0].set_title("Score Norms")
        axs[0].legend()
        axs[0].set_xlabel("Step")
        axs[1].plot(steps_axis, diag_agg['dlog_tb'].mean(axis=-1), label="TB")
        axs[1].plot(steps_axis, diag_agg['dlog_normal'].mean(axis=-1), label="Normal")
        axs[1].set_title("Divergence (dlog p)")
        axs[1].legend()
        axs[1].set_xlabel("Step")
        axs[2].plot(steps_axis, diag_agg['kappa'].mean(axis=-1))
        axs[2].set_title("Kappa (mean over batch)")
        axs[2].set_xlabel("Step")
        axs[2].set_ylim(0, 1)
        axs[3].hist(diag_agg['kappa'].flatten(), bins=50, range=(0, 1))
        axs[3].set_title("Kappa Distribution")
        diag_plot_path = os.path.join(output_dir, "diagnostics_plot.png")
        plt.tight_layout()
        plt.savefig(diag_plot_path)
        plt.close()
        print(f"Saved diagnostics plot to {diag_plot_path}")


if __name__ == "__main__":
    main()