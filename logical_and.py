import os
import json
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import flax
import optax
import tensorflow as tf
from tqdm import tqdm
from typing import Any
from flax.training.train_state import TrainState
from flax.core.frozen_dict import freeze
from torchvision.utils import save_image, make_grid
import torch

# Component Imports
from run.ldm import load_autoencoder_
from models.cxr_unet import ScoreNet
from diffusion.vp_equation import (
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    alpha_bar_fn,
    get_kappa  # From your provided vp_equation.py logic
)


class TrainStateWithEMA(TrainState):
    ema_params: Any = None


def load_ldm_checkpoint(run_dir, ckpt_name, seed):
    meta_path = os.path.join(run_dir, "ldm_meta.json")
    ckpt_path = os.path.join(run_dir, "ckpts", ckpt_name)

    with open(meta_path, 'r') as f:
        config = json.load(f)

    with open(config['ae_config_path'], 'r') as f:
        ae_cfg = json.load(f)

    # Reconstruct architecture parameters
    if isinstance(config['ldm_ch_mults'], str):
        ldm_chans = tuple(config['ldm_base_ch'] * int(m) for m in config['ldm_ch_mults'].split(','))
    else:
        ldm_chans = tuple(config['ldm_base_ch'] * int(m) for m in config['ldm_ch_mults'])

    attn_res = tuple(int(r) for r in str(config['ldm_attn_res']).split(','))
    n_down = len(ae_cfg['ch_mults'].split(',')) - 1 if isinstance(ae_cfg['ch_mults'], str) else len(
        ae_cfg['ch_mults']) - 1
    latent_size = config['img_size'] // (2 ** n_down)
    z_channels = ae_cfg['z_channels']

    model = ScoreNet(
        z_channels=z_channels,
        channels=ldm_chans,
        num_res_blocks=config['ldm_num_res_blocks'],
        attn_resolutions=attn_res,
        dtype=jnp.float32
    )

    # Load parameters from msgpack
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()
    raw_state = flax.serialization.msgpack_restore(blob)

    # Use EMA if available and configured
    params = freeze(raw_state.get('ema_params') or raw_state.get('params'))
    return model, params, config, latent_size, z_channels


def superdiff_and_sampler(
        rng, model1, params1, model2, params2, ae_model, ae_params,
        latent_size, z_channels, batch_size, n_steps=500, z_std=1.0
):
    """
    Implements Proposition 6: Density Control for Logical AND.
    Solves linear system to find weights kappa that balance model densities.
    """
    rng, step_rng = jax.random.split(rng)
    x_shape = (batch_size, latent_size, latent_size, z_channels)
    x = jax.random.normal(step_rng, x_shape) * marginal_prob_std_fn(jnp.array([1.0]))[0]

    timesteps = jnp.linspace(1.0, 1e-5, n_steps + 1)

    for i in tqdm(range(n_steps), desc="SuperDiff AND Sampling"):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_now
        vec_t = jnp.ones(batch_size) * t_now

        # 1. Get raw scores (eps_theta) from both experts
        eps1 = model1.apply({'params': params1}, x, vec_t)
        eps2 = model2.apply({'params': params2}, x, vec_t)

        # 2. Convert to SDE scores: s = -eps / std
        std_t = marginal_prob_std_fn(vec_t)[:, None, None, None]
        s1 = -eps1 / std_t
        s2 = -eps2 / std_t

        # 3. Density Control (Proposition 6)
        # Assuming divergence is unavailable or constant for VP-SDE,
        # we balance relative change. In a simplified unconditional AND:
        # We find kappa such that we stay on the equal density locus.

        # This borrows the 'get_kappa' logic from your vp_equation.py
        # For unconditional AND, we assume div1=div2=0 (balanced priors)
        kappa = get_kappa(vec_t, (jnp.zeros(batch_size), jnp.zeros(batch_size)), (s1, s2))
        kappa = jnp.clip(kappa, 0.0, 1.0)[:, None, None, None]

        # 4. Composite Score
        s_and = kappa * s1 + (1.0 - kappa) * s2
        eps_and = -s_and * std_t

        # 5. Ancestral Step (DDPM style update)
        g_t = diffusion_coeff_fn(vec_t)[:, None, None, None]
        # Simplified Euler-Maruyama step for latent transition
        noise = jax.random.normal(jax.random.fold_in(rng, i), x.shape)

        # Drift update using composite expert direction
        drift = -0.5 * (g_t ** 2) * s_and * jnp.abs(dt)
        diffusion = g_t * jnp.sqrt(jnp.abs(dt)) * noise
        x = x + drift + diffusion

    # Final Decode
    z_for_decode = x * z_std
    x_hat = ae_model.apply({'params': ae_params}, z_for_decode, method=ae_model.decode, train=False)
    x_hat = jnp.clip(x_hat, 0., 1.)
    x_hat = jnp.transpose(x_hat, (0, 3, 1, 2))
    grid = make_grid(torch.from_numpy(np.asarray(x_hat)), nrow=int(np.sqrt(batch_size)))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir_1", type=str, required=True, help="Normal LDM Directory")
    parser.add_argument("--run_dir_2", type=str, required=True, help="TB LDM Directory")
    parser.add_argument("--ckpt_name", type=str, default="last.flax")
    parser.add_argument("--output_name", type=str, default="superdiff_and.png")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    # Load Expert 1 (Normal)
    m1, p1, cfg1, sz, ch = load_ldm_checkpoint(args.run_dir_1, args.ckpt_name, args.seed)
    # Load Expert 2 (TB)
    m2, p2, cfg2, _, _ = load_ldm_checkpoint(args.run_dir_2, args.ckpt_name, args.seed)

    # Load AE (Shared between runs)
    ae_model, ae_params = load_autoencoder_(cfg1['ae_config_path'], cfg1['ae_ckpt_path'])

    # Output Setup
    out_dir = os.path.join(args.run_dir_1, "final_samples")
    os.makedirs(out_dir, exist_ok=True)

    # Sample
    rng = jax.random.PRNGKey(args.seed)
    z_std = 1.0 / cfg1.get('latent_scale_factor', 1.0)

    grid = superdiff_and_sampler(
        rng, m1, p1, m2, p2, ae_model, ae_params,
        sz, ch, args.batch_size, n_steps=args.steps, z_std=z_std
    )

    save_path = os.path.join(out_dir, args.output_name)
    save_image(grid, save_path)
    print(f"SuperDiff AND samples saved to: {save_path}")


if __name__ == "__main__":
    main()