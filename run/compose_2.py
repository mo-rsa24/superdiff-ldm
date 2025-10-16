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
from typing import Any
from flax.training.train_state import TrainState

from diffusion.vp_equation import (
    marginal_prob_std_fn, score_function_hutchinson_estimator, diffusion_coeff_fn, get_kappa,
    gram_and_rhs_from_scores, solve_kappa_and, sum_except_batch
)



class TrainStateWithEMA(TrainState):
    ema_params: Any = None
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

from models.cxr_unet import ScoreNet
from run.ldm import load_autoencoder  # Directly import the trusted loader


def load_ldm(config_path: str, ckpt_path: str) -> Tuple[ScoreNet, Dict]:
    print(f"Loading LDM from config: {config_path}")
    with open(config_path, 'r') as f:
        loaded_json = json.load(f)
        meta = loaded_json.get('args', loaded_json)

    # --- Load VAE metadata ---
    vae_config_path = meta['ae_config_path']
    with open(vae_config_path, 'r') as f:
        vae_loaded_json = json.load(f)
        vae_meta = vae_loaded_json.get('args', vae_loaded_json)
    z_channels = vae_meta['z_channels']

    # --- Construct U-Net parameters ---
    ldm_chans = tuple(meta['ldm_base_ch'] * int(m) for m in meta['ldm_ch_mults'].split(','))
    attn_res = tuple(int(r) for r in meta['ldm_attn_res'].split(',') if r)
    num_res_blocks = meta['ldm_num_res_blocks']

    # --- Instantiate ScoreNet ---
    ldm_model = ScoreNet(
        z_channels=z_channels,
        channels=ldm_chans,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_res,
    )
    rng = jax.random.PRNGKey(0)
    latent_size = meta['img_size'] // 4
    fake_latents = jnp.ones((1, latent_size, latent_size, z_channels))
    fake_time = jnp.ones((1,))
    ldm_variables = ldm_model.init({'params': rng, 'dropout': rng}, fake_latents, fake_time)
    tx = optax.chain(
        optax.clip_by_global_norm(meta.get('grad_clip', 1.0)),
        optax.adamw(meta.get('lr', 3e-5), weight_decay=meta.get('weight_decay', 0.01))
    )
    dummy_state = TrainStateWithEMA.create(
        apply_fn=ldm_model.apply,
        params=ldm_variables['params'],
        ema_params=ldm_variables['params'],
        tx=tx
    )

    print(f"Loading LDM checkpoint from: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()
    restored_state = from_bytes(dummy_state, blob)
    print("✅ LDM loaded successfully.")
    use_ema = meta.get('use_ema', False)
    if use_ema and hasattr(restored_state, 'ema_params') and restored_state.ema_params is not None:
        print("INFO: Using EMA parameters for composition.")
        params_to_return = restored_state.ema_params
    else:
        print("INFO: Using standard model parameters for composition.")
        params_to_return = restored_state.params

    return ldm_model, params_to_return


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Compose two pretrained LDM models using Superdiffusion.")
    parser.add_argument("--run_tb", type=str, required=True, help="Path to the TB model run directory.")
    parser.add_argument("--run_normal", type=str, required=True, help="Path to the Normal model run directory.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of samples to generate.")
    parser.add_argument("--steps", type=int, default=700, help="Number of reverse diffusion steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="composed_output",
                        help="Directory to save generated samples.")
    parser.add_argument("--mode", type=str, default="analytic",
                        choices=["analytic", "and", "or", "poe"],
                        help="SuperDiff mode: current analytic kappa, AND (Prop.6), OR (Prop.3), or PoE.")
    parser.add_argument("--or_T", type=float, default=1.0, help="Softmax temperature T for OR.")
    parser.add_argument("--or_bias_l", type=float, default=0.0, help="Bias ℓ for OR.")

    args = parser.parse_args()

    # --- Setup ---
    key = jax.random.PRNGKey(args.seed)
    output_dir = os.path.join(args.output_dir, f"compose_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    print(f"🚀 Starting composition run. Outputs will be saved to: {output_dir}")

    # --- Load Models using the robust, consistent loaders ---
    tb_config_path = os.path.join(args.run_tb, "ldm_meta.json")
    tb_ckpt_path = os.path.join(args.run_tb, "ckpts/last.flax")
    normal_config_path = os.path.join(args.run_normal, "ldm_meta.json")
    normal_ckpt_path = os.path.join(args.run_normal, "ckpts/last.flax")

    ldm_tb_model, ldm_tb_params = load_ldm(tb_config_path, tb_ckpt_path)
    ldm_normal_model, ldm_normal_params = load_ldm(normal_config_path, normal_ckpt_path)

    state_tb = TrainState.create(apply_fn=ldm_tb_model.apply, params=ldm_tb_params, tx=optax.identity())
    state_normal = TrainState.create(apply_fn=ldm_normal_model.apply, params=ldm_normal_params, tx=optax.identity())

    with open(tb_config_path, 'r') as f:
        tb_loaded_json = json.load(f)
        tb_meta = tb_loaded_json.get('args', tb_loaded_json)

    ae_config_path = tb_meta['ae_config_path']
    ae_ckpt_path = tb_meta['ae_ckpt_path']
    vae_def, vae_params = load_autoencoder(ae_config_path, ae_ckpt_path)
    latent_scale_factor = tb_meta['latent_scale_factor']

    @jax.jit
    def vae_decode_fn(params, latents):
        return vae_def.apply({'params': params}, latents, method=vae_def.decode)

    # --- Prepare for Composition Sampling ---
    latent_size = tb_meta['img_size'] // 4
    z_channels = vae_def.enc_cfg['z_ch']
    sample_shape = (args.batch_size, latent_size, latent_size, z_channels)
    logq_tb = jnp.zeros((args.batch_size,))
    logq_norm = jnp.zeros((args.batch_size,))

    @functools.partial(jax.jit, static_argnums=(3, 4, 5))
    def composed_score_fn(x, t, key, batch_size, score_fn_tb, score_fn_normal):
        # 1) scores from eps (∇ log q = -eps/σ), Alg.1 input
        eps_tb = score_fn_tb({'params': state_tb.params}, x, t)
        eps_normal = score_fn_normal({'params': state_normal.params}, x, t)
        sigma_t = marginal_prob_std_fn(t)[:, None, None, None]
        s_tb = -eps_tb / sigma_t
        s_norm = -eps_normal / sigma_t

        # Hutchinson divergences (used by AND RHS and diagnostics)
        key, div_key_tb, div_key_norm = jax.random.split(key, 3)
        dlog_tb = score_function_hutchinson_estimator(x, t, score_fn_tb, state_tb.params, div_key_tb)[0]
        dlog_norm = score_function_hutchinson_estimator(x, t, score_fn_normal, state_normal.params, div_key_norm)[0]

        # --- MODE LOGIC ---
        if args.mode == "poe":
            # Product-of-experts: κ=[1,1], u = s_tb + s_norm
            u = s_tb + s_norm
            kappa_diag = jnp.stack([jnp.ones_like(dlog_tb), jnp.ones_like(dlog_tb)], axis=1)

        elif args.mode == "and":
            # AND (Prop.6): solve linear equations G κ = b per-batch
            G, b = gram_and_rhs_from_scores((s_tb, s_norm), (dlog_tb, dlog_norm))  # [B,2,2],[B,2]
            kappa = solve_kappa_and(G, b)  # [B,2]
            # nonneg + normalize for stability
            kappa = jnp.clip(kappa, 0.0, 10.0)
            kappa = kappa / (jnp.sum(kappa, axis=1, keepdims=True) + 1e-8)
            k1, k2 = kappa[:, 0, None, None, None], kappa[:, 1, None, None, None]
            u = k1 * s_tb + k2 * s_norm
            kappa_diag = kappa

        elif args.mode == "or":
            # OR (Prop.3): κ = softmax(T * log q_i^t(x) + ℓ)
            # --- Itô density estimator (Thm.1): increment d log q for each model ---
            # We use the VP-SDE reverse vector field’s standard EM discretization:
            #   dx = (g^2 * s_i) dt + g dW    and the estimator term: ⟨∇, g^2 s_i⟩ dt
            # Practical, stable proxy:
            g_t = diffusion_coeff_fn(t)[:, None, None, None]
            dt = 1e-6 + 0.0 * jnp.mean(t)  # dt is provided outside; we pass it back via closure below

            # Hutchinson for divergence of (g^2 s_i): reuse the same v ~ N(0,I)
            # div(g^2 s) ≈ g^2 * div(s)  because g depends only on t here.
            div_s_tb_term = g_t.squeeze() ** 2 * dlog_tb
            div_s_norm_term = g_t.squeeze() ** 2 * dlog_norm

            # Accumulate log q (we update outer-scope scalars via return and rebind in the loop)
            new_logq_tb = logq_tb + div_s_tb_term * dt
            new_logq_norm = logq_norm + div_s_norm_term * dt

            # Softmax over T * logq + ℓ
            z1 = args.or_T * new_logq_tb + args.or_bias_l
            z2 = args.or_T * new_logq_norm + args.or_bias_l
            z = jnp.stack([z1, z2], axis=1)
            kappa = jax.nn.softmax(z, axis=1)  # [B,2]
            k1, k2 = kappa[:, 0, None, None, None], kappa[:, 1, None, None, None]
            u = k1 * s_tb + k2 * s_norm
            kappa_diag = kappa
            # return the updated logq’s through diagnostics so caller can rebind
            return u, {
                'score_norm_tb': jnp.linalg.norm(s_tb.reshape(batch_size, -1), axis=-1),
                'score_norm_normal': jnp.linalg.norm(s_norm.reshape(batch_size, -1), axis=-1),
                'score_norm_composed': jnp.linalg.norm(u.reshape(batch_size, -1), axis=-1),
                'dlog_tb': dlog_tb, 'dlog_normal': dlog_norm,
                'kappa': kappa[:, :, None, None],  # keep old plotting code happy
                'logq_tb_next': new_logq_tb,
                'logq_norm_next': new_logq_norm,
            }

        else:
            # args.mode == "analytic": keep your current heuristic kappa (existing behavior)
            kappa = get_kappa(t, (dlog_tb, dlog_norm), (s_tb, s_norm))
            kappa = jnp.clip(kappa, 0.0, 1.0)
            u = (1.0 - kappa) * s_norm + kappa * s_tb
            kappa_diag = jnp.concatenate([1.0 - kappa, kappa], axis=1) if kappa.ndim == 4 else kappa  # for plotting

        return u, {
            'score_norm_tb': jnp.linalg.norm(s_tb.reshape(batch_size, -1), axis=-1),
            'score_norm_normal': jnp.linalg.norm(s_norm.reshape(batch_size, -1), axis=-1),
            'score_norm_composed': jnp.linalg.norm(u.reshape(batch_size, -1), axis=-1),
            'dlog_tb': dlog_tb, 'dlog_normal': dlog_norm,
            'kappa': kappa_diag[:, :, None, None] if kappa_diag.ndim == 2 else kappa_diag,
        }

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
        score, diagnostics = composed_score_fn(x, t_batch, step_key, args.batch_size, state_tb.apply_fn,
                                               state_normal.apply_fn)
        diagnostics_history.append(jax.device_get(diagnostics))
        # --- OR branch: update the running logq estimates ---
        if args.mode == "or":
            logq_tb = diagnostics['logq_tb_next']
            logq_norm = diagnostics['logq_norm_next']
        # EM update
        g_t = diffusion_coeff_fn(t_batch)[:, None, None, None]
        noise = jax.random.normal(noise_key, x.shape)
        x = x + (g_t ** 2) * score * dt + g_t * jnp.sqrt(jnp.abs(dt)) * noise

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
        axs[0].plot(steps_axis, diag_agg['score_norm_composed'].mean(axis=-1), label="Composed", linestyle='--', alpha=0.8)
        axs[0].set_title("Score Norms"); axs[0].legend(); axs[0].set_xlabel("Step")
        axs[1].plot(steps_axis, diag_agg['dlog_tb'].mean(axis=-1), label="TB")
        axs[1].plot(steps_axis, diag_agg['dlog_normal'].mean(axis=-1), label="Normal")
        axs[1].set_title("Divergence (dlog p)"); axs[1].legend(); axs[1].set_xlabel("Step")
        axs[2].plot(steps_axis, diag_agg['kappa'].mean(axis=(1, 2, 3)))
        axs[2].set_title("Kappa (mean over batch)"); axs[2].set_xlabel("Step"); axs[2].set_ylim(0, 1)
        axs[3].hist(diag_agg['kappa'].flatten(), bins=50, range=(0, 1))
        axs[3].set_title("Kappa Distribution")
        diag_plot_path = os.path.join(output_dir, "diagnostics_plot.png")
        plt.tight_layout()
        plt.savefig(diag_plot_path)
        plt.close()
        print(f"Saved diagnostics plot to {diag_plot_path}")


if __name__ == "__main__":
    main()