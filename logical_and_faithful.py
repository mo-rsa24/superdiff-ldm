import os
import json
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import flax
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from flax.core.frozen_dict import freeze
from torchvision.utils import save_image, make_grid
import torch

# Component Imports
from run.ldm import load_autoencoder_
from models.cxr_unet import ScoreNet
from diffusion.vp_equation import marginal_prob_std_fn, diffusion_coeff_fn


def load_ldm_checkpoint(run_dir, ckpt_name):
    """
    Loads the Latent Diffusion Model checkpoint and configuration.
    """
    meta_path = os.path.join(run_dir, "ldm_meta.json")
    ckpt_path = os.path.join(run_dir, "ckpts", ckpt_name)
    with open(meta_path, 'r') as f:
        config = json.load(f)
    with open(config['ae_config_path'], 'r') as f:
        ae_cfg = json.load(f)

    ldm_chans = tuple(config['ldm_base_ch'] * int(m) for m in (
        config['ldm_ch_mults'].split(',') if isinstance(config['ldm_ch_mults'], str) else config['ldm_ch_mults']))

    model = ScoreNet(z_channels=ae_cfg['z_channels'], channels=ldm_chans, num_res_blocks=config['ldm_num_res_blocks'],
                     attn_resolutions=tuple(int(r) for r in str(config['ldm_attn_res']).split(',')), dtype=jnp.float32)

    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        raw_state = flax.serialization.msgpack_restore(f.read())
    return model, freeze(raw_state.get('ema_params') or raw_state.get('params')), config, config['img_size'] // (
            2 ** (len(ae_cfg['ch_mults'].split(',')) - 1)), ae_cfg['z_channels']


def get_ito_kappa_and_updates(s1, s2, g_t, dt, noise, x, f_drift=None):
    """
    Calculates the mixing weight (kappa) and density updates using the Itô Density Estimator
    (Theorem 1 & Prop 6), completely avoiding Hutchinson's divergence estimator.

    Args:
        s1, s2: Score functions for model 1 and 2.
        g_t: Diffusion coefficient g(t).
        dt: Time step size (scalar).
        noise: Sampled noise (epsilon).
        x: Current latent state.
        f_drift: Forward drift f(x). For VP-SDE f(x) = -0.5 * beta * x.
                 Note: The explicit f terms cancel in the kappa calculation for VP-SDE
                 but are needed for the absolute LL tracking.
    """
    # Vectorized operations over batch and spatial dims
    diff_s = s1 - s2

    # Calculate norms squared sum(|s|^2)
    s1_sq = jnp.sum(s1 ** 2, axis=(1, 2, 3))
    s2_sq = jnp.sum(s2 ** 2, axis=(1, 2, 3))
    diff_sq = jnp.sum(diff_s ** 2, axis=(1, 2, 3)) + 1e-12  # Avoid div by zero

    # Project noise onto the difference vector: <noise, s1-s2>
    noise_proj = jnp.sum(noise * diff_s, axis=(1, 2, 3))

    # 1. Calculate Kappa (Mixing Weight) for Logical AND
    # Analytical solution derived from equating Itô density updates (dlog_q1 = dlog_q2).
    # For VP-SDE, drift terms cancel, leaving a projection of noise scaled by diffusion.
    # Formula derived from Prop 6 in context of Eq 13.
    kappa = 0.5 - (noise_proj / (g_t.squeeze() * jnp.sqrt(jnp.abs(dt)) * diff_sq))

    # Clip kappa for stability (though theoretically unbounded, typically [0,1])
    kappa = jnp.clip(kappa, 0.0, 1.0)[:, None, None, None]

    # 2. Compute Density Updates (Log Likelihood tracking)
    # Using Eq 13 from Theorem 1: d log q = <dx, score> + <f - 0.5*g^2*score, score> dt
    # For the mixed step dx, the randomness is shared.

    # Itô correction terms (drift part of Eq 13)
    # term = <f, s> - 0.5 * g^2 * |s|^2
    # For VP-SDE, f(x) is handled outside or implicitly.
    # Here we calculate the g^2 part explicitly.
    g2 = (g_t.squeeze() ** 2)
    ito_term_1 = -0.5 * g2 * s1_sq
    ito_term_2 = -0.5 * g2 * s2_sq

    if f_drift is not None:
        # If tracking absolute likelihoods, add <f, s>
        ito_term_1 += jnp.sum(f_drift * s1, axis=(1, 2, 3))
        ito_term_2 += jnp.sum(f_drift * s2, axis=(1, 2, 3))

    return kappa, ito_term_1, ito_term_2


def run_comparison_sampling(rng, m1, p1, m2, p2, ae_model, ae_params, sz, ch, num_steps=1000, mode="average",
                            z_std=1.0):
    rng, step_rng = jax.random.split(rng)
    shape = (1, sz, sz, ch)
    x = jax.random.normal(step_rng, shape)
    t_vals = jnp.linspace(1.0, 1e-5, num_steps)
    dt = -1.0 / num_steps  # Negative time step for reverse SDE
    def get_score(model, params, x_t, t_scal):
        # Expand t to batch
        t_vec = jnp.ones((x_t.shape[0],)) * t_scal
        return model.apply({'params': params}, x_t, t_vec)

    print(f"Sampling with mode: {mode}...")

    for i in tqdm(range(num_steps)):
        t = t_vals[i]
        t_scalar = t
        s1 = get_score(m1, p1, x, t_scalar)
        s2 = get_score(m2, p2, x, t_scalar)
        if mode == "average":
            s_final = 0.5 * (s1 + s2)

        elif mode == "superdiff":
            g_t = diffusion_coeff_fn(t_scalar)
            rng, k_rng = jax.random.split(rng)
            eps = jax.random.normal(k_rng, x.shape)

            diff_s = s1 - s2
            norm_sq = jnp.sum(diff_s ** 2)
            dot_val = jnp.sum(eps * diff_s)
            denom = g_t * jnp.sqrt(jnp.abs(dt)) * norm_sq + 1e-8

            kappa = 0.5 - (dot_val / denom)
            kappa = jnp.clip(kappa, -1.0, 2.0)
            s_final = kappa * s1 + (1.0 - kappa) * s2

        else:
            raise ValueError(f"Unknown mode: {mode}")
        g_t = diffusion_coeff_fn(t_scalar)
        beta_t = g_t ** 2  # beta(t) = g(t)^2
        f_x = -0.5 * beta_t * x
        drift = f_x - (beta_t * s_final)
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, x.shape)
        diffusion = g_t * jnp.sqrt(jnp.abs(dt)) * z
        x = x + drift * dt + diffusion
    x_scaled = x / z_std
    out = ae_model.apply({'params': ae_params}, x_scaled, method=ae_model.decode)

    return out, None

def plot_trajectories(traces, out_path):
    plt.figure(figsize=(10, 5))
    for name, (ln, lt) in traces.items():
        if len(ln) > 0:
            plt.plot(ln, label=f"{name} - Model 1 LL", linestyle='--')
            plt.plot(lt, label=f"{name} - Model 2 LL")
    plt.title("Log-Likelihood Equilibrium Tracking (Itô Estimator)")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Cumulative Log-Density Change")
    plt.legend()
    plt.savefig(out_path)
    print(f"Trajectory plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir_1", type=str, required=True)
    parser.add_argument("--run_dir_2", type=str, required=True)
    args = parser.parse_args()

    m1, p1, cfg1, sz, ch = load_ldm_checkpoint(args.run_dir_1, "last.flax")
    m2, p2, _, _, _ = load_ldm_checkpoint(args.run_dir_2, "last.flax")
    ae_model, ae_params = load_autoencoder_(cfg1['ae_config_path'], cfg1['ae_ckpt_path'])

    z_std = 1.0 / cfg1.get('latent_scale_factor', 1.0)
    rng = jax.random.PRNGKey(42)

    # 1. Generate Average Samples (Baseline)
    print("Generating Average Samples...")
    img_avg, trace_avg = run_comparison_sampling(rng, m1, p1, m2, p2, ae_model, ae_params, sz, ch, 8, mode="average",
                                                 z_std=z_std)

    # 2. Generate SUPERDIFF Samples (Faithful Implementation)
    print("Generating SUPERDIFF (AND) Samples...")
    img_sd, trace_sd = run_comparison_sampling(rng, m1, p1, m2, p2, ae_model, ae_params, sz, ch, 8, mode="superdiff",
                                               z_std=z_std)

    # Combine into a single grid
    comparison = jnp.concatenate([img_avg, img_sd], axis=0)
    # Transpose for PyTorch format (B, C, H, W)
    grid = make_grid(torch.from_numpy(np.asarray(jnp.transpose(comparison, (0, 3, 1, 2)))), nrow=8)
    save_image(grid, "cxr_comparison_averaging_vs_superdiff_faithful.png")

    plot_trajectories({"Averaging": trace_avg, "SuperDiff": trace_sd}, "ll_trajectories_faithful.png")


if __name__ == "__main__":
    main()