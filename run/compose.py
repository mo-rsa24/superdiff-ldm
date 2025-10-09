#!/usr/bin/env python3
# compose.py — compose two pretrained CXR score models and sample images
# Usage examples:
#   python run/compose.py --mode superdiff --sampler em --batch_size 16
#   python run/compose.py --mode fixed --alpha 0.5 --sampler pc
# Place this file in the `run/` directory.

import os, sys, json, math, functools
from datetime import datetime
from typing import Tuple, Optional
import argparse
from tqdm import tqdm
import numpy as np

# --- ensure repo root on sys.path (so 'diffusion', 'models', etc. can be resolved)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from diffusion import sde_lib, sampling
from models import unet, ae_kl
from config import lib as config_lib
from utils import image_manipulation

# --- arg parsing ---
parser = argparse.ArgumentParser(description="Compose two pretrained CXR diffusion models.")
parser.add_argument("--run_tb", type=str, default="ldm-tb-proto-3_lr3e-5_wd0.01_ch96_20251008-192534",
                    help="Name of the TB model run directory.")
parser.add_argument("--run_normal", type=str, default="ldm-normal-proto-3_lr3e-5_wd0.01_ch96_20251008-192824",
                    help="Name of the Normal model run directory.")
parser.add_argument("--mode", type=str, default="superdiff", choices=["fixed", "time", "superdiff"],
                    help="Composition mode.")
parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for 'fixed' composition.")
parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for Superdiff kappa calculation.")
parser.add_argument("--sampler", type=str, default="em", choices=["em", "pc"],
                    help="Sampler to use (Euler-Maruyama or Predictor-Corrector).")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for sampling.")
parser.add_argument("--steps", type=int, default=1000, help="Number of sampling steps.")
parser.add_argument("--eps", type=float, default=1e-3, help="SDE terminal time.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--diagnostics", action='store_true', default=True,
                    help="Enable diagnostic logging and plotting for Superdiff.")
args = parser.parse_args()


def load_model_and_params(run_name: str) -> Tuple[config_lib.Config, unet.UNet, ae_kl.AutoencoderKL, dict, dict]:
    """Loads config, models, and parameters for a given run."""
    run_dir = os.path.join("runs", run_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Load config
    cfg = config_lib.Config(config_lib.config_from_json(os.path.join(run_dir, "config.json")))

    # Load models
    score_model = unet.UNet(cfg.model.unet)
    vae = ae_kl.AutoencoderKL(cfg.model.vae)

    # Load checkpoints
    score_params = checkpoints.restore_checkpoint(os.path.join(run_dir, "checkpoints", "score_model"), target=None)
    vae_params = checkpoints.restore_checkpoint(os.path.join(cfg.model.vae_ckpt_path, "checkpoints", "vae"),
                                                target=None)

    if score_params is None or vae_params is None:
        raise ValueError(f"Failed to load checkpoints from {run_dir} or {cfg.model.vae_ckpt_path}")

    print(f"Successfully loaded model '{run_name}'")
    return cfg, score_model, vae, score_params, vae_params


def get_composed_sampler(sde, score_fn_a, score_fn_b, sampler_name: str, batch_size: int, img_size: int,
                         steps: int, eps: float, alpha: float, mode: str):
    """Creates a sampler with a composed score function."""
    if mode == "fixed":
        composed_score_fn = sampling.get_fixed_composed_score_fn(score_fn_a, score_fn_b, alpha)
    elif mode == "time":
        composed_score_fn = sampling.get_time_composed_score_fn(sde, score_fn_a, score_fn_b)
    else:
        raise ValueError(f"Unknown composition mode: {mode}")

    shape = (batch_size, img_size, img_size, 1)
    if sampler_name == "em":
        sampler = sampling.get_em_sampler(sde, composed_score_fn, shape, inverse_scaler=lambda x: x,
                                          denoise=True, steps=steps, eps=eps)
    elif sampler_name == "pc":
        sampler = sampling.get_pc_sampler(sde, composed_score_fn, shape, inverse_scaler=lambda x: x,
                                          denoise=True, steps=steps, eps=eps, predictor='em', corrector='none')
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    return sampler


def get_superdiff_sampler_and_diagnostics(sde, score_fn_tb, score_fn_normal, guidance_scale,
                                          shape, key, steps, eps):
    """
    Creates and runs a sampler implementing the Superdiff algorithm with diagnostics.
    This is based on the Euler-Maruyama sampler.
    """
    rng = key
    batch_size = shape[0]

    # Initialize diagnostics storage
    diagnostics = {
        'kappa': jnp.zeros((steps, batch_size)),
        'score_norm_tb': jnp.zeros((steps, batch_size)),
        'score_norm_normal': jnp.zeros((steps, batch_size)),
        'score_norm_composed': jnp.zeros((steps, batch_size)),
        'dlog_tb': jnp.zeros((steps, batch_size)),
        'dlog_normal': jnp.zeros((steps, batch_size)),
        'intermediate_latents': []
    }

    # Interval to save intermediate latents
    save_interval = steps // 10

    @jax.jit
    def superdiff_step_fn(i, state):
        rng, x, current_diagnostics = state

        # 0. Setup for current step
        rng, step_rng = jax.random.split(rng)
        t = sde.ts[i]
        vec_t = jnp.full(batch_size, t)

        # 1. Get the scores from your two models
        s_tb = score_fn_tb(x, vec_t)
        s_normal = score_fn_normal(x, vec_t)

        # 2. Compute the divergence (dlog_p) for both models using Hutchinson trace estimator
        eps_div = jax.random.randint(step_rng, x.shape, 0, 2, dtype=x.dtype) * 2 - 1

        # Divergence for the TB model
        _, jvp_tb = jax.jvp(lambda xt: score_fn_tb(xt, vec_t), (x,), (eps_div,))
        dlog_tb = -jnp.sum(eps_div * jvp_tb, axis=(1, 2, 3))

        # Divergence for the Normal model
        _, jvp_normal = jax.jvp(lambda xt: score_fn_normal(xt, vec_t), (x,), (eps_div,))
        dlog_normal = -jnp.sum(eps_normal * jvp_normal, axis=(1, 2, 3)) if 'eps_normal' in locals() else -jnp.sum(
            eps_div * jvp_normal, axis=(1, 2, 3))

        # 3. Compute kappa
        sigma = sde.scheduler.sigmas[i]
        dsigma = sde.scheduler.sigmas[i + 1] - sigma

        s_diff = s_tb - s_normal

        kappa_numerator = sigma * (dlog_tb - dlog_normal) + jnp.sum((s_tb - s_normal) * (s_tb + s_normal),
                                                                    axis=(1, 2, 3))
        kappa_numerator += -jnp.sum((s_tb - s_normal) * (guidance_scale * s_normal), axis=(1, 2, 3))
        kappa_denominator = guidance_scale * jnp.sum((s_tb - s_normal) ** 2, axis=(1, 2, 3))

        # Avoid division by zero
        kappa = kappa_numerator / (kappa_denominator + 1e-9)

        # 4. Compute the composed score
        kappa_reshaped = kappa[:, None, None, None]
        composed_score = s_normal + kappa_reshaped * (s_tb - s_normal)

        # 5. Update latents using Euler-Maruyama step
        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, x.shape)
        drift = sde.g2[i] * composed_score * sde.dt
        diffusion = sde.g[i] * jnp.sqrt(sde.dt) * noise
        x_mean = x + drift
        x = x_mean + diffusion

        # 6. Log diagnostics
        current_diagnostics['kappa'] = current_diagnostics['kappa'].at[i].set(kappa)
        current_diagnostics['score_norm_tb'] = current_diagnostics['score_norm_tb'].at[i].set(
            jnp.linalg.norm(s_tb.reshape(batch_size, -1), axis=-1))
        current_diagnostics['score_norm_normal'] = current_diagnostics['score_norm_normal'].at[i].set(
            jnp.linalg.norm(s_normal.reshape(batch_size, -1), axis=-1))
        current_diagnostics['score_norm_composed'] = current_diagnostics['score_norm_composed'].at[i].set(
            jnp.linalg.norm(composed_score.reshape(batch_size, -1), axis=-1))
        current_diagnostics['dlog_tb'] = current_diagnostics['dlog_tb'].at[i].set(dlog_tb)
        current_diagnostics['dlog_normal'] = current_diagnostics['dlog_normal'].at[i].set(dlog_normal)

        return rng, x, current_diagnostics

    # --- Main Sampling Loop ---
    # We cannot save intermediate latents inside a jitted loop.
    # So we run the loop manually to capture them.
    rng, x_rng = jax.random.split(rng)
    x = sde.prior_sampling(x_rng, shape)

    state = (rng, x, diagnostics)

    # Manual loop with tqdm for progress bar
    for i in tqdm(range(sde.N - 1, -1, -1), desc="Superdiff Sampling"):
        # We cannot jit the entire loop if we want to save intermediate states
        state = superdiff_step_fn(i, state)

        # Save intermediate latent state
        if i % save_interval == 0:
            diagnostics['intermediate_latents'].append(state[1])  # append x

    final_rng, final_x, final_diagnostics = state

    # Denoise final step
    t_eps = jnp.full(batch_size, eps)
    final_score = score_fn_normal(final_x, t_eps) + (final_diagnostics['kappa'][-1, :, None, None, None] * (
                score_fn_tb(final_x, t_eps) - score_fn_normal(final_x, t_eps)))
    final_x = final_x + sde.scheduler.get_g2(eps) * final_score * (
                sde.scheduler.get_sigma(eps) ** 2 - sde.scheduler.get_sigma(0) ** 2) / (
                          2 * sde.scheduler.get_sigma(eps))

    return final_x, final_diagnostics


def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    key = jax.random.PRNGKey(args.seed)

    # --- Load models ---
    print("Loading TB model...")
    cfg_a, score_model_a, vae, params_a, vae_params = load_model_and_params(args.run_tb)

    print("\nLoading Normal model...")
    cfg_b, score_model_b, _, params_b, _ = load_model_and_params(args.run_normal)

    # --- Create SDE and score functions ---
    sde = sde_lib.VESDE(beta_min=cfg_a.model.sde.beta_min, beta_max=cfg_a.model.sde.beta_max, N=args.steps)
    img_size = cfg_a.data.image_size

    # Create individual score functions
    score_fn_a = sampling.get_score_fn(sde, score_model_a, params_a, train=False)
    score_fn_b = sampling.get_score_fn(sde, score_model_b, params_b, train=False)

    # VAE decoding function
    vae_decode_fn = functools.partial(vae.apply, {'params': vae_params}, method=vae.decode)

    # --- Create output directory ---
    samples_dir = os.path.join("samples", f"composition_{ts}")
    os.makedirs(samples_dir, exist_ok=True)
    print(f"Saving samples and diagnostics to: {samples_dir}")

    # --- Sampling ---
    key, sample_key = jax.random.split(key)

    if args.mode == 'superdiff':
        shape = (args.batch_size, img_size // 8, img_size // 8, cfg_a.model.vae.z_channels)
        latents, diagnostics = get_superdiff_sampler_and_diagnostics(
            sde, score_fn_a, score_fn_b, args.guidance_scale, shape, sample_key, args.steps, args.eps
        )
    else:  # Handle older composition modes
        sampler = get_composed_sampler(sde, score_fn_a, score_fn_b, args.sampler, args.batch_size,
                                       img_size // 8, args.steps, args.eps, args.alpha, args.mode)
        latents, _ = sampler(sample_key)

    # --- Decode and Save final images ---
    # Use the appropriate latent scale factor from your training run, not the LDM default.
    # This is the reciprocal of the factor used to scale latents *before* training the UNet.
    latent_scale_factor = 3.133603
    latents = 1 / latent_scale_factor * latents
    imgs = vae_decode_fn(latents)

    # NHWC [0,1] -> NCHW tensor grid
    imgs = jnp.clip(imgs, 0.0, 1.0)
    imgs = jnp.transpose(imgs.reshape((-1, img_size, img_size, 1)), (0, 3, 1, 2))

    import torch
    from torchvision.utils import make_grid, save_image
    imgs_t = torch.tensor(np.asarray(imgs))
    grid = make_grid(imgs_t, nrow=int(math.sqrt(max(1, imgs_t.shape[0]))))

    out_png = os.path.join(samples_dir, f"compose_{args.mode}_alpha{args.alpha:g}_{ts}.png")
    save_image(grid, out_png)
    print(f"Saved final image grid to {out_png}")

    # --- Process and save diagnostics for Superdiff ---
    if args.mode == 'superdiff' and args.diagnostics:
        import matplotlib.pyplot as plt

        # 1. Save raw diagnostics data
        np.savez(os.path.join(samples_dir, "diagnostics.npz"), **diagnostics)

        # 2. Create and save plots
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))
        steps_axis = np.arange(args.steps)

        # Kappa Stability
        axs[0].plot(steps_axis, diagnostics['kappa'].mean(axis=-1))
        axs[0].set_title("Kappa Stability (mean over batch)")
        axs[0].set_xlabel("Sampling Step")
        axs[0].set_ylabel("Kappa Value")

        # Score Norms
        axs[1].plot(steps_axis, diagnostics['score_norm_tb'].mean(axis=-1), label="TB Score Norm")
        axs[1].plot(steps_axis, diagnostics['score_norm_normal'].mean(axis=-1), label="Normal Score Norm")
        axs[1].plot(steps_axis, diagnostics['score_norm_composed'].mean(axis=-1), label="Composed Score Norm",
                    linestyle='--')
        axs[1].set_title("Score Norms (mean over batch)")
        axs[1].set_xlabel("Sampling Step")
        axs[1].set_ylabel("L2 Norm")
        axs[1].legend()

        # Divergence Values
        axs[2].plot(steps_axis, diagnostics['dlog_tb'].mean(axis=-1), label="TB Divergence")
        axs[2].plot(steps_axis, diagnostics['dlog_normal'].mean(axis=-1), label="Normal Divergence")
        axs[2].set_title("Divergence Values (mean over batch)")
        axs[2].set_xlabel("Sampling Step")
        axs[2].set_ylabel("Divergence")
        axs[2].legend()

        # Kappa Dist
        axs[3].hist(diagnostics['kappa'].flatten(), bins=50)
        axs[3].set_title("Distribution of Kappa values")
        axs[3].set_xlabel("Kappa")
        axs[3].set_ylabel("Frequency")

        plt.tight_layout()
        plot_path = os.path.join(samples_dir, "diagnostics_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved diagnostics plot to {plot_path}")

        # 3. Decode and save intermediate latents
        intermediate_imgs = []
        for inter_latents in diagnostics['intermediate_latents']:
            inter_latents = 1 / 0.18215 * inter_latents
            inter_imgs = vae_decode_fn(inter_latents)
            inter_imgs = jnp.clip(inter_imgs, 0.0, 1.0)
            inter_imgs = jnp.transpose(inter_imgs.reshape((-1, img_size, img_size, 1)), (0, 3, 1, 2))
            intermediate_imgs.append(torch.tensor(np.asarray(inter_imgs)))

        if intermediate_imgs:
            # Take the first image from each saved batch to show its evolution
            evolution_grid = make_grid([imgs[0] for imgs in intermediate_imgs], nrow=len(intermediate_imgs))
            evo_path = os.path.join(samples_dir, "intermediate_samples.png")
            save_image(evolution_grid, evo_path)
            print(f"Saved intermediate sample evolution to {evo_path}")


if __name__ == "__main__":
    main()

