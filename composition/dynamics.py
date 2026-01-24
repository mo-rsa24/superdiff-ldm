from operator import itemgetter
import jax
import jax.numpy as jnp

from composition.utils import decode_image, save_image_grid
from diffusion.vp_equation import marginal_prob_std, marginal_prob_std_fn, diffusion_coeff_fn, alpha_bar_fn, score_function_hutchinson_estimator
from PIL import Image
import numpy as np
from flax.training.train_state import TrainState
from typing import Any, List, Tuple
from models.cxr_unet import ScoreNet
from diffusion.sampling import Euler_Maruyama_sampler, DDPM_ancestral_sampler
from diffusion.vp_equation import marginal_prob_std_fn, diffusion_coeff_fn, alpha_bar_fn
from tqdm import tqdm
from run.ldm import load_autoencoder_


def get_score(model, params, x, t):
    """
    Computes the score (epsilon) for a given latent x and time t.
    Note: ScoreNet in run/ldm.py is trained to predict epsilon.
    t is expected in [0, 1].
    """
    return model.apply({'params': params}, x, t)

def generate_ldm_samples(rng, normal, tb, ae_model, ae_params, batch_size: int = 4,  sampler: str = "Ancestral", n_steps=200, latent_scale_factor=0.99937266, output_path: str ="x_ray_samples.png"):
    print("Loading Model 1 (Normal)...")
    model_normal, params_normal, cfg1, lsize, zch = itemgetter("ldm_model", "params", "config", "latent_size",
                                                               "z_channels")(normal)
    print("Loading Model 2 (TB)...")
    model_tb, params_tb, cfg2, _, _ = itemgetter("ldm_model", "params", "config", "latent_size", "z_channels")(tb)
    print("Sampling Images For Both Models")
    # 1. Select the Sampler
    if sampler == 'Euler':
        sampler_fn = Euler_Maruyama_sampler
    elif sampler == 'Ancestral':
        sampler_fn = DDPM_ancestral_sampler
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    sample_kwargs = {
        "rng": rng,
        "ae_model": ae_model,
        "ae_params": ae_params,
        "marginal_prob_std_fn": marginal_prob_std_fn,
        "diffusion_coeff_fn": diffusion_coeff_fn,
        "alpha_bar_fn": alpha_bar_fn,
        "latent_size": lsize,
        "batch_size": batch_size,
        "z_channels": zch,
        "z_std": 1.0,
        "n_steps": n_steps,
    }
    # 3. Generate Samples
    print(f"Generating samples for Model 1 (Normal) using {sampler}...")
    _, latents_normal = sampler_fn(ldm_model=model_normal, ldm_params=params_normal, **sample_kwargs)

    print(f"Generating samples for Model 2 (TB) using {sampler}...")
    _, latents_tb = sampler_fn(ldm_model=model_tb, ldm_params=params_tb, **sample_kwargs)

    # 4. Decode and Construct 4x2 Grid
    # Stack latents: [Normal_1...Normal_4, TB_1...TB_4]
    combined_latents = jnp.concatenate([latents_normal, latents_tb], axis=0)

    print("Decoding independent samples...")
    decoded_imgs = decode_image(ae_model, ae_params, combined_latents, latent_scale_factor=latent_scale_factor)
    decoded_imgs_np = np.array(decoded_imgs)  # Shape: (8, H, W)
    sample_out_path = output_path.replace(".png", f"_reference_{sampler}.png")
    save_image_grid(decoded_imgs_np, sample_out_path , 4, 2)
    print(f"Reference samples (Normal top, TB bottom) saved to {sample_out_path}")

def ddpm_ancestral_superdiff_and_uncond(
    rng,
    latents,
    model_normal, params_normal,
    model_tb, params_tb,
    num_inference_steps,
    lift=0.0,
    kappa_clip=2.0,
):
    """
    Hybrid SUPERDIFF-AND:
      - models output epsilon (ε̂)
      - convert to score: s = -ε̂ / σ(t)
      - compute kappa using divergence estimates (Hutchinson) + inner-product term
      - mix scores, convert back to ε̂_mix
      - take a DDPM ancestral posterior step using ε̂_mix (sharp, stable)

    Inputs/outputs match your code structure. :contentReference[oaicite:1]{index=1}
    """
    timesteps = jnp.linspace(1.0, 1e-5, num_inference_steps + 1)
    kappa_log = []

    x = latents

    def _sum_except_batch(xx):
        # sum over (H,W,C) axes -> keep batch dimension
        axes = tuple(range(1, xx.ndim))
        return jnp.sum(xx, axis=axes, keepdims=True)

    for i in tqdm(range(num_inference_steps), desc="SuperDiff AND (DDPM ancestral)"):
        t_now  = timesteps[i]
        t_next = timesteps[i + 1]

        B = x.shape[0]
        vec_t      = jnp.full((B,), t_now)
        vec_t_next = jnp.full((B,), t_next)

        # ---- (1) ε predictions from both experts ----
        eps_normal = model_normal.apply({'params': params_normal}, x, vec_t)
        eps_tb     = model_tb.apply({'params': params_tb}, x, vec_t)

        # ---- (2) Convert ε -> score: s = -ε/σ ----
        std_now = marginal_prob_std_fn(vec_t)[:, None, None, None]  # σ(t)
        sN = -eps_normal / std_now
        sT = -eps_tb     / std_now

        # ---- (3) Divergence estimates div(s) via Hutchinson ----
        # estimator expects a function returning ε̂; it internally converts to div(score)
        rng, k1, k2 = jax.random.split(rng, 3)

        divN, _ = score_function_hutchinson_estimator(
            x, vec_t,
            score_fn=model_normal.apply,
            params=params_normal,
            key=jax.random.fold_in(k1, i),
        )
        divT, _ = score_function_hutchinson_estimator(
            x, vec_t,
            score_fn=model_tb.apply,
            params=params_tb,
            key=jax.random.fold_in(k2, i),
        )

        divN_b = divN[:, None, None, None]
        divT_b = divT[:, None, None, None]

        # ---- (4) Compute κ (SuperDiff-inspired, score-space) ----
        # Structure matches your vp_equation-style expression: g(t)^2 (divN-divT) + <sN, sN-sT> / ||sN-sT||^2
        g2 = diffusion_coeff_fn(vec_t)[:, None, None, None] ** 2

        numerator   = g2 * (divN_b - divT_b) + _sum_except_batch(sN * (sN - sT))
        if lift is not None:
            numerator = numerator + (std_now * (lift[:, None, None, None] / num_inference_steps))

        denominator = _sum_except_batch((sN - sT) ** 2) + 1e-12
        kappa = numerator / denominator

        # stability clamp (recommended for TB ∧ Normal)
        if kappa_clip is not None:
            kappa = jnp.clip(kappa, -kappa_clip, kappa_clip)

        kappa_log.append(kappa.squeeze())

        # ---- (5) Mix scores, convert back to ε̂_mix ----
        s_mix = sN + kappa * (sT - sN)
        eps_mix = -std_now * s_mix

        # ---- (6) DDPM ancestral posterior step using ε̂_mix ----
        alpha_bar_now  = alpha_bar_fn(vec_t)[:, None, None, None]
        alpha_bar_next = alpha_bar_fn(vec_t_next)[:, None, None, None]
        sqrt_alpha_bar_now = jnp.sqrt(alpha_bar_now + 1e-5)

        # x0 estimate (same form as your DDPM sampler)
        pred_x0 = (x - std_now * eps_mix) / sqrt_alpha_bar_now

        ratio = alpha_bar_now / (alpha_bar_next + 1e-5)
        sigma_sq = (1.0 - alpha_bar_next) / (1.0 - alpha_bar_now + 1e-8) * (1.0 - ratio)
        sigma = jnp.sqrt(jnp.clip(sigma_sq, 0.0, None))
        dir_xt_coeff = jnp.sqrt(jnp.clip(1.0 - alpha_bar_next - sigma_sq, 0.0, None))

        rng, kn = jax.random.split(rng)
        noise = jax.random.normal(jax.random.fold_in(kn, i), x.shape)

        x = (jnp.sqrt(alpha_bar_next) * pred_x0) + (dir_xt_coeff * eps_mix) + (sigma * noise)

    return x, jnp.array(kappa_log)

def stochastic_super_diff_and_uncond(
        latents,
        model_normal, params_normal,
        model_tb, params_tb,
        num_inference_steps,
        lift=0.0,
        score = False
):
    """
    Implements Stochastic SuperDiff AND logic for two unconditional score fields.
    Mapping:
      - Normal Model -> Background/Reference (vel_bg)
      - TB Model     -> Object/Constraint (vel_obj)
    """

    timesteps = jnp.linspace(1.0, 1e-5, num_inference_steps + 1)

    # Initialize logs
    kappa_log = []

    for i in tqdm(range(num_inference_steps), desc="SuperDiff AND"):
        t_current = timesteps[i]
        t_next = timesteps[i + 1]
        t_batch = jnp.full((latents.shape[0],), t_current)
        sigma = marginal_prob_std(t_current)
        sigma_next = marginal_prob_std(t_next)
        dsigma = sigma_next - sigma

        vel_normal = get_score(model_normal, params_normal, latents, t_batch)
        vel_tb = get_score(model_tb, params_tb, latents, t_batch)
        if score:
            vel_normal = - vel_normal / sigma
            vel_tb = - vel_tb / sigma

        # Independent step (baseline trajectory)
        noise = jax.random.normal(jax.random.PRNGKey(i), latents.shape) * jnp.sqrt(2 * jnp.abs(dsigma) * sigma)
        dx_ind = 2 * dsigma * vel_normal + noise

        term1 = (jnp.abs(dsigma) * (vel_normal - vel_tb) * (vel_normal + vel_tb)).sum(axis=(1, 2, 3))
        term2 = (dx_ind * (vel_tb - vel_normal)).sum(axis=(1, 2, 3))
        term3 = sigma * lift / num_inference_steps
        numerator = term1 - term2 + term3
        denominator = 2 * dsigma * ((vel_tb - vel_normal) ** 2).sum(axis=(1, 2, 3))

        kappa = numerator / (denominator + 1e-8)
        kappa_log.append(kappa)
        kappa_b = kappa[:, None, None, None]

        # Composite Vector Field
        vf = vel_normal + kappa_b * (vel_tb - vel_normal)

        # 4. Update Latents (Euler-Maruyama step)
        dx = 2 * dsigma * vf + noise
        latents = latents + dx

    return latents, jnp.array(kappa_log)

def get_sweep_configuration(latent_size, z_channels: int =4, lift_values: Tuple[float] = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0), num_rows: int = 4, seed: int =0):
    rng = jax.random.PRNGKey(seed)
    row_latents = []
    t0 = 1.0
    sigma_max = marginal_prob_std_fn(jnp.array([t0]))[0]
    for i in range(num_rows):
        rng, seed_rng = jax.random.split(rng)
        # Generate ONE sample (NHWC)
        z0 = jax.random.normal(seed_rng, (1, latent_size, latent_size, z_channels)) * sigma_max
        # Tile it NUM_COLS times
        z0_row = jnp.tile(z0, (len(lift_values), 1, 1, 1))
        row_latents.append(z0_row)
    latents = jnp.concatenate(row_latents, axis=0)
    lifts_jnp = jnp.array(lift_values)
    lift_batch = jnp.tile(lifts_jnp, num_rows)  # Shape (Total Batch,)
    print(f"Total Batch Shape: {latents.shape}")
    return latents, lift_batch