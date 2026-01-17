import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import torch
from torchvision.utils import make_grid


def _sum_except_batch(x):
    axes = tuple(range(1, x.ndim))
    return jnp.sum(x, axis=axes, keepdims=True)


def _broadcast_time(t_scalar, x):
    """Make t broadcast like x: (N,1[,1,1...])"""
    N = x.shape[0]
    extra_ones = (1,) * (x.ndim - 1)
    return jnp.ones((N,) + extra_ones, dtype=x.dtype) * t_scalar


def DDPM_ancestral_sampler(
        rng, ldm_model, ldm_params, ae_model, ae_params,
        marginal_prob_std_fn, diffusion_coeff_fn,
        latent_size, batch_size, z_channels, z_std=1.0,
        n_steps=500, eps=1e-5
):
    """
    Ancestral Sampling (DDPM) for VP-SDE.
    This is numerically more stable than Euler-Maruyama near t=0.

    Uses the discretization:
    x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + dir_xt * eps_theta + sigma * noise
    """
    print(f"Running Ancestral (DDPM) Sampler with {n_steps} steps...")
    rngs = jax.random.split(rng, batch_size)
    single_sample_shape = (latent_size, latent_size, z_channels)

    # Start from pure noise at t=1.0
    # For VP-SDE, marginal_std(1.0) is effectively 1.0, but we use the fn for correctness.
    init_x = jax.vmap(lambda key: jax.random.normal(key, single_sample_shape))(rngs)
    init_x = init_x * marginal_prob_std_fn(jnp.array([1.0]))[0]

    # Time steps from 1.0 down to eps
    timesteps = jnp.linspace(1.0, eps, n_steps + 1)
    x = init_x

    for i in tqdm(range(n_steps), desc="DDPM Sampling"):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]  # t_next < t_now

        # Broadcast time to batch
        vec_t = jnp.ones(batch_size) * t_now
        vec_t_next = jnp.ones(batch_size) * t_next

        # 1. Predict Noise
        eps_theta = ldm_model.apply({'params': ldm_params}, x, vec_t)

        # 2. Derive alpha_bar from marginal_prob_std (std = sqrt(1 - alpha_bar))
        # This avoids needing to pass alpha_bar_fn explicitly
        std_now = marginal_prob_std_fn(vec_t)[:, None, None, None]
        std_next = marginal_prob_std_fn(vec_t_next)[:, None, None, None]

        alpha_bar_now = jnp.clip(1.0 - std_now ** 2, 0.0, 1.0)
        alpha_bar_next = jnp.clip(1.0 - std_next ** 2, 0.0, 1.0)
        sqrt_alpha_bar_now = jnp.sqrt(alpha_bar_now + 1e-8)
        pred_x0 = (x - std_now * eps_theta) / sqrt_alpha_bar_now
        ratio = alpha_bar_now / (alpha_bar_next + 1e-8)
        sigma_sq = (1.0 - alpha_bar_next) / (1.0 - alpha_bar_now + 1e-8) * (1.0 - ratio)
        sigma = jnp.sqrt(jnp.clip(sigma_sq, 0.0, None))
        dir_xt_coeff = jnp.sqrt(jnp.clip(1.0 - alpha_bar_next - sigma_sq, 0.0, None))
        noise = jax.random.normal(jax.random.fold_in(rng, i), x.shape)

        x = (jnp.sqrt(alpha_bar_next) * pred_x0) + (dir_xt_coeff * eps_theta) + (sigma * noise)
        if i % 250 == 0 or i == n_steps - 1:
            print(f"\n[Step {i}/{n_steps}] t_now={t_now:.4f} -> t_next={t_next:.4f}")
            
            # 1. Check Coefficients
            print(f"  alpha_bar_now: {alpha_bar_now[0,0,0,0]:.6f}")
            print(f"  std_now:       {std_now[0,0,0,0]:.6f}")
            print(f"  coeff_prev:    {jnp.sqrt(alpha_bar_next)[0,0,0,0]:.6f}")
            print(f"  coeff_dir:     {dir_xt_coeff[0,0,0,0]:.6f}")
            print(f"  sigma:         {sigma[0,0,0,0]:.6f}")

            # 2. Check Magnitudes (Mean/Max/NaNs)
            x_norm = jnp.linalg.norm(x[0].ravel())
            eps_norm = jnp.linalg.norm(eps_theta[0].ravel())
            pred_x0_val = pred_x0[0].ravel()
            
            print(f"  |eps_theta|:   {eps_norm:.4f} (Mean: {eps_theta.mean():.4f})")
            print(f"  |x_prev|:      {x_norm:.4f} (Min: {x.min():.2f}, Max: {x.max():.2f})")
            print(f"  x0_pred stats: Min={pred_x0_val.min():.2f}, Max={pred_x0_val.max():.2f}")

            # 3. Check for explosion
            if jnp.isnan(x).any():
                print("!!! NaN DETECTED IN LATENTS !!!")
                break
            if x_norm > 1e5:
                print("!!! EXPLOSION DETECTED !!!")
                break

    # Decode
    z_for_decode = x * z_std
    x_hat = ae_model.apply({'params': ae_params}, z_for_decode, method=ae_model.decode, train=False)

    print(f"Sample Stats: min={x_hat.min():.4f}, max={x_hat.max():.4f}, mean={x_hat.mean():.4f}")

    x_hat = jnp.clip(x_hat, 0., 1.)
    x_hat = jnp.transpose(x_hat, (0, 3, 1, 2))  # NHWC -> NCHW
    x_hat_t = torch.from_numpy(np.asarray(x_hat))

    grid = make_grid(x_hat_t, nrow=int(jnp.sqrt(batch_size)))
    return grid, x