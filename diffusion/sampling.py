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
        marginal_prob_std_fn, diffusion_coeff_fn,alpha_bar_fn,
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

        alpha_bar_now = alpha_bar_fn(vec_t)[:, None, None, None]
        alpha_bar_next = alpha_bar_fn(vec_t_next)[:, None, None, None]
        std_now = marginal_prob_std_fn(vec_t)[:, None, None, None]
        sqrt_alpha_bar_now = jnp.sqrt(alpha_bar_now + 1e-5)
        pred_x0 = (x - std_now * eps_theta) / sqrt_alpha_bar_now
        ratio = alpha_bar_now / (alpha_bar_next + 1e-5)
        sigma_sq = (1.0 - alpha_bar_next) / (1.0 - alpha_bar_now + 1e-8) * (1.0 - ratio)
        sigma = jnp.sqrt(jnp.clip(sigma_sq, 0.0, None))
        dir_xt_coeff = jnp.sqrt(jnp.clip(1.0 - alpha_bar_next - sigma_sq, 0.0, None))
        noise = jax.random.normal(jax.random.fold_in(rng, i), x.shape)
        x = (jnp.sqrt(alpha_bar_next) * pred_x0) + (dir_xt_coeff * eps_theta) + (sigma * noise)

    # Decode
    z_for_decode = x * z_std
    x_hat = ae_model.apply({'params': ae_params}, z_for_decode, method=ae_model.decode, train=False)

    print(f"Sample Stats: min={x_hat.min():.4f}, max={x_hat.max():.4f}, mean={x_hat.mean():.4f}")

    x_hat = jnp.clip(x_hat, 0., 1.)
    x_hat = jnp.transpose(x_hat, (0, 3, 1, 2))  # NHWC -> NCHW
    x_hat_t = torch.from_numpy(np.asarray(x_hat))

    grid = make_grid(x_hat_t, nrow=int(jnp.sqrt(batch_size)))
    return grid, x