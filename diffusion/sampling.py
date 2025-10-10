import numpy as np
from jax import *
import jax
import jax.numpy as jnp
from tqdm import tqdm
import torch
from torchvision.utils import make_grid

from diffusion.vp_equation import beta


def _sum_except_batch(x):
    axes = tuple(range(1, x.ndim))
    return jnp.sum(x, axis=axes, keepdims=True)


def _broadcast_time(t_scalar, x):
    """Make t broadcast like x: (N,1[,1,1...])"""
    N = x.shape[0]
    extra_ones = (1,) * (x.ndim - 1)
    return jnp.ones((N,) + extra_ones, dtype=x.dtype) * t_scalar


def Euler_Maruyama_sampler(
    rng, ldm_model, ldm_params, ae_model, ae_params,
    marginal_prob_std_fn, diffusion_coeff_fn,
    latent_size, batch_size, z_channels, z_std=1.0,
    n_steps=700, eps=1e-3  # Use eps=1e-3 for stability with this SDE
):
    """
    Corrected Euler-Maruyama sampler using the correct reverse-time SDE.
    """
    print("Running CORRECTED Euler-Maruyama sampler with stable equations...")
    rngs = jax.random.split(rng, batch_size)
    single_sample_shape = (latent_size, latent_size, z_channels)
    init_x = jax.vmap(lambda key: jax.random.normal(key, single_sample_shape))(rngs)
    init_x = init_x * marginal_prob_std_fn(jnp.ones(batch_size))[:, None, None, None]

    time_steps = jnp.linspace(1., eps, n_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    for i, t in enumerate(tqdm(time_steps, desc="Sampling")):
        step_key = jax.random.fold_in(rng, i)
        vec_t = jnp.ones(batch_size) * t
        g = diffusion_coeff_fn(vec_t)
        std = marginal_prob_std_fn(vec_t)
        assert jnp.all(jnp.isfinite(g))
        predicted_noise = ldm_model.apply({'params': ldm_params}, x, vec_t)
        beta_vec = (g ** 2)[:, None, None, None]
        score = -predicted_noise / (std[:, None, None, None] + 1e-8)
        drift = -0.5 * beta_vec * x - beta_vec * score

        # drift = -0.5 * beta(vec_t)[:, None, None, None] * x - (g**2)[:, None, None, None] * score

        diffusion = g[:, None, None, None] * jax.random.normal(step_key, x.shape)
        x_mean = x - drift * step_size
        x = x_mean + diffusion * jnp.sqrt(step_size)
    final_z_for_decode = x # The sampler already produces a latent at the correct scale
    z_for_decode = final_z_for_decode * z_std
    x_hat = ae_model.apply({'params': ae_params}, z_for_decode, method=ae_model.decode, train=False)
    x_hat = jnp.clip(x_hat, 0., 1.)
    x_hat = jnp.transpose(x_hat, (0, 3, 1, 2)) # NHWC -> NCHW
    x_hat_t = torch.from_numpy(np.asarray(x_hat))

    grid = make_grid(x_hat_t, nrow=int(jnp.sqrt(batch_size)))
    return grid, x

signal_to_noise_ratio = 0.16  # @param {'type':'number'}

## The number of sampling steps.
num_steps = 500  # @param {'type':'integer'}


def select_sampler(name: str):
    name = name.lower()
    if name in ("em", "euler", "euler-maruyama"):
        return Euler_Maruyama_sampler
    raise ValueError(f"Unknown sampler: {name}")