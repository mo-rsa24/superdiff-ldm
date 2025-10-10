from typing import Callable
import jax
import jax.numpy as jnp
from jax import vmap

# ---------------------------
# Cosine VP schedule (Nichol-Dhariwal)
#   alpha_bar(t) = cos^2( (t + s)/(1 + s) * pi/2 ),  t in [0,1]
#   Use s≈0.008 for numerical stability.
# ---------------------------
_COSINE_S = 0.008
_PI_OVER_2 = jnp.pi / 2.0
_EPS = 1e-12

def _alpha_bar_cosine(t: jnp.ndarray, s: float = _COSINE_S) -> jnp.ndarray:
    u = (t + s) / (1.0 + s)
    # Clamp inside [0,1] to avoid tiny negatives from roundoff
    c = jnp.cos(jnp.clip(u, 0.0, 1.0) * _PI_OVER_2)
    return jnp.clip(c * c, _EPS, 1.0)

def alpha_bar_fn(t: jnp.ndarray) -> jnp.ndarray:
    """ᾱ(t) for VP-SDE."""
    return _alpha_bar_cosine(t)

def log_alpha_bar(t: jnp.ndarray) -> jnp.ndarray:
    """log ᾱ(t) for VP-SDE."""
    return jnp.log(alpha_bar_fn(t))

def alpha_fn(t: jnp.ndarray) -> jnp.ndarray:
    """α(t) = sqrt(ᾱ(t)) used in forward perturbation x_t = α(t) x_0 + σ(t) ε."""
    return jnp.sqrt(alpha_bar_fn(t))

def log_alpha(t: jnp.ndarray) -> jnp.ndarray:
    """log α(t) = 1/2 log ᾱ(t)."""
    return 0.5 * log_alpha_bar(t)

# d/dt log α(t) and d/dt log ᾱ(t)
dlog_alphadt: Callable[[jnp.ndarray], jnp.ndarray] = jax.grad(lambda tt: jnp.sum(log_alpha(tt)))
_dlogab_dt:    Callable[[jnp.ndarray], jnp.ndarray] = jax.grad(lambda tt: jnp.sum(log_alpha_bar(tt)))

def beta(t: jnp.ndarray) -> jnp.ndarray:
    """
    β(t) = - d/dt log ᾱ(t)  (VP-SDE definition)
    """
    return jnp.clip(-_dlogab_dt(t), _EPS, 1e12)

def marginal_prob_std(t: jnp.ndarray) -> jnp.ndarray:
    """
    σ(t) = sqrt(1 - ᾱ(t))  — standard deviation of p_0t(x_t | x_0) for VP-SDE.
    """
    return jnp.sqrt(jnp.clip(1.0 - alpha_bar_fn(t), _EPS, 1.0))

def diffusion_coeff(t: jnp.ndarray) -> jnp.ndarray:
    """
    g(t) = sqrt(β(t)) for VP-SDE.
    """
    return jnp.sqrt(beta(t))

# Vectorized (batch) versions used everywhere
marginal_prob_std_fn = vmap(marginal_prob_std)
diffusion_coeff_fn   = vmap(diffusion_coeff)
alpha_fn             = vmap(alpha_fn)
