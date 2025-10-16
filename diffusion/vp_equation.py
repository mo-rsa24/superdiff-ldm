from typing import Callable
import jax
import jax.numpy as jnp
from jax import vmap
import functools
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

@functools.partial(jax.jit, static_argnums=(2,))
def score_function_hutchinson_estimator(x, t, score_fn, params, key):
  v = jax.random.normal(key, x.shape)
  def epsilon_fn(y):
      return score_fn({'params': params}, y, t)
  _, jvp_val = jax.jvp(epsilon_fn, (x,), (v,))
  sigma_t = marginal_prob_std_fn(t)[:, None, None, None]
  divergence = -jnp.sum(v * jvp_val, axis=(1, 2, 3)) / sigma_t.squeeze()

  return divergence, divergence

def _sum_except_batch(x):
  """Sum over all non-batch axes, keep batch axis."""
  axes = tuple(range(1, x.ndim))
  return jnp.sum(x, axis=axes, keepdims=True)


@jax.jit
def get_kappa(t, divlogs, scores):
  """
  Calculates the optimal mixing coefficient kappa for Superdiffusion, consistent
  with the VP-SDE defined in this file.

  Args:
    t: The current timestep, a JAX array of shape (N,).
    divlogs: A tuple of two JAX arrays (div_score1, div_score2),
             representing the divergence of the two score functions.
             Shape of each is (N,).
    scores: A tuple of two JAX arrays (score1, score2), representing the
            score functions of the two models. Shape of each is (N, H, W, C).

  Returns:
    A JAX array of shape (N, 1, 1, 1) representing kappa, which can be
    broadcast to the shape of the latents.
  """
  div1, div2 = divlogs
  s1, s2 = scores
  div1 = div1[:, None, None, None]
  div2 = div2[:, None, None, None]
  g_t_squared = diffusion_coeff_fn(t)[:, None, None, None]**2
  numerator = g_t_squared * (div1 - div2) + _sum_except_batch(s1 * (s1 - s2))
  denominator = _sum_except_batch((s1 - s2)**2) + 1e-12 # Add epsilon for stability

  kappa = numerator / denominator
  return kappa

# Vectorized (batch) versions used everywhere
marginal_prob_std_fn = vmap(marginal_prob_std)
diffusion_coeff_fn   = vmap(diffusion_coeff)
alpha_fn             = vmap(alpha_fn)


def sum_except_batch(x):
    axes = tuple(range(1, x.ndim))
    return jnp.sum(x, axis=axes, keepdims=True)

def gram_and_rhs_from_scores(scores, dlogs):
    """
    scores: tuple/list of M arrays [B,H,W,C]
    dlogs : tuple/list of M arrays [B]
    returns:
      G: [B,M,M] Gram matrix with G_ij = <s_i, s_j>
      b: [B,M]    RHS vector with b_i = dlog_i
    """
    flats = [s.reshape(s.shape[0], -1) for s in scores]          # [B,D] each
    S = jnp.stack(flats, axis=-1)                                # [B,D,M]
    G = jnp.einsum("bdm,bdn->bmn", S, S)                         # [B,M,M]
    b = jnp.stack(dlogs, axis=1)                                 # [B,M]
    return G, b

def solve_kappa_and(G, b, eps=1e-6):
    eye = jnp.eye(G.shape[-1])[None]
    G_reg = G + eps * eye
    # batched solve
    return jax.vmap(lambda Gb, bb: jnp.linalg.solve(Gb, bb))(G_reg, b)  # [B,M]