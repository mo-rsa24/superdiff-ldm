"""
Loss functions for Multi-head SepVAE training.

This module implements:
1. Reconstruction loss (MSE)
2. KL divergence losses (standard + conditional priors + free-bits)
3. Nulling loss (head routing penalty)
4. Mutual Information (MI) discriminator for disentanglement
5. Backbone perceptual loss (multi-scale CheSS feature matching)
6. PatchGAN adversarial loss (hinge-based)
7. Combined SepVAE loss with all components

Loss formula:
    L_total = L_rec + β·L_KL + λ·L_null + κ·L_MI + γ·L_perceptual + α·L_GAN
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass
from typing import Dict, Tuple

from losses.lpips_gan import NLayerDiscriminator, hinge_d_loss, hinge_g_loss


# ============================================================================
# Reconstruction Loss
# ============================================================================

def reconstruction_loss(x_true: jnp.ndarray, x_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Gaussian reconstruction loss (Mean Squared Error).

    Args:
        x_true: Ground truth images (B, H, W, C)
        x_pred: Reconstructed images (B, H, W, C)

    Returns:
        Scalar loss (mean over batch)
    """
    mse_per_sample = jnp.mean((x_true - x_pred) ** 2, axis=(1, 2, 3))
    return jnp.mean(mse_per_sample)


# ============================================================================
# KL Divergence Losses
# ============================================================================

def kl_divergence_standard(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """
    KL divergence KL(q(z|x) || N(0, I)) for standard Gaussian prior.

    Formula:
        KL = 0.5 * sum(μ² + σ² - 1 - log(σ²))

    Args:
        mu: Mean of posterior (B, ..., latent_dim)
            Can be flat (B, D) or spatial (B, H, W, C)
        logvar: Log-variance of posterior (B, ..., latent_dim)

    Returns:
        Per-sample KL divergence (B,)
    """
    # Sum over all dimensions except batch
    sum_axes = tuple(range(1, mu.ndim))
    return 0.5 * jnp.sum(
        jnp.square(mu) + jnp.exp(logvar) - 1.0 - logvar,
        axis=sum_axes
    )


def kl_divergence_conditional(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    labels: jnp.ndarray,
    disease_id: int,
    sigma_inactive: float = 0.1
) -> jnp.ndarray:
    """
    Conditional KL divergence with label-dependent prior.

    For disease-specific heads:
    - Prior is N(0, I) when disease is present (label == disease_id)
    - Prior is N(0, σ_inactive²·I) when disease is absent (tighter prior)

    This soft pressure pulls inactive heads toward zero.

    Args:
        mu: Mean of posterior (B, ..., latent_dim)
            Can be flat (B, D) or spatial (B, H, W, C)
        logvar: Log-variance of posterior (B, ..., latent_dim)
        labels: Disease labels (B,) in {0, 1, 2}
        disease_id: Which disease this head represents (1=effusion, 2=cardio)
        sigma_inactive: Std dev of tight prior for inactive heads (default: 0.1)

    Returns:
        Per-sample KL divergence (B,)
    """
    # Determine which samples are active for this disease head
    # Broadcast to match mu/logvar shape
    is_active_shape = (labels.shape[0],) + (1,) * (mu.ndim - 1)  # (B, 1, 1, 1) for spatial
    is_active = (labels == disease_id).astype(jnp.float32).reshape(is_active_shape)

    # Prior variance: 1.0 if active, sigma_inactive² if inactive
    prior_logvar = jnp.where(is_active > 0.5, 0.0, jnp.log(sigma_inactive ** 2))
    prior_var = jnp.exp(prior_logvar)

    # KL(q(z|x) || N(0, prior_var·I))
    # Formula: KL = 0.5 * sum((μ²/σ_p² + σ²/σ_p² - 1 - log(σ²/σ_p²))
    #            = 0.5 * sum(μ²/σ_p² + exp(log_σ²)/σ_p² - 1 - (log_σ² - log_σ_p²))
    sum_axes = tuple(range(1, mu.ndim))
    kl = 0.5 * jnp.sum(
        jnp.square(mu) / prior_var +
        jnp.exp(logvar) / prior_var -
        1.0 -
        (logvar - prior_logvar),
        axis=sum_axes
    )

    return kl


def kl_divergence_free_bits(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    free_bits: float = 1.0,
) -> jnp.ndarray:
    """
    KL divergence with free-bits (per-channel minimum threshold).

    Free-bits prevents posterior collapse by exempting the first `free_bits`
    nats per channel from the KL penalty. This ensures each latent channel
    can encode at least `free_bits` nats of information without penalty.

    Algorithm:
        1. Compute per-element KL: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
        2. Average over spatial dims (H, W) per channel → (B, C)
        3. Clamp per-channel KL to min of `free_bits`
        4. Sum over channels → (B,)

    Args:
        mu: Mean of posterior (B, H, W, C) — spatial latents
        logvar: Log-variance of posterior (B, H, W, C)
        free_bits: Minimum KL per channel in nats (default: 1.0)

    Returns:
        Per-sample KL divergence (B,) with free-bits applied
    """
    # Per-element KL
    kl_elem = 0.5 * (jnp.square(mu) + jnp.exp(logvar) - 1.0 - logvar)

    if mu.ndim == 4:
        # Spatial latents: average over H, W → per-channel KL (B, C)
        kl_per_channel = jnp.mean(kl_elem, axis=(1, 2))
    else:
        # Flat latents: treat each dim as a "channel" (B, D)
        kl_per_channel = kl_elem

    # Apply free-bits: clamp per-channel KL to minimum
    kl_clamped = jnp.maximum(kl_per_channel, free_bits)

    if mu.ndim == 4:
        # Scale back: multiply by spatial size so total KL is comparable
        H, W = mu.shape[1], mu.shape[2]
        return jnp.sum(kl_clamped, axis=-1) * H * W
    else:
        return jnp.sum(kl_clamped, axis=-1)


def kl_divergence_conditional_free_bits(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    labels: jnp.ndarray,
    disease_id: int,
    sigma_inactive: float = 0.1,
    free_bits: float = 1.0,
) -> jnp.ndarray:
    """
    Conditional KL divergence with free-bits and label-dependent prior.

    Combines conditional prior (tight prior for inactive heads) with
    free-bits mechanism to prevent posterior collapse on active heads.

    Args:
        mu: Mean of posterior (B, H, W, C) or (B, D)
        logvar: Log-variance of posterior
        labels: Disease labels (B,)
        disease_id: Which disease this head represents
        sigma_inactive: Std dev of tight prior for inactive heads
        free_bits: Minimum KL per channel in nats

    Returns:
        Per-sample KL divergence (B,)
    """
    is_active_shape = (labels.shape[0],) + (1,) * (mu.ndim - 1)
    is_active = (labels == disease_id).astype(jnp.float32).reshape(is_active_shape)

    prior_logvar = jnp.where(is_active > 0.5, 0.0, jnp.log(sigma_inactive ** 2))
    prior_var = jnp.exp(prior_logvar)

    # Per-element KL against conditional prior
    kl_elem = 0.5 * (
        jnp.square(mu) / prior_var +
        jnp.exp(logvar) / prior_var -
        1.0 -
        (logvar - prior_logvar)
    )

    if mu.ndim == 4:
        kl_per_channel = jnp.mean(kl_elem, axis=(1, 2))  # (B, C)
    else:
        kl_per_channel = kl_elem

    kl_clamped = jnp.maximum(kl_per_channel, free_bits)

    if mu.ndim == 4:
        H, W = mu.shape[1], mu.shape[2]
        return jnp.sum(kl_clamped, axis=-1) * H * W
    else:
        return jnp.sum(kl_clamped, axis=-1)


def compute_kl_losses(
    latents_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    labels: jnp.ndarray,
    sigma_inactive: float = 0.1,
    free_bits: float = 0.0,
) -> Dict[str, jnp.ndarray]:
    """
    Compute all KL divergence losses.

    Args:
        latents_dict: Dict with keys 'common', 'cardiomegaly', 'effusion'
                     Each value is tuple (mu, logvar)
        labels: Disease labels (B,)
        sigma_inactive: Tight prior std dev for inactive heads
        free_bits: Per-channel free-bits threshold (0.0 = disabled)

    Returns:
        Dict with 'common', 'cardiomegaly', 'effusion' losses (all scalars)
    """
    mu_c, logvar_c = latents_dict['common']
    mu_cardio, logvar_cardio = latents_dict['cardiomegaly']
    mu_effusion, logvar_effusion = latents_dict['effusion']

    if free_bits > 0.0:
        # Free-bits KL: prevents posterior collapse
        kl_common = jnp.mean(kl_divergence_free_bits(mu_c, logvar_c, free_bits=free_bits))
        kl_cardio = jnp.mean(kl_divergence_conditional_free_bits(
            mu_cardio, logvar_cardio, labels, disease_id=2,
            sigma_inactive=sigma_inactive, free_bits=free_bits
        ))
        kl_effusion = jnp.mean(kl_divergence_conditional_free_bits(
            mu_effusion, logvar_effusion, labels, disease_id=1,
            sigma_inactive=sigma_inactive, free_bits=free_bits
        ))
    else:
        # Original KL (backward compatible)
        kl_common = jnp.mean(kl_divergence_standard(mu_c, logvar_c))
        kl_cardio = jnp.mean(kl_divergence_conditional(
            mu_cardio, logvar_cardio, labels, disease_id=2, sigma_inactive=sigma_inactive
        ))
        kl_effusion = jnp.mean(kl_divergence_conditional(
            mu_effusion, logvar_effusion, labels, disease_id=1, sigma_inactive=sigma_inactive
        ))

    return {
        'common': kl_common,
        'cardiomegaly': kl_cardio,
        'effusion': kl_effusion,
    }


# ============================================================================
# Nulling Loss
# ============================================================================

def nulling_loss(inactive_mus: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Penalize non-zero means for inactive disease heads.

    This is the hard routing constraint:
    - Normal (label=0): both disease heads should be zero
    - Cardiomegaly (label=2): effusion head should be zero
    - Effusion (label=1): cardiomegaly head should be zero

    Formula:
        L_null = mean(||μ_cardio_inactive||²) + mean(||μ_effusion_inactive||²)

    Args:
        inactive_mus: Dict with 'cardiomegaly', 'effusion' arrays
                     Can be flat (B, D) or spatial (B, H, W, C)
                     Already masked to zero for active samples

    Returns:
        Scalar nulling loss
    """
    # Sum over all dims except batch, then average over batch
    mu_cardio = inactive_mus['cardiomegaly']
    mu_effusion = inactive_mus['effusion']

    sum_axes = tuple(range(1, mu_cardio.ndim))
    loss_cardio = jnp.mean(jnp.sum(jnp.square(mu_cardio), axis=sum_axes))
    loss_effusion = jnp.mean(jnp.sum(jnp.square(mu_effusion), axis=sum_axes))

    return loss_cardio + loss_effusion


# ============================================================================
# Mutual Information Discriminator
# ============================================================================

class MIDiscriminator(nn.Module):
    """
    Mutual Information discriminator for disentanglement.

    Discriminates between:
    - Joint pairs: (z_common, z_disease) from same sample
    - Shuffled pairs: (z_common, z_disease) from different samples

    If the discriminator can't tell the difference, then z_common and z_disease
    are independent (low mutual information).

    Architecture:
        concat(z_c, z_d) → Dense(512) → ReLU → Dense(512) → ReLU → Dense(1)

    Attributes:
        hidden_dim: Hidden layer size (default: 512)
    """
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, z_common: jnp.ndarray, z_disease: jnp.ndarray, train: bool = True):
        """
        Args:
            z_common: Common latents (B, latent_dim_common)
            z_disease: Disease latents (B, latent_dim_disease)
            train: Training mode (unused, for API compatibility)

        Returns:
            Logits (B, 1) for binary classification (1=joint, 0=shuffled)
        """
        # Concatenate
        z = jnp.concatenate([z_common, z_disease], axis=-1)

        # 3-layer MLP
        h = nn.Dense(self.hidden_dim, name='fc1')(z)
        h = nn.relu(h)

        h = nn.Dense(self.hidden_dim, name='fc2')(h)
        h = nn.relu(h)

        logits = nn.Dense(1, name='fc3')(h)

        return logits  # (B, 1)


def mi_discriminator_loss(
    discriminator_fn,
    latents_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    labels: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute MI discriminator loss and VAE penalty.

    The discriminator is trained to distinguish:
    - Real pairs: (z_common, z_disease) from the same image
    - Fake pairs: (z_common, z_disease) from shuffled indices

    The VAE is trained to fool the discriminator (maximize MI penalty).

    Args:
        discriminator_fn: Function that takes (z_c, z_d, train) and returns logits
        latents_dict: Dict with (mu, logvar) for each head
                     Can be flat (B, D) or spatial (B, H, W, C)
        labels: Disease labels (B,) to select active disease head
        key: JAX PRNG key for shuffling

    Returns:
        disc_loss: Discriminator loss (binary cross-entropy)
        mi_penalty: Penalty for VAE (fool discriminator)
    """
    mu_c, _ = latents_dict['common']
    mu_cardio, _ = latents_dict['cardiomegaly']
    mu_effusion, _ = latents_dict['effusion']

    B = mu_c.shape[0]

    # If spatial, apply global average pooling to get flat representations
    if mu_c.ndim == 4:  # (B, H, W, C)
        mu_c = jnp.mean(mu_c, axis=(1, 2))  # (B, C)
        mu_cardio = jnp.mean(mu_cardio, axis=(1, 2))
        mu_effusion = jnp.mean(mu_effusion, axis=(1, 2))

    # Select active disease head based on labels
    # For cardio samples (label=2): use cardio latents
    # For effusion samples (label=1): use effusion latents
    # For normal samples (label=0): use zero latents (both are inactive)
    is_cardio = (labels == 2).astype(jnp.float32)[:, None]
    is_effusion = (labels == 1).astype(jnp.float32)[:, None]

    z_disease_joint = mu_cardio * is_cardio + mu_effusion * is_effusion

    # Joint pairs (real): (z_common[i], z_disease[i]) from same sample
    logits_joint = discriminator_fn(mu_c, z_disease_joint, train=True)

    # Shuffled pairs (fake): (z_common[i], z_disease[perm[i]]) from different samples
    key_shuffle = jax.random.split(key, 1)[0]
    perm = jax.random.permutation(key_shuffle, B)
    z_disease_shuffled = z_disease_joint[perm]
    logits_shuffled = discriminator_fn(mu_c, z_disease_shuffled, train=True)

    # Binary cross-entropy loss
    # Discriminator tries to output 1 for joint, 0 for shuffled
    # Using softplus for numerical stability: softplus(-x) = -log(sigmoid(x))
    loss_real = jnp.mean(nn.softplus(-logits_joint))  # Want logit → +∞
    loss_fake = jnp.mean(nn.softplus(logits_shuffled))  # Want logit → -∞
    disc_loss = loss_real + loss_fake

    # MI penalty for VAE: fool discriminator by making joint pairs look shuffled
    # Want discriminator to output 0 for joint pairs → minimize logits
    mi_penalty = jnp.mean(logits_joint)

    return disc_loss, mi_penalty


# ============================================================================
# Backbone Perceptual Loss
# ============================================================================

def backbone_perceptual_loss(
    x_orig: jnp.ndarray,
    x_rec: jnp.ndarray,
    backbone_apply_fn,
    backbone_variables: Dict,
) -> jnp.ndarray:
    """
    Perceptual loss using frozen CheSS backbone multi-scale features.

    Uses the same ResNet-50 backbone as the encoder to compute feature-level
    distance between original and reconstructed images. This leverages the
    medical-domain features learned by CheSS contrastive pretraining.

    Computes L1 distance at layers 2, 3, and 4, normalized per layer.

    Args:
        x_orig: Original images (B, 512, 512, 1) in [-1, 1] range
        x_rec: Reconstructed images (B, 512, 512, 1) in [0, 1] range
        backbone_apply_fn: Function to call backbone (returns multi-scale features)
        backbone_variables: Backbone model variables (params + batch_stats)

    Returns:
        Scalar perceptual loss (mean over batch and layers)
    """
    # Convert x_rec from [0,1] to [-1,1] to match backbone's expected input
    x_rec_scaled = x_rec * 2.0 - 1.0

    # Extract multi-scale features
    # Stop gradient on original features (target — should not change)
    feats_orig = backbone_apply_fn(backbone_variables, x_orig, return_multiscale=True)
    feats_orig = jax.tree_util.tree_map(jax.lax.stop_gradient, feats_orig)

    # Allow gradient to flow through reconstruction features → decoder can learn
    # Backbone weights are frozen (not in trainable params), but gradient flows
    # through the fixed computation graph back to x_rec
    feats_rec = backbone_apply_fn(backbone_variables, x_rec_scaled, return_multiscale=True)

    # L1 distance at each scale, normalized by spatial size
    loss = jnp.float32(0.0)
    for layer_name in ['layer2', 'layer3', 'layer4']:
        f_orig = feats_orig[layer_name]
        f_rec = feats_rec[layer_name]
        # Mean absolute difference per sample, then average over batch
        loss += jnp.mean(jnp.abs(f_orig - f_rec))

    return loss / 3.0  # Average over 3 layers


# ============================================================================
# PatchGAN Adversarial Loss
# ============================================================================

def patchgan_disc_loss(
    disc_apply_fn,
    disc_params: Dict,
    x_real: jnp.ndarray,
    x_fake: jnp.ndarray,
) -> jnp.ndarray:
    """
    PatchGAN discriminator loss (hinge).

    Args:
        disc_apply_fn: Discriminator apply function
        disc_params: Discriminator parameters
        x_real: Real images (B, H, W, C)
        x_fake: Fake/reconstructed images (B, H, W, C) — detached

    Returns:
        Scalar discriminator loss
    """
    logits_real = disc_apply_fn({'params': disc_params}, x_real)
    logits_fake = disc_apply_fn({'params': disc_params}, jax.lax.stop_gradient(x_fake))
    return hinge_d_loss(logits_real, logits_fake)


def patchgan_gen_loss(
    disc_apply_fn,
    disc_params: Dict,
    x_fake: jnp.ndarray,
) -> jnp.ndarray:
    """
    PatchGAN generator loss (hinge).

    Args:
        disc_apply_fn: Discriminator apply function
        disc_params: Discriminator parameters
        x_fake: Fake/reconstructed images (B, H, W, C)

    Returns:
        Scalar generator loss
    """
    logits_fake = disc_apply_fn({'params': disc_params}, x_fake)
    return hinge_g_loss(logits_fake)


# ============================================================================
# Combined SepVAE Loss
# ============================================================================

@dataclass
class SepVAELossConfig:
    """
    Configuration for SepVAE loss weights.

    Attributes:
        weight_rec: Reconstruction loss weight (default: 1.0)
        weight_kl_common: Common KL weight (default: 1e-4)
        weight_kl_disease: Disease KL weight (default: 1e-4)
        weight_null: Nulling loss weight (default: 1e-3)
        weight_mi: MI penalty weight (default: 1e-3)
        weight_perceptual: Backbone perceptual loss weight (default: 0.0 = disabled)
        weight_adversarial: PatchGAN generator loss weight (default: 0.0 = disabled)
        sigma_inactive: Prior std dev for inactive heads (default: 1.0, i.e. standard prior)
        free_bits: Per-channel free-bits for KL (default: 0.0 = disabled)
        disc_start_epoch: Epoch at which discriminator loss kicks in (default: 10)
    """
    weight_rec: float = 1.0
    weight_kl_common: float = 1e-4
    weight_kl_disease: float = 1e-4
    weight_null: float = 1e-3
    weight_mi: float = 1e-3
    weight_perceptual: float = 0.0
    weight_adversarial: float = 0.0
    sigma_inactive: float = 1.0
    free_bits: float = 0.0
    disc_start_epoch: int = 10


def sepvae_loss(
    model,
    mi_disc,
    params,
    mi_params,
    batch: Dict[str, jnp.ndarray],
    key: jax.random.PRNGKey,
    cfg: SepVAELossConfig,
    batch_stats: Dict = None,
    kl_anneal: jnp.ndarray = None,
    # New: adversarial & perceptual
    patch_disc=None,
    patch_disc_params: Dict = None,
    backbone_apply_fn=None,
    backbone_variables: Dict = None,
    current_epoch: int = 0,
) -> Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]]:
    """
    Complete SepVAE loss function with perceptual and adversarial terms.

    This combines:
    - Reconstruction loss (MSE)
    - KL divergence (common + disease heads, optional free-bits), scaled by kl_anneal
    - Nulling loss (inactive heads)
    - MI penalty (disentanglement)
    - Backbone perceptual loss (optional, multi-scale CheSS feature matching)
    - PatchGAN generator loss (optional, hinge-based)

    Args:
        model: SepVAE model
        mi_disc: MI discriminator model
        params: SepVAE parameters
        mi_params: MI discriminator parameters
        batch: Dict with 'x_norm', 'x_disease1', 'x_disease2', 'disease_labels'
        key: JAX PRNG key
        cfg: SepVAELossConfig with loss weights
        batch_stats: BatchNorm running statistics for frozen backbone
        kl_anneal: KL annealing factor in [0, 1] (default: None = no annealing, i.e. 1.0).
                   Used for KL warmup: linearly ramp from 0 to 1 over initial epochs.
        patch_disc: PatchGAN discriminator model (None = disabled)
        patch_disc_params: PatchGAN discriminator parameters
        backbone_apply_fn: Function to call backbone for perceptual loss (None = disabled)
        backbone_variables: Backbone variables for perceptual loss
        current_epoch: Current training epoch (for disc warmup)

    Returns:
        total_loss: Scalar
        (logs, mi_disc_loss, patch_disc_loss): Tuple of logging dict and discriminator losses
    """
    # Unpack batch from VinBigData triplets
    x_norm = batch['x_norm']  # (B, 512, 512, 1)
    x_disease1 = batch['x_disease1']  # Effusion
    x_disease2 = batch['x_disease2']  # Cardiomegaly
    labels = batch['disease_labels']  # (3*B,) = [0, 0, ..., 1, 1, ..., 2, 2, ...]

    # Stack into single batch: (3*B, 512, 512, 1)
    x = jnp.concatenate([x_norm, x_disease1, x_disease2], axis=0)

    # Build variables dict (params + optional batch_stats for frozen BN)
    variables = {'params': params}
    if batch_stats is not None:
        variables['batch_stats'] = batch_stats

    # Forward pass
    key1, key2 = jax.random.split(key)
    x_rec, latents_dict, inactive_mus = model.apply(
        variables, x, labels, key=key1, train=True
    )

    # 1. Reconstruction loss
    # Note: x is in [-1, 1], x_rec is in [0, 1]
    x_normalized = (x + 1.0) / 2.0  # Convert to [0, 1] for comparison
    l_rec = reconstruction_loss(x_normalized, x_rec)

    # 2. KL losses (with optional free-bits)
    kl_losses = compute_kl_losses(
        latents_dict, labels,
        sigma_inactive=cfg.sigma_inactive,
        free_bits=cfg.free_bits,
    )
    l_kl_raw = (
        cfg.weight_kl_common * kl_losses['common'] +
        cfg.weight_kl_disease * (kl_losses['cardiomegaly'] + kl_losses['effusion'])
    )

    # Apply KL annealing (warmup): scale weighted KL by anneal factor
    _kl_anneal = kl_anneal if kl_anneal is not None else jnp.float32(1.0)
    l_kl = l_kl_raw * _kl_anneal

    # 3. Nulling loss
    l_null = nulling_loss(inactive_mus)

    # 4. MI loss
    discriminator_fn = lambda z_c, z_d, train: mi_disc.apply(
        {'params': mi_params}, z_c, z_d, train=train
    )
    mi_disc_loss, mi_penalty = mi_discriminator_loss(
        discriminator_fn, latents_dict, labels, key2
    )

    # 5. Backbone perceptual loss (optional)
    if cfg.weight_perceptual > 0.0 and backbone_apply_fn is not None:
        l_perceptual = backbone_perceptual_loss(
            x, x_rec, backbone_apply_fn, backbone_variables
        )
    else:
        l_perceptual = jnp.float32(0.0)

    # 6. PatchGAN generator loss (optional, with warmup)
    patch_disc_loss_val = jnp.float32(0.0)
    l_gen = jnp.float32(0.0)
    disc_factor = jnp.float32(0.0)
    if cfg.weight_adversarial > 0.0 and patch_disc is not None and patch_disc_params is not None:
        # Warmup: ramp from 0 to 1 over 5000 steps after disc_start_epoch
        disc_factor = jnp.where(
            current_epoch >= cfg.disc_start_epoch,
            jnp.clip((current_epoch - cfg.disc_start_epoch) / 5.0, 0.0, 1.0),
            0.0,
        )
        disc_apply_fn = patch_disc.apply
        l_gen = patchgan_gen_loss(disc_apply_fn, patch_disc_params, x_rec)
        # Discriminator loss (for separate optimizer)
        patch_disc_loss_val = patchgan_disc_loss(
            disc_apply_fn, patch_disc_params, x_normalized, x_rec
        )

    # Total VAE loss
    total_loss = (
        cfg.weight_rec * l_rec +
        l_kl +
        cfg.weight_null * l_null +
        cfg.weight_mi * mi_penalty +
        cfg.weight_perceptual * l_perceptual +
        cfg.weight_adversarial * disc_factor * l_gen
    )

    # Logging dict
    logs = {
        'loss/total': total_loss,
        'loss/reconstruction': l_rec,
        'loss/kl_common': kl_losses['common'],
        'loss/kl_cardiomegaly': kl_losses['cardiomegaly'],
        'loss/kl_effusion': kl_losses['effusion'],
        'loss/kl_total': kl_losses['common'] + kl_losses['cardiomegaly'] + kl_losses['effusion'],
        'loss/kl_weighted': l_kl,
        'loss/kl_anneal': _kl_anneal,
        'loss/nulling': l_null,
        'loss/mi_penalty': mi_penalty,
        'loss/mi_disc': mi_disc_loss,
        'loss/perceptual': l_perceptual,
        'loss/gen_adversarial': l_gen,
        'loss/disc_factor': disc_factor,
        'loss/patch_disc': patch_disc_loss_val,
    }

    return total_loss, (logs, mi_disc_loss, patch_disc_loss_val)


# ============================================================================
# Testing
# ============================================================================

def test_losses():
    """Test all loss functions with dummy data."""
    print("Testing loss functions...")

    B, H, W, C = 4, 512, 512, 1
    latent_dim_common = 128
    latent_dim_disease = 64

    # Dummy data
    x_true = jnp.ones((B, H, W, C)) * 0.5
    x_pred = jnp.ones((B, H, W, C)) * 0.4

    mu_c = jnp.zeros((B, latent_dim_common))
    logvar_c = jnp.zeros((B, latent_dim_common))
    mu_cardio = jnp.ones((B, latent_dim_disease)) * 0.5
    logvar_cardio = jnp.zeros((B, latent_dim_disease))
    mu_effusion = jnp.ones((B, latent_dim_disease)) * 0.3
    logvar_effusion = jnp.zeros((B, latent_dim_disease))

    labels = jnp.array([0, 1, 2, 1])  # Normal, Effusion, Cardio, Effusion

    # Test reconstruction loss
    l_rec = reconstruction_loss(x_true, x_pred)
    print(f"✓ Reconstruction loss: {l_rec:.6f}")

    # Test KL divergence
    kl_std = jnp.mean(kl_divergence_standard(mu_c, logvar_c))
    print(f"✓ KL divergence (standard): {kl_std:.6f}")

    latents_dict = {
        'common': (mu_c, logvar_c),
        'cardiomegaly': (mu_cardio, logvar_cardio),
        'effusion': (mu_effusion, logvar_effusion),
    }

    kl_losses = compute_kl_losses(latents_dict, labels)
    print(f"✓ KL losses: common={kl_losses['common']:.6f}, "
          f"cardio={kl_losses['cardiomegaly']:.6f}, "
          f"effusion={kl_losses['effusion']:.6f}")

    # Test nulling loss
    inactive_mus = {
        'cardiomegaly': mu_cardio * 0.5,
        'effusion': mu_effusion * 0.3,
    }
    l_null = nulling_loss(inactive_mus)
    print(f"✓ Nulling loss: {l_null:.6f}")

    # Test MI discriminator
    print("\nTesting MI discriminator...")
    mi_disc = MIDiscriminator(hidden_dim=512)

    z_c = mu_c
    z_d = mu_cardio

    key = jax.random.PRNGKey(0)
    variables = mi_disc.init(key, z_c, z_d, train=True)

    logits = mi_disc.apply(variables, z_c, z_d, train=True)
    print(f"✓ MI discriminator output shape: {logits.shape}")

    discriminator_fn = lambda z_c, z_d, train: mi_disc.apply(variables, z_c, z_d, train=train)
    disc_loss, mi_penalty = mi_discriminator_loss(discriminator_fn, latents_dict, labels, key)
    print(f"✓ MI discriminator loss: {disc_loss:.6f}, penalty: {mi_penalty:.6f}")

    print("\n✓ All loss tests passed!")


if __name__ == "__main__":
    test_losses()
