"""
Loss functions for Multi-head SepVAE training.

This module implements:
1. Reconstruction loss (MSE)
2. KL divergence losses (standard + conditional priors)
3. Nulling loss (head routing penalty)
4. Mutual Information (MI) discriminator for disentanglement
5. Combined SepVAE loss with all components

Loss formula:
    L_total = L_rec + β_c·L_KL^c + β_s·(L_KL^sp + L_KL^se) + λ·L_null + κ·L_MI
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass
from typing import Dict, Tuple


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
        mu: Mean of posterior (B, latent_dim)
        logvar: Log-variance of posterior (B, latent_dim)

    Returns:
        Per-sample KL divergence (B,)
    """
    return 0.5 * jnp.sum(
        jnp.square(mu) + jnp.exp(logvar) - 1.0 - logvar,
        axis=-1
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
        mu: Mean of posterior (B, latent_dim)
        logvar: Log-variance of posterior (B, latent_dim)
        labels: Disease labels (B,) in {0, 1, 2}
        disease_id: Which disease this head represents (1=effusion, 2=cardio)
        sigma_inactive: Std dev of tight prior for inactive heads (default: 0.1)

    Returns:
        Per-sample KL divergence (B,)
    """
    # Determine which samples are active for this disease head
    is_active = (labels == disease_id).astype(jnp.float32)[:, None]  # (B, 1)

    # Prior variance: 1.0 if active, sigma_inactive² if inactive
    prior_logvar = jnp.where(is_active > 0.5, 0.0, jnp.log(sigma_inactive ** 2))
    prior_var = jnp.exp(prior_logvar)

    # KL(q(z|x) || N(0, prior_var·I))
    # Formula: KL = 0.5 * sum((μ²/σ_p² + σ²/σ_p² - 1 - log(σ²/σ_p²))
    #            = 0.5 * sum(μ²/σ_p² + exp(log_σ²)/σ_p² - 1 - (log_σ² - log_σ_p²))
    kl = 0.5 * jnp.sum(
        jnp.square(mu) / prior_var +
        jnp.exp(logvar) / prior_var -
        1.0 -
        (logvar - prior_logvar),
        axis=-1
    )

    return kl


def compute_kl_losses(
    latents_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    labels: jnp.ndarray,
    sigma_inactive: float = 0.1
) -> Dict[str, jnp.ndarray]:
    """
    Compute all KL divergence losses.

    Args:
        latents_dict: Dict with keys 'common', 'cardiomegaly', 'effusion'
                     Each value is tuple (mu, logvar)
        labels: Disease labels (B,)
        sigma_inactive: Tight prior std dev for inactive heads

    Returns:
        Dict with 'common', 'cardiomegaly', 'effusion' losses (all scalars)
    """
    mu_c, logvar_c = latents_dict['common']
    mu_cardio, logvar_cardio = latents_dict['cardiomegaly']
    mu_effusion, logvar_effusion = latents_dict['effusion']

    # Common head: always uses standard prior N(0, I)
    kl_common = jnp.mean(kl_divergence_standard(mu_c, logvar_c))

    # Disease heads: conditional priors based on labels
    # Cardiomegaly head is active when label=2
    kl_cardio = jnp.mean(
        kl_divergence_conditional(mu_cardio, logvar_cardio, labels, disease_id=2, sigma_inactive=sigma_inactive)
    )

    # Effusion head is active when label=1
    kl_effusion = jnp.mean(
        kl_divergence_conditional(mu_effusion, logvar_effusion, labels, disease_id=1, sigma_inactive=sigma_inactive)
    )

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
        inactive_mus: Dict with 'cardiomegaly', 'effusion' arrays (B, latent_dim)
                     Already masked to zero for active samples

    Returns:
        Scalar nulling loss
    """
    loss_cardio = jnp.mean(jnp.sum(jnp.square(inactive_mus['cardiomegaly']), axis=-1))
    loss_effusion = jnp.mean(jnp.sum(jnp.square(inactive_mus['effusion']), axis=-1))

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
        sigma_inactive: Tight prior std dev for inactive heads (default: 0.1)
    """
    weight_rec: float = 1.0
    weight_kl_common: float = 1e-4
    weight_kl_disease: float = 1e-4
    weight_null: float = 1e-3
    weight_mi: float = 1e-3
    sigma_inactive: float = 0.1


def sepvae_loss(
    model,
    mi_disc,
    params,
    mi_params,
    batch: Dict[str, jnp.ndarray],
    key: jax.random.PRNGKey,
    cfg: SepVAELossConfig
) -> Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], jnp.ndarray]]:
    """
    Complete SepVAE loss function.

    This combines:
    - Reconstruction loss
    - KL divergence (common + disease heads)
    - Nulling loss (inactive heads)
    - MI penalty (disentanglement)

    Args:
        model: SepVAE model
        mi_disc: MI discriminator model
        params: SepVAE parameters
        mi_params: MI discriminator parameters
        batch: Dict with 'x_norm', 'x_disease1', 'x_disease2', 'disease_labels'
        key: JAX PRNG key
        cfg: SepVAELossConfig with loss weights

    Returns:
        total_loss: Scalar
        (logs, disc_loss): Tuple of (dict with individual losses, discriminator loss)
    """
    # Unpack batch from VinBigData triplets
    x_norm = batch['x_norm']  # (B, 512, 512, 1)
    x_disease1 = batch['x_disease1']  # Effusion
    x_disease2 = batch['x_disease2']  # Cardiomegaly
    labels = batch['disease_labels']  # (3*B,) = [0, 0, ..., 1, 1, ..., 2, 2, ...]

    # Stack into single batch: (3*B, 512, 512, 1)
    x = jnp.concatenate([x_norm, x_disease1, x_disease2], axis=0)

    # Forward pass
    key1, key2 = jax.random.split(key)
    x_rec, latents_dict, inactive_mus = model.apply(
        {'params': params}, x, labels, key=key1, train=True
    )

    # 1. Reconstruction loss
    # Note: x is in [-1, 1], x_rec is in [0, 1]
    x_normalized = (x + 1.0) / 2.0  # Convert to [0, 1] for comparison
    l_rec = reconstruction_loss(x_normalized, x_rec)

    # 2. KL losses
    kl_losses = compute_kl_losses(latents_dict, labels, sigma_inactive=cfg.sigma_inactive)
    l_kl = (
        cfg.weight_kl_common * kl_losses['common'] +
        cfg.weight_kl_disease * (kl_losses['cardiomegaly'] + kl_losses['effusion'])
    )

    # 3. Nulling loss
    l_null = nulling_loss(inactive_mus)

    # 4. MI loss
    discriminator_fn = lambda z_c, z_d, train: mi_disc.apply(
        {'params': mi_params}, z_c, z_d, train=train
    )
    disc_loss, mi_penalty = mi_discriminator_loss(
        discriminator_fn, latents_dict, labels, key2
    )

    # Total VAE loss
    total_loss = (
        cfg.weight_rec * l_rec +
        l_kl +
        cfg.weight_null * l_null +
        cfg.weight_mi * mi_penalty
    )

    # Logging dict
    logs = {
        'loss/total': total_loss,
        'loss/reconstruction': l_rec,
        'loss/kl_common': kl_losses['common'],
        'loss/kl_cardiomegaly': kl_losses['cardiomegaly'],
        'loss/kl_effusion': kl_losses['effusion'],
        'loss/kl_total': kl_losses['common'] + kl_losses['cardiomegaly'] + kl_losses['effusion'],
        'loss/nulling': l_null,
        'loss/mi_penalty': mi_penalty,
        'loss/mi_disc': disc_loss,
    }

    return total_loss, (logs, disc_loss)


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
