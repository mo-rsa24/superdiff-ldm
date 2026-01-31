"""
Multi-head SepVAE with frozen CheSS ResNet-50 backbone.

This module implements a Separated Variational Autoencoder (SepVAE) with:
- Frozen CheSS ResNet-50 backbone for feature extraction
- Three parallel MLP heads for disentangled latent spaces:
  * Common head (128-dim): Shared anatomy features
  * Cardiomegaly head (64-dim): Disease-specific features
  * Pleural Effusion head (64-dim): Disease-specific features
- Head nulling logic to prevent information leakage
- Shared decoder for reconstruction

Architecture:
    Input (512×512×1) → ResNet50 (frozen) → h(2048)
                                           ↓
                        ┌──────────────────┼──────────────────┐
                        ↓                  ↓                  ↓
                   Common Head       Cardio Head       Effusion Head
                    (128-dim)          (64-dim)           (64-dim)
                        ↓                  ↓                  ↓
                        └──────────────────┬──────────────────┘
                                           ↓
                                    Head Nulling
                                           ↓
                                  Concat latents (256)
                                           ↓
                                       Decoder
                                           ↓
                                Output (512×512×1)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, Sequence

from models.resnet_jax import ResNet50CheSS
from models.ae_kl import ResBlock, Up


class ConvHead(nn.Module):
    """
    Convolutional head that projects spatial features to latent μ and log_σ.

    Architecture:
        h(64×64×2048) → Conv(3×3, hidden_ch) → ReLU → Conv(3×3, out_ch*2)
        Split output into μ and log_σ

    Attributes:
        out_channels: Output channels (latent channels)
        hidden_channels: Hidden layer channels (default: 512)
    """
    out_channels: int
    hidden_channels: int = 512

    @nn.compact
    def __call__(self, h):
        """
        Args:
            h: Spatial features (B, 64, 64, in_channels)

        Returns:
            mu: Mean of latent distribution (B, 64, 64, out_channels)
            logvar: Log-variance of latent distribution (B, 64, 64, out_channels)
        """
        # Hidden conv layer
        x = nn.Conv(self.hidden_channels, kernel_size=(3, 3), padding='SAME', name='conv1')(h)
        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.relu(x)

        # Output layer (2x out_channels for mu and logvar)
        x = nn.Conv(self.out_channels * 2, kernel_size=(3, 3), padding='SAME', name='conv2')(x)

        # Split into mu and logvar
        mu, logvar = jnp.split(x, 2, axis=-1)

        return mu, logvar


class SepVAEEncoder(nn.Module):
    """
    Three-head encoder with frozen ResNet-50 backbone for spatial latents.

    Outputs spatial latent maps (64×64):
    - Common head: 64×64×4 channels (shared anatomy)
    - Cardiomegaly head: 64×64×2 channels (disease-specific)
    - Pleural Effusion head: 64×64×2 channels (disease-specific)

    Total: 64×64×8 for LDM training

    Attributes:
        z_channels_common: Common latent channels (default: 4)
        z_channels_disease: Disease-specific latent channels (default: 2)
        frozen_backbone: Whether to freeze backbone gradients (default: True)
    """
    z_channels_common: int = 4
    z_channels_disease: int = 2
    frozen_backbone: bool = True

    def setup(self):
        """Initialize backbone and three Conv heads."""
        self.backbone = ResNet50CheSS()
        self.head_common = ConvHead(out_channels=self.z_channels_common, name='head_common')
        self.head_cardiomegaly = ConvHead(out_channels=self.z_channels_disease, name='head_cardiomegaly')
        self.head_effusion = ConvHead(out_channels=self.z_channels_disease, name='head_effusion')

    def __call__(self, x, train: bool = True):
        """
        Forward pass through encoder.

        Args:
            x: Input images (B, 512, 512, 1)
            train: Training mode (unused, for API compatibility)

        Returns:
            Dict with keys 'common', 'cardiomegaly', 'effusion'
            Each value is a tuple (mu, logvar) with shape (B, 64, 64, channels)
        """
        # Extract spatial features (B, 64, 64, 2048)
        h = self.backbone(x, return_spatial=True)

        # Freeze backbone if specified
        if self.frozen_backbone:
            h = jax.lax.stop_gradient(h)

        # Three parallel Conv heads
        mu_c, logvar_c = self.head_common(h)  # (B, 64, 64, 4)
        mu_cardio, logvar_cardio = self.head_cardiomegaly(h)  # (B, 64, 64, 2)
        mu_effusion, logvar_effusion = self.head_effusion(h)  # (B, 64, 64, 2)

        return {
            'common': (mu_c, logvar_c),
            'cardiomegaly': (mu_cardio, logvar_cardio),
            'effusion': (mu_effusion, logvar_effusion),
        }


def apply_head_nulling(
    latents_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    labels: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Apply disease-specific head nulling based on labels (spatial version).

    Nulling logic:
    - label=0 (Normal): Zero both disease heads
    - label=1 (Pleural Effusion): Zero Cardiomegaly head
    - label=2 (Cardiomegaly): Zero Pleural Effusion head

    This prevents information leakage from inactive heads.

    Args:
        latents_dict: Dict with keys 'common', 'cardiomegaly', 'effusion'
                     Each value is tuple (mu, logvar) with shape (B, 64, 64, channels)
        labels: Disease labels (B,) with values in {0, 1, 2}
        key: JAX PRNG key for sampling

    Returns:
        z_concat: Concatenated spatial latents (B, 64, 64, 8) with nulling applied
        inactive_mus: Dict of masked (inactive) means for nulling loss
    """
    B = labels.shape[0]

    # Sample from common head (always active)
    mu_c, logvar_c = latents_dict['common']  # (B, 64, 64, 4)
    key_c, key = jax.random.split(key)
    eps_c = jax.random.normal(key_c, mu_c.shape)
    z_c = mu_c + jnp.exp(0.5 * logvar_c) * eps_c

    # Sample from disease heads
    mu_cardio, logvar_cardio = latents_dict['cardiomegaly']  # (B, 64, 64, 2)
    mu_effusion, logvar_effusion = latents_dict['effusion']  # (B, 64, 64, 2)

    key_cardio, key_effusion = jax.random.split(key)
    eps_cardio = jax.random.normal(key_cardio, mu_cardio.shape)
    eps_effusion = jax.random.normal(key_effusion, mu_effusion.shape)

    z_cardio = mu_cardio + jnp.exp(0.5 * logvar_cardio) * eps_cardio
    z_effusion = mu_effusion + jnp.exp(0.5 * logvar_effusion) * eps_effusion

    # Create nulling masks for spatial tensors
    # Cardiomegaly head is active only when label=2
    mask_cardio = jnp.where(labels == 2, 1.0, 0.0)[:, None, None, None]  # (B, 1, 1, 1)

    # Effusion head is active only when label=1
    mask_effusion = jnp.where(labels == 1, 1.0, 0.0)[:, None, None, None]  # (B, 1, 1, 1)

    # Apply masks to sampled latents (broadcast across spatial dimensions)
    z_cardio_masked = z_cardio * mask_cardio
    z_effusion_masked = z_effusion * mask_effusion

    # Concatenate along channel dimension: [common(4), cardio(2), effusion(2)] = 8 channels
    z_concat = jnp.concatenate([z_c, z_cardio_masked, z_effusion_masked], axis=-1)  # (B, 64, 64, 8)

    # Track inactive means for nulling loss
    # Inactive means should be pushed to zero by the loss
    inactive_mus = {
        'cardiomegaly': mu_cardio * (1.0 - mask_cardio),
        'effusion': mu_effusion * (1.0 - mask_effusion),
    }

    return z_concat, inactive_mus


class SepVAEDecoder(nn.Module):
    """
    Shared decoder that reconstructs 512×512 images from spatial latents.

    Architecture:
        z(64×64×8) → Process → Upsample (3×) → 128×128 → 256×256 → 512×512
                   → Conv1×1 → Sigmoid → Output (512×512×1)

    Each upsampling stage uses:
    - ResBlocks (2 per stage)
    - Bilinear upsampling + Conv (from ae_kl.py)
    - GroupNorm + Swish activation

    Attributes:
        ch_mults: Channel multipliers for each stage (default: 256→128→64→32)
        num_res_blocks: ResBlocks per stage (default: 2)
        dropout: Dropout rate (default: 0.0)
        z_channels: Total latent channels (4 + 2 + 2 = 8)
    """
    ch_mults: Sequence[int] = (256, 128, 64, 32)  # 4 stages: 64→128→256→512
    num_res_blocks: int = 2
    dropout: float = 0.0
    z_channels: int = 8  # 4 (common) + 2 (cardio) + 2 (effusion)

    @nn.compact
    def __call__(self, z, train: bool = True):
        """
        Decode spatial latents to reconstructed images.

        Args:
            z: Spatial latents (B, 64, 64, 8)
            train: Training mode for dropout

        Returns:
            Reconstructed images (B, 512, 512, 1) in [0, 1] range
        """
        # Initial processing: project from z_channels to first channel mult
        h = nn.Conv(self.ch_mults[0], kernel_size=(3, 3), padding='SAME', name='z_proj')(z)
        # h is now (B, 64, 64, 256)

        # Progressive upsampling: 64 → 128 → 256 → 512
        # 4 stages total
        for i in reversed(range(len(self.ch_mults))):
            ch = self.ch_mults[i]

            # ResBlocks at current resolution
            for j in range(self.num_res_blocks):
                h = ResBlock(ch=ch, dropout=self.dropout)(h, train=train)

            # Upsample (except for last iteration which is already at 512×512)
            if i > 0:
                target_ch = self.ch_mults[i - 1]
                h = Up(ch=target_ch)(h)
            # After i=3: 64×64 → stays 64×64
            # After i=2: 64×64 → 128×128
            # After i=1: 128×128 → 256×256
            # After i=0: 256×256 → 512×512

        # Final layers: GroupNorm → Swish → Conv → Sigmoid
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME', name='conv_out')(h)
        h = nn.sigmoid(h)  # Output in [0, 1]

        return h


class SepVAE(nn.Module):
    """
    Complete SepVAE model with frozen backbone, three heads, and shared decoder.

    This is the main model that combines:
    - SepVAEEncoder: ResNet-50 backbone + 3 Conv heads → 64×64×8
    - Head nulling logic
    - SepVAEDecoder: Progressive upsampling decoder

    Spatial latents: 64×64×8 for LDM training
    - Common: 4 channels
    - Cardiomegaly: 2 channels
    - Pleural Effusion: 2 channels

    Attributes:
        z_channels_common: Common latent channels (default: 4)
        z_channels_disease: Disease-specific latent channels (default: 2)
        frozen_backbone: Whether to freeze backbone (default: True)
    """
    z_channels_common: int = 4
    z_channels_disease: int = 2
    frozen_backbone: bool = True

    def setup(self):
        """Initialize encoder and decoder."""
        self.encoder = SepVAEEncoder(
            z_channels_common=self.z_channels_common,
            z_channels_disease=self.z_channels_disease,
            frozen_backbone=self.frozen_backbone
        )
        self.decoder = SepVAEDecoder(
            z_channels=self.z_channels_common + 2 * self.z_channels_disease
        )

    def __call__(self, x, labels, *, key, train: bool = True):
        """
        Full forward pass: encode → null → decode.

        Args:
            x: Input images (B, 512, 512, 1) in [-1, 1] range
            labels: Disease labels (B,) in {0: Normal, 1: Effusion, 2: Cardio}
            key: JAX PRNG key for sampling
            train: Training mode

        Returns:
            x_rec: Reconstructed images (B, 512, 512, 1) in [0, 1] range
            latents_dict: Dict with encoder outputs (mu, logvar) for each head
                         Each is spatial: (B, 64, 64, channels)
            inactive_mus: Dict with masked inactive means for nulling loss
        """
        # Encode (get spatial mu and logvar for all three heads)
        latents_dict = self.encoder(x, train=train)

        # Sample spatial latents with head nulling
        key_sample, key_dec = jax.random.split(key)
        z_concat, inactive_mus = apply_head_nulling(latents_dict, labels, key_sample)
        # z_concat: (B, 64, 64, 8)

        # Decode
        x_rec = self.decoder(z_concat, train=train)

        return x_rec, latents_dict, inactive_mus

    def encode(self, x):
        """
        Encode images to spatial latent distributions (no sampling).

        Args:
            x: Input images (B, 512, 512, 1)

        Returns:
            latents_dict: Dict with (mu, logvar) for each head
                         Each is spatial: (B, 64, 64, channels)
        """
        return self.encoder(x, train=False)

    def extract_backbone_features(self, x):
        """
        Extract raw spatial features from the frozen CheSS backbone.

        Args:
            x: Input images (B, 512, 512, 1)

        Returns:
            Spatial features (B, 64, 64, 2048) before ConvHeads
        """
        h = self.encoder.backbone(x, return_spatial=True)
        if self.frozen_backbone:
            h = jax.lax.stop_gradient(h)
        return h

    def decode(self, z):
        """
        Decode spatial latents to images.

        Args:
            z: Spatial latents (B, 64, 64, 8)

        Returns:
            Reconstructed images (B, 512, 512, 1) in [0, 1]
        """
        return self.decoder(z, train=False)


# Testing utilities
def test_sepvae():
    """Test SepVAE model with dummy inputs."""
    print("Testing SepVAE model (Spatial Latents)...")

    # Create model
    model = SepVAE(
        z_channels_common=4,
        z_channels_disease=2,
        frozen_backbone=True
    )

    # Dummy inputs
    batch_size = 2
    x = jnp.ones((batch_size, 512, 512, 1)) * 0.5  # Mid-gray images
    labels = jnp.array([0, 1])  # Normal and Effusion

    # Initialize
    key = jax.random.PRNGKey(0)
    key_init, key_forward = jax.random.split(key)

    print("Initializing model...")
    variables = model.init({'params': key_init, 'dropout': key_init}, x, labels, key=key_forward, train=True)

    print(f"Model parameters: {sum(p.size for p in jax.tree_util.tree_leaves(variables['params'])):,}")

    # Forward pass
    print("Running forward pass...")
    x_rec, latents_dict, inactive_mus = model.apply(
        variables, x, labels, key=key_forward, train=True
    )

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {x_rec.shape}")
    print(f"Output range: [{x_rec.min():.3f}, {x_rec.max():.3f}]")

    print("\nSpatial Latent dimensions (64×64×channels):")
    for head_name, (mu, logvar) in latents_dict.items():
        print(f"  {head_name}: μ{mu.shape}, log_σ{logvar.shape}")

    print("\nTotal latent space: 64×64×8 (ready for LDM)")

    print("\nInactive means (should be zero for active heads):")
    for head_name, inactive_mu in inactive_mus.items():
        # Sum over spatial and channel dims
        inactive_sum = jnp.sum(inactive_mu, axis=(1, 2, 3))
        print(f"  {head_name}: {inactive_sum}")

    print("\n✓ SepVAE test passed!")


if __name__ == "__main__":
    test_sepvae()
