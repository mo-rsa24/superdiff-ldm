"""
Multi-head SepVAE with frozen CheSS ResNet-50 backbone.

This module implements a Separated Variational Autoencoder (SepVAE) with:
- Frozen CheSS ResNet-50 backbone for feature extraction
- Optional Feature Pyramid Network (FPN) for multi-scale features
- Three parallel Conv heads for disentangled latent spaces:
  * Common head (64×64×4): Shared anatomy features
  * Cardiomegaly head (64×64×2): Disease-specific features
  * Pleural Effusion head (64×64×2): Disease-specific features
- Head nulling logic to prevent information leakage
- Shared decoder for reconstruction
- Optional partial backbone unfreezing (layer3/layer4)

Architecture (with FPN enabled):
    Input (512×512×1) → ResNet50 (frozen/partial)
                         ↓ multi-scale features
                    FPN fusion (64×64×512)
                         ↓
      ┌──────────────────┼──────────────────┐
      ↓                  ↓                  ↓
 Common Head       Cardio Head       Effusion Head
  (64×64×4)         (64×64×2)          (64×64×2)
      ↓                  ↓                  ↓
      └──────────────────┬──────────────────┘
                         ↓
                  Head Nulling
                         ↓
                Concat latents (64×64×8)
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


class FPN(nn.Module):
    """
    Feature Pyramid Network for fusing multi-scale backbone features.

    Combines layer2 (64×64×512), layer3 (32×32×1024), and layer4 (16×16×2048)
    into a single feature map at 64×64 resolution via top-down pathway with
    lateral connections.

    This addresses the spatial resolution bottleneck where bilinear upsampling
    of layer4 alone (16×16→64×64) loses fine-grained spatial information.

    Attributes:
        out_channels: Number of output channels (default: 512)
    """
    out_channels: int = 512

    @nn.compact
    def __call__(self, features: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Fuse multi-scale features via FPN.

        Args:
            features: Dict with keys:
                'layer2': (B, 64, 64, 512)
                'layer3': (B, 32, 32, 1024)
                'layer4': (B, 16, 16, 2048)

        Returns:
            Fused features (B, 64, 64, out_channels)
        """
        # Lateral connections: 1×1 conv to project each scale to common channels
        lat2 = nn.Conv(self.out_channels, (1, 1), name='lateral_layer2')(features['layer2'])
        lat3 = nn.Conv(self.out_channels, (1, 1), name='lateral_layer3')(features['layer3'])
        lat4 = nn.Conv(self.out_channels, (1, 1), name='lateral_layer4')(features['layer4'])

        # Top-down pathway: upsample deeper features and add to shallower
        # Layer4 (16×16) → upsample to 32×32 → add to layer3
        B = lat4.shape[0]
        lat4_up = jax.image.resize(lat4, (B, 32, 32, self.out_channels), method='bilinear')
        p3 = lat3 + lat4_up

        # P3 (32×32) → upsample to 64×64 → add to layer2
        p3_up = jax.image.resize(p3, (B, 64, 64, self.out_channels), method='bilinear')
        p2 = lat2 + p3_up

        # Smooth with 3×3 conv to reduce aliasing from upsampling
        fused = nn.Conv(self.out_channels, (3, 3), padding='SAME', name='smooth')(p2)

        return fused  # (B, 64, 64, out_channels)


class ConvHead(nn.Module):
    """
    Convolutional head that projects spatial features to latent μ and log_σ.

    Architecture:
        h(64×64×C_in) → Conv(3×3, hidden_ch) → GroupNorm → ReLU → Conv(3×3, out_ch*2)
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

    Supports two modes:
    - Legacy (use_fpn=False): Single-scale features from layer4 (bilinear 16→64)
    - FPN (use_fpn=True): Multi-scale features fused via Feature Pyramid Network

    Supports partial unfreezing:
    - frozen_backbone=True, unfreeze_from=None: Fully frozen (default)
    - frozen_backbone=True, unfreeze_from='layer3': Unfreeze layer3+layer4
    - frozen_backbone=True, unfreeze_from='layer4': Unfreeze layer4 only
    - frozen_backbone=False: Nothing frozen

    Attributes:
        z_channels_common: Common latent channels (default: 4)
        z_channels_disease: Disease-specific latent channels (default: 2)
        frozen_backbone: Whether to freeze backbone gradients (default: True)
        use_fpn: Whether to use Feature Pyramid Network (default: False)
        fpn_channels: FPN output channels (default: 512)
        unfreeze_from: Layer name to unfreeze from ('layer3' or 'layer4', default: None)
    """
    z_channels_common: int = 4
    z_channels_disease: int = 2
    frozen_backbone: bool = True
    use_fpn: bool = False
    fpn_channels: int = 512
    unfreeze_from: str = None

    def setup(self):
        """Initialize backbone, optional FPN, and three Conv heads."""
        self.backbone = ResNet50CheSS()
        if self.use_fpn:
            self.fpn = FPN(out_channels=self.fpn_channels)
        self.head_common = ConvHead(out_channels=self.z_channels_common, name='head_common')
        self.head_cardiomegaly = ConvHead(out_channels=self.z_channels_disease, name='head_cardiomegaly')
        self.head_effusion = ConvHead(out_channels=self.z_channels_disease, name='head_effusion')

    def _apply_selective_freeze(self, features: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Apply stop_gradient selectively based on unfreeze_from setting.

        When unfreeze_from='layer3': freeze layer2, unfreeze layer3+layer4
        When unfreeze_from='layer4': freeze layer2+layer3, unfreeze layer4
        When unfreeze_from=None: freeze everything
        """
        if not self.frozen_backbone or self.unfreeze_from is None:
            if self.frozen_backbone:
                # Fully frozen
                return jax.tree_util.tree_map(jax.lax.stop_gradient, features)
            else:
                # Nothing frozen
                return features

        frozen = {}
        if self.unfreeze_from == 'layer3':
            frozen['layer2'] = jax.lax.stop_gradient(features['layer2'])
            frozen['layer3'] = features['layer3']  # Trainable
            frozen['layer4'] = features['layer4']  # Trainable
        elif self.unfreeze_from == 'layer4':
            frozen['layer2'] = jax.lax.stop_gradient(features['layer2'])
            frozen['layer3'] = jax.lax.stop_gradient(features['layer3'])
            frozen['layer4'] = features['layer4']  # Trainable
        else:
            frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, features)
        return frozen

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
        if self.use_fpn:
            # Multi-scale feature extraction
            features = self.backbone(x, return_multiscale=True)
            features = self._apply_selective_freeze(features)
            h = self.fpn(features)  # (B, 64, 64, fpn_channels)
        else:
            # Legacy: single-scale features from layer4 upsampled to 64×64
            h = self.backbone(x, return_spatial=True)
            if self.frozen_backbone:
                h = jax.lax.stop_gradient(h)

        # Three parallel Conv heads
        mu_c, logvar_c = self.head_common(h)
        mu_cardio, logvar_cardio = self.head_cardiomegaly(h)
        mu_effusion, logvar_effusion = self.head_effusion(h)

        return {
            'common': (mu_c, logvar_c),
            'cardiomegaly': (mu_cardio, logvar_cardio),
            'effusion': (mu_effusion, logvar_effusion),
        }


def apply_head_nulling(
    latents_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
    labels: jnp.ndarray,
    key: jax.random.PRNGKey,
    sigma_inactive: float = 0.1,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Apply disease-specific soft head nulling based on labels (spatial version).

    Soft nulling replaces inactive disease channels with samples from the
    prior N(0, sigma_inactive²) instead of hard zeros. This:
    - Produces a continuous latent distribution (no delta at zero)
    - Is consistent with the conditional KL prior
    - Handles label noise gracefully
    - Makes downstream LDM training viable (smooth marginal q(z))

    Nulling logic:
    - label=0 (Normal): Both disease heads → sample from prior
    - label=1 (Pleural Effusion): Cardiomegaly head → sample from prior
    - label=2 (Cardiomegaly): Pleural Effusion head → sample from prior

    Args:
        latents_dict: Dict with keys 'common', 'cardiomegaly', 'effusion'
                     Each value is tuple (mu, logvar) with shape (B, 64, 64, channels)
        labels: Disease labels (B,) with values in {0, 1, 2}
        key: JAX PRNG key for sampling
        sigma_inactive: Std dev for inactive channel prior (default: 0.1)

    Returns:
        z_concat: Concatenated spatial latents (B, 64, 64, 8) with soft nulling
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

    key_cardio, key_effusion, key_noise_c, key_noise_e = jax.random.split(key, 4)
    eps_cardio = jax.random.normal(key_cardio, mu_cardio.shape)
    eps_effusion = jax.random.normal(key_effusion, mu_effusion.shape)

    z_cardio = mu_cardio + jnp.exp(0.5 * logvar_cardio) * eps_cardio
    z_effusion = mu_effusion + jnp.exp(0.5 * logvar_effusion) * eps_effusion

    # Create nulling masks for spatial tensors
    # Cardiomegaly head is active only when label=2
    mask_cardio = jnp.where(labels == 2, 1.0, 0.0)[:, None, None, None]  # (B, 1, 1, 1)

    # Effusion head is active only when label=1
    mask_effusion = jnp.where(labels == 1, 1.0, 0.0)[:, None, None, None]  # (B, 1, 1, 1)

    # Soft nulling: inactive channels get samples from prior N(0, sigma_inactive²)
    # instead of hard zeros. This maintains a continuous latent distribution.
    noise_cardio = sigma_inactive * jax.random.normal(key_noise_c, z_cardio.shape)
    noise_effusion = sigma_inactive * jax.random.normal(key_noise_e, z_effusion.shape)

    z_cardio_masked = z_cardio * mask_cardio + noise_cardio * (1.0 - mask_cardio)
    z_effusion_masked = z_effusion * mask_effusion + noise_effusion * (1.0 - mask_effusion)

    # Concatenate along channel dimension: [common(4), cardio(2), effusion(2)] = 8 channels
    z_concat = jnp.concatenate([z_c, z_cardio_masked, z_effusion_masked], axis=-1)  # (B, 64, 64, 8)

    # Track inactive means for nulling loss (still penalize encoder mu toward 0)
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
        ch_mults: Channel multipliers indexed low→high res (default: 32, 64, 128, 256)
                  reversed() iteration gives 256ch@64×64, 128ch@128×128, 64ch@256×256, 32ch@512×512
        num_res_blocks: ResBlocks per stage (default: 2)
        dropout: Dropout rate (default: 0.0)
        z_channels: Total latent channels (4 + 2 + 2 = 8)
    """
    ch_mults: Sequence[int] = (32, 64, 128, 256)  # 4 stages: 64(256ch)→128(128ch)→256(64ch)→512(32ch)
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
        # Initial processing: lightweight projection from z_channels
        h = nn.Conv(self.ch_mults[0], kernel_size=(3, 3), padding='SAME', name='z_proj')(z)
        # h is now (B, 64, 64, 32); first ResBlock expands to ch_mults[-1]=256

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
            # i=3: 256ch @ 64×64  → Up(128) → 128×128
            # i=2: 128ch @ 128×128 → Up(64) → 256×256
            # i=1: 64ch  @ 256×256 → Up(32) → 512×512
            # i=0: 32ch  @ 512×512 → no Up

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
    - SepVAEEncoder: ResNet-50 backbone + optional FPN + 3 Conv heads → 64×64×8
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
        use_fpn: Whether to use Feature Pyramid Network (default: False)
        fpn_channels: FPN output channels (default: 512)
        unfreeze_from: Layer to unfreeze from ('layer3', 'layer4', or None)
        sigma_inactive: Std dev for soft nulling prior on inactive channels (default: 0.1)
    """
    z_channels_common: int = 4
    z_channels_disease: int = 2
    frozen_backbone: bool = True
    use_fpn: bool = False
    fpn_channels: int = 512
    unfreeze_from: str = None
    sigma_inactive: float = 0.1

    def setup(self):
        """Initialize encoder and decoder."""
        self.encoder = SepVAEEncoder(
            z_channels_common=self.z_channels_common,
            z_channels_disease=self.z_channels_disease,
            frozen_backbone=self.frozen_backbone,
            use_fpn=self.use_fpn,
            fpn_channels=self.fpn_channels,
            unfreeze_from=self.unfreeze_from,
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
            inactive_mus: Dict with masked inactive means for nulling loss
        """
        # Encode
        latents_dict = self.encoder(x, train=train)

        # Sample spatial latents with head nulling
        key_sample, key_dec = jax.random.split(key)
        z_concat, inactive_mus = apply_head_nulling(
            latents_dict, labels, key_sample, sigma_inactive=self.sigma_inactive
        )

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

    def extract_multiscale_features(self, x):
        """
        Extract multi-scale features from the backbone (for perceptual loss).

        Args:
            x: Input images (B, 512, 512, 1) in [-1, 1] range

        Returns:
            Dict with 'layer2', 'layer3', 'layer4' feature maps
        """
        features = self.encoder.backbone(x, return_multiscale=True)
        # Always stop gradient for perceptual loss (feature extractor only)
        return jax.tree_util.tree_map(jax.lax.stop_gradient, features)

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

    for use_fpn in [False, True]:
        print(f"\n--- use_fpn={use_fpn} ---")
        model = SepVAE(
            z_channels_common=4,
            z_channels_disease=2,
            frozen_backbone=True,
            use_fpn=use_fpn,
            unfreeze_from='layer3' if use_fpn else None,
        )

        batch_size = 2
        x = jnp.ones((batch_size, 512, 512, 1)) * 0.5
        labels = jnp.array([0, 1])

        key = jax.random.PRNGKey(0)
        key_init, key_forward = jax.random.split(key)

        print("Initializing model...")
        variables = model.init({'params': key_init, 'dropout': key_init}, x, labels, key=key_forward, train=True)

        print(f"Model parameters: {sum(p.size for p in jax.tree_util.tree_leaves(variables['params'])):,}")

        print("Running forward pass...")
        x_rec, latents_dict, inactive_mus = model.apply(
            variables, x, labels, key=key_forward, train=True
        )

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {x_rec.shape}")
        print(f"Output range: [{x_rec.min():.3f}, {x_rec.max():.3f}]")

        print("Spatial Latent dimensions (64×64×channels):")
        for head_name, (mu, logvar) in latents_dict.items():
            print(f"  {head_name}: μ{mu.shape}, log_σ{logvar.shape}")

    print("\n✓ SepVAE test passed!")


if __name__ == "__main__":
    test_sepvae()
