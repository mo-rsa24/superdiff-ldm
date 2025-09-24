# models/cxr_unet.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, Sequence

class GaussianFourierProjection(nn.Module):
    embed_dim: int
    scale: float = 30.
    @nn.compact
    def __call__(self, x):
        W = self.param('W', jax.nn.initializers.normal(stddev=self.scale),
                       (self.embed_dim // 2,))
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

class DenseToMap(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)[:, None, None, :]

def _pick_gn_groups(C: int) -> int:
    g = min(32, C)
    while g > 1 and (C % g) != 0:
        g //= 2
    return max(1, g)

class ResBlock(nn.Module):
    c: int
    embed_dim: int
    scale_skip: bool = True

    @nn.compact
    def __call__(self, x, t_embed):
        act = nn.swish
        in_ch = x.shape[-1]
        h = nn.GroupNorm(num_groups=_pick_gn_groups(in_ch))(x)
        h = act(h)
        h = nn.Conv(self.c, (3, 3), padding='SAME', use_bias=False)(h)
        h = h + DenseToMap(self.c)(t_embed)
        h = nn.GroupNorm(num_groups=_pick_gn_groups(self.c))(h)
        h = act(h)
        h = nn.Conv(self.c, (3, 3), padding='SAME', use_bias=False)(h)
        if in_ch != self.c:
            x = nn.Conv(self.c, (1, 1), padding='SAME', use_bias=False, name='skip_proj')(x)
        if self.scale_skip:
            x = x * (1.0 / jnp.sqrt(2.0))
        return act(h + x)


class SelfAttention2D(nn.Module):
    num_heads: int = 4
    @nn.compact
    def __call__(self, x):
        B,H,W,C = x.shape
        h = nn.LayerNorm()(x)
        h = h.reshape((B, H*W, C))
        h = nn.SelfAttention(num_heads=self.num_heads)(h)
        h = h.reshape((B,H,W,C))
        return x + h

class ScoreNet(nn.Module):
    # MODIFIED: Default channels updated for 32x32 latent space input
    z_channels: int = 3
    channels: Sequence[int] = (128, 256, 512)
    embed_dim: int = 256
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,) # Apply attention at 16x16
    num_heads: int = 4

    @nn.compact
    def __call__(self, x, t):
        # NOTE: The marginal_prob_std function is now passed to the loss function,
        # not stored in the model, to keep the model definition clean.
        act = nn.swish
        # Time embedding
        temb = act(nn.Dense(self.embed_dim)(GaussianFourierProjection(self.embed_dim)(t)))

        # --- Encoder ---
        # Initial projection
        h = nn.Conv(self.channels[0], (3,3), padding='SAME')(x)
        skips = [h]
        # Downsampling blocks
        for i, ch in enumerate(self.channels):
            for _ in range(self.num_res_blocks):
                h = ResBlock(ch, self.embed_dim)(h, temb)
                if h.shape[1] in self.attn_resolutions:
                    h = SelfAttention2D(num_heads=self.num_heads)(h)
                skips.append(h)
            if i < len(self.channels) - 1:
                h = nn.Conv(self.channels[i+1], (3,3), strides=(2,2), padding='SAME')(h)
                skips.append(h)

        # --- Bottleneck ---
        h = ResBlock(self.channels[-1], self.embed_dim)(h, temb)
        h = SelfAttention2D(num_heads=self.num_heads)(h)
        h = ResBlock(self.channels[-1], self.embed_dim)(h, temb)

        # --- Decoder ---
        for i in reversed(range(len(self.channels))):
            for _ in range(self.num_res_blocks + 1): # +1 for skip connections
                h = jnp.concatenate([h, skips.pop()], axis=-1)
                h = ResBlock(self.channels[i], self.embed_dim)(h, temb)
                if h.shape[1] in self.attn_resolutions:
                    h = SelfAttention2D(num_heads=self.num_heads)(h)
            if i > 0:
                h = nn.ConvTranspose(self.channels[i-1], (4,4), strides=(2,2), padding='SAME')(h)

        # Final projection
        h = nn.GroupNorm(num_groups=_pick_gn_groups(h.shape[-1]))(h)
        h = act(h)
        # MODIFIED: Output has z_channels (3) instead of 1
        out = nn.Conv(self.z_channels, (3,3), padding='SAME',
                      kernel_init=nn.initializers.zeros)(h)
        return out