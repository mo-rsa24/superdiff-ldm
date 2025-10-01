from typing import Any, Callable, Sequence, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass

# ---------------- Utils ----------------

@dataclass
class DiagonalGaussian:
    mean: jnp.ndarray
    logvar: jnp.ndarray  # log(σ^2)

    def sample(self, key):
        eps = jax.random.normal(key, self.mean.shape)
        return self.mean + jnp.exp(0.5 * self.logvar) * eps

    def mode(self):
        return self.mean

    def kl(self):
        # KL(q||N(0,1)) = 0.5 * sum(μ^2 + σ^2 - 1 - logσ^2) per-element
        return 0.5 * jnp.sum(
            jnp.square(self.mean) + jnp.exp(self.logvar) - 1.0 - self.logvar,
            axis=tuple(range(1, self.mean.ndim))
        )

# --------------- Building blocks ---------------

class ResBlock(nn.Module):
    ch: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train=True):
        h = nn.GroupNorm(num_groups=32)(x)
        h = nn.swish(h)
        h = nn.Conv(self.ch, (3,3), padding="SAME")(h)
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        if self.dropout > 0 and train:
            h = nn.Dropout(self.dropout)(h, deterministic=not train)
        h = nn.Conv(self.ch, (3,3), padding="SAME")(h)
        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1,1))(x)
        return x + h

class Down(nn.Module):
    ch: int
    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.ch, (3,3), strides=(2,2), padding="SAME")(x)

class Up(nn.Module):
    ch: int
    @nn.compact
    def __call__(self, x):
        B,H,W,C = x.shape
        x = jax.image.resize(x, (B, H*2, W*2, C), method="nearest")
        return nn.Conv(self.ch, (3,3), padding="SAME")(x)

# --------------- Encoder/Decoder ---------------
class SelfAttention2D(nn.Module):
    num_heads: int = 1 # A single head is often sufficient for the AE
    @nn.compact
    def __call__(self, x):
        B,H,W,C = x.shape
        h = nn.GroupNorm(num_groups=32)(x) # Use GroupNorm instead of LayerNorm
        h = h.reshape((B, H*W, C))
        h = nn.SelfAttention(num_heads=self.num_heads, qkv_features=C, out_features=C)(h)
        h = h.reshape((B,H,W,C))
        return x + h

class Encoder(nn.Module):
    ch_mults: Sequence[int] = (128, 256, 512, 512)
    in_ch: int = 1
    z_ch: int = 4
    num_res_blocks: int = 2
    dropout: float = 0.0
    double_z: bool = True
    attn_resolutions: Sequence[int] = () # Add this argument

    @nn.compact
    def __call__(self, x, train=True):
        h = nn.Conv(self.ch_mults[0], (3,3), padding="SAME")(x)
        current_res = x.shape[1]

        for i, ch in enumerate(self.ch_mults):
            for _ in range(self.num_res_blocks):
                h = ResBlock(ch, self.dropout)(h, train=train)
                if current_res in self.attn_resolutions:
                    h = SelfAttention2D()(h) # Apply attention
            if i < len(self.ch_mults)-1:
                h = Down(self.ch_mults[i+1])(h)
                current_res //= 2 # Update resolution

        h = ResBlock(self.ch_mults[-1], self.dropout)(h, train=train)
        if current_res in self.attn_resolutions:
             h = SelfAttention2D()(h) # Apply attention in bottleneck

        out_ch = (2*self.z_ch) if self.double_z else self.z_ch
        h = nn.Conv(out_ch, (3,3), padding="SAME")(h)
        return h


class Decoder(nn.Module):
    ch_mults: Sequence[int] = (128, 256, 512, 512)
    out_ch: int = 1
    z_ch: int = 4
    num_res_blocks: int = 2
    dropout: float = 0.0
    attn_resolutions: Sequence[int] = ()  # Add this argument

    @nn.compact
    def __call__(self, z, train=True):
        h = nn.Conv(self.ch_mults[-1], (3, 3), padding="SAME")(z)
        current_res = z.shape[1]

        h = ResBlock(self.ch_mults[-1], self.dropout)(h, train=train)
        if current_res in self.attn_resolutions:
            h = SelfAttention2D()(h)  # Apply attention in bottleneck

        for i in reversed(range(len(self.ch_mults))):
            if i > 0:
                h = Up(self.ch_mults[i - 1])(h)
                current_res *= 2  # Update resolution

            for _ in range(self.num_res_blocks):
                h = ResBlock(self.ch_mults[i], self.dropout)(h, train=train)
                if current_res in self.attn_resolutions:
                    h = SelfAttention2D()(h)  # Apply attention

        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_ch, (3, 3), padding="SAME")(h)
        h = nn.sigmoid(h)
        return h

class AutoencoderKL(nn.Module):
    enc_cfg: dict
    dec_cfg: dict
    embed_dim: int
    def setup(self):
        enc_z = self.enc_cfg.get("z_ch", self.embed_dim)
        dec_z = self.dec_cfg.get("z_ch", self.embed_dim)
        enc_kwargs = {k: v for k, v in self.enc_cfg.items() if k != "z_ch"}
        dec_kwargs = {k: v for k, v in self.dec_cfg.items() if k != "z_ch"}

        self.encoder = Encoder(z_ch=enc_z, **enc_kwargs)
        self.decoder = Decoder(z_ch=dec_z, **dec_kwargs)

    def __call__(self, x, *, rng=None, sample_posterior=True, train=True):
        moments = self.encoder(x, train=train)
        mu, logvar = jnp.split(moments, 2, axis=-1)
        q = DiagonalGaussian(mu, logvar)

        if sample_posterior:
            if rng is None:
                raise ValueError("RNG key must be provided when sampling from the posterior.")
            z = q.sample(rng)
        else:
            z = q.mode()

        xrec = self.decoder(z, train=train)
        return xrec, q

    def encode(self, x, *, train=True):
        moments = self.encoder(x, train=train)
        mu, logvar = jnp.split(moments, 2, axis=-1)
        return DiagonalGaussian(mu, logvar)

    def decode(self, z, *, train=True):
        return self.decoder(z, train=train)