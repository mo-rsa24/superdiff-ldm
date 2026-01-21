from typing import List

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from notebooks.utils import get_text_embedding


@torch.no_grad
def get_vel(unet, t, sigma, latents, embeddings, eps=None, get_div=False, device=torch.device("cuda"), dtype=torch.float16):
    # unet.set_attn_processor(AttnProcessor())
    t = t.to(device, dtype=torch.float16)

    def v(_x, _e):
        _x = _x.to(device=device, dtype=dtype)
        _e = _e.to(device=device, dtype=dtype)

        denom = torch.sqrt(sigma * sigma + 1.0)  # stays fp16 now
        x_in = _x / denom

        with torch.autocast("cuda", dtype=dtype):
            return unet(x_in, t, encoder_hidden_states=_e).sample
    # v = lambda _x, _e: unet(_x / ((sigma**2 + 1) ** 0.5), t, encoder_hidden_states=_e).sample
    embeds = torch.cat(embeddings)
    latent_input = latents
    if get_div:
        with torch.enable_grad():
            with sdpa_kernel(SDPBackend.MATH):
                vel, div = torch.func.jvp(v, (latent_input, embeds), (eps, torch.zeros_like(embeds)))
                div = -(eps*div).sum((1,2,3))
    else:
        with torch.no_grad():
            vel = v(latent_input, embeds)
            div = None

    return vel, div

def get_latents(unet, scheduler, z_channels: int =4, device = torch.device("cuda"), dtype = torch.float16, height: int = 512, width: int = 512, num_inference_steps: int = 500, batch_size: int = 6):
    generator = torch.cuda.manual_seed(1)
    latents = torch.randn(
        (batch_size, z_channels, height // 4, width // 4),
        generator=generator,
        device=device,
        dtype=dtype
    )
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    return latents