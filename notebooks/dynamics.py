from typing import List

import torch
from diffusers import EulerDiscreteScheduler
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

def get_latents(scheduler, z_channels: int =4, device = torch.device("cuda"), dtype = torch.float16, height: int = 512, width: int = 512, num_inference_steps: int = 500, batch_size: int = 6):
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

def superDiffAND(obj_prompt: List[str], bg_prompt: List[str], scheduler: EulerDiscreteScheduler, num_inference_steps: int = 512, batch_size: int = 6, device = torch.device("cuda")):
    obj_embeddings = get_text_embedding(obj_prompt * batch_size)
    bg_embeddings = get_text_embedding(bg_prompt * batch_size)
    uncond_embeddings = get_text_embedding([""] * batch_size)

    lift = 0.0
    ll_obj = torch.ones((num_inference_steps + 1, batch_size), device=device)
    ll_bg = torch.ones((num_inference_steps + 1, batch_size), device=device)
    kappa = 0.5 * torch.ones((num_inference_steps + 1, batch_size), device=device)
    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]
        vel_obj, _ = get_vel(t, sigma, latents, [obj_embeddings])
        vel_bg, _ = get_vel(t, sigma, latents, [bg_embeddings])
        vel_uncond, _ = get_vel(t, sigma, latents, [uncond_embeddings])

        noise = torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)
        dx_ind = 2 * dsigma * (vel_uncond + guidance_scale * (vel_bg - vel_uncond)) + noise
        kappa[i + 1] = (torch.abs(dsigma) * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3)) - (
                    dx_ind * ((vel_obj - vel_bg))).sum((1, 2, 3)) + sigma * lift / num_inference_steps
        kappa[i + 1] /= 2 * dsigma * guidance_scale * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))

        vf = vel_uncond + guidance_scale * (
                    (vel_bg - vel_uncond) + kappa[i + 1][:, None, None, None] * (vel_obj - vel_bg))
        dx = 2 * dsigma * vf + noise
        latents += dx

        ll_obj[i + 1] = ll_obj[i] + (-torch.abs(dsigma) / sigma * (vel_obj) ** 2 - (dx * (vel_obj / sigma))).sum(
            (1, 2, 3))
        ll_bg[i + 1] = ll_bg[i] + (-torch.abs(dsigma) / sigma * (vel_bg) ** 2 - (dx * (vel_bg / sigma))).sum((1, 2, 3))