from typing import List

import torch
from diffusers import EulerDiscreteScheduler
from torch.nn.attention import SDPBackend, sdpa_kernel

from notebooks.utils import get_text_embedding


@torch.no_grad
def get_vel(
    unet,
    t,
    sigma,
    latents,
    embeddings,
    eps=None,
    get_div=False,
    device=torch.device("cuda"),
    dtype=torch.float16,
    added_cond_kwargs=None,
):
    t = t.to(device, dtype=torch.float16)

    def v(_x, _e):
        _x = _x.to(device=device, dtype=dtype)
        _e = _e.to(device=device, dtype=dtype)

        denom = torch.sqrt(sigma * sigma + 1.0)  # stays fp16 now
        x_in = _x / denom

        with torch.autocast("cuda", dtype=dtype):
            if added_cond_kwargs is None:
                return unet(x_in, t, encoder_hidden_states=_e).sample

            cond_kwargs = {
                key: value.to(device=device, dtype=dtype)
                for key, value in added_cond_kwargs.items()
            }
            return unet(
                x_in,
                t,
                encoder_hidden_states=_e,
                added_cond_kwargs=cond_kwargs,
            ).sample
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

def get_latents(scheduler, z_channels: int =4, device = torch.device("cuda"), dtype = torch.float16,  num_inference_steps: int = 500, batch_size: int = 6, latent_width: int = 64, latent_height: int = 64, seed: int = None):
    # Use provided seed, or respect global seed if not provided
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None  # Use global random state (set by torch.manual_seed)

    latents = torch.randn(
        (batch_size, z_channels, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=dtype
    )
    scheduler.set_timesteps(num_inference_steps)
    # FlowMatchEulerDiscreteScheduler has no init_noise_sigma (starts from N(0,1))
    if hasattr(scheduler, 'init_noise_sigma'):
        latents = latents * scheduler.init_noise_sigma
    return latents


def stochastic_super_diff_and(
        latents,
        obj_prompt: List[str],
        bg_prompt: List[str],
        scheduler: EulerDiscreteScheduler,
        unet,
        tokenizer,
        text_encoder,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 100,
        batch_size: int = 4,
        device=torch.device("cuda"),
        dtype=torch.float16,
        lift: float = 0.0
):
    obj_embeddings = get_text_embedding(obj_prompt * batch_size, tokenizer, text_encoder, device)
    bg_embeddings = get_text_embedding(bg_prompt * batch_size, tokenizer, text_encoder, device)
    uncond_embeddings = get_text_embedding([""] * batch_size, tokenizer, text_encoder, device)

    ll_obj = torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=dtype)
    ll_bg = torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=dtype)
    kappa = 0.5 * torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=dtype)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]

        # Pass unet and dtype
        vel_obj, _ = get_vel(unet, t, sigma, latents, [obj_embeddings], device=device, dtype=dtype)
        vel_bg, _ = get_vel(unet, t, sigma, latents, [bg_embeddings], device=device, dtype=dtype)
        vel_uncond, _ = get_vel(unet, t, sigma, latents, [uncond_embeddings], device=device, dtype=dtype)

        noise = torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)

        # SuperDiff Logic
        dx_ind = 2 * dsigma * (vel_uncond + guidance_scale * (vel_bg - vel_uncond)) + noise

        # Terms for Kappa
        term1 = (torch.abs(dsigma) * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
        term2 = (dx_ind * (vel_obj - vel_bg)).sum((1, 2, 3))
        term3 = sigma * lift / num_inference_steps

        numerator = term1 - term2 + term3
        denominator = 2 * dsigma * guidance_scale * ((vel_obj - vel_bg) ** 2).sum((1, 2, 3))

        # Update Kappa with stability epsilon
        kappa[i + 1] = numerator / (denominator + 1e-8)

        # Composite Vector Field
        vf = vel_uncond + guidance_scale * (
                    (vel_bg - vel_uncond) + kappa[i + 1][:, None, None, None] * (vel_obj - vel_bg))

        dx = 2 * dsigma * vf + noise
        latents += dx

        ll_obj[i + 1] = ll_obj[i] + (-torch.abs(dsigma) / sigma * (vel_obj) ** 2 - (dx * (vel_obj / sigma))).sum(
            (1, 2, 3))
        ll_bg[i + 1] = ll_bg[i] + (-torch.abs(dsigma) / sigma * (vel_bg) ** 2 - (dx * (vel_bg / sigma))).sum((1, 2, 3))
    return latents, kappa, ll_obj, ll_bg


def _solve_kappa_and(velocities, vel_uncond, dsigma, sigma, noise,
                     guidance_scale, lift, num_inference_steps):
    """
    Solve the M+1 linear system from Proposition 6 for AND composition.

    Finds κ = [κ₁, ..., κₘ] with Σκ = 1 such that all log-densities
    evolve at the same rate:  d log qⁱ = d log qʲ  ∀ i,j ∈ [M].

    Derivation (generalising the 2-prompt analytical formula):
    ---------------------------------------------------------
    Composite update:
        dx = 2·dσ·(v_unc + gs·Σₖ κₖ(vₖ − v_unc)) + noise

    Split into κ-independent and κ-dependent parts:
        dx_base = 2·dσ·v_unc + noise          (independent of κ)
        dx      = dx_base + 2·dσ·gs·Σₖ κₖ(vₖ − v_unc)

    Itô density estimator (Theorem 1) for model i:
        d log qⁱ ≈ −|dσ|/σ · ‖vⁱ‖² − ⟨dx, vⁱ/σ⟩

    Setting d log q⁰ = d log qʲ  for j = 1, …, M−1 and
    substituting dx gives a linear equation in κ for each j:

        Σₖ κₖ · [2·dσ·gs · ⟨vₖ − v_unc, vⱼ − v₀⟩]
            = |dσ| · (‖v₀‖² − ‖vⱼ‖²) − ⟨dx_base, vⱼ − v₀⟩ − σ·ℓ/N

    Together with Σₖ κₖ = 1 this is an M × M system  A κ = b.

    For M = 2 this reduces to the closed-form in stochastic_super_diff_and.
    """
    M = len(velocities)
    B = velocities[0].shape[0]
    dev = velocities[0].device

    # Flatten spatial dims: [M, B, D]
    vels = torch.stack([v.flatten(1) for v in velocities])          # [M, B, D]
    v_unc = vel_uncond.flatten(1)                                    # [B, D]
    dx_base = (2 * dsigma * vel_uncond + noise).flatten(1)           # [B, D]

    # Differences needed for the linear system
    # u_diff[k] = vₖ − v_unc   (how each model differs from uncond)
    u_diff = vels - v_unc.unsqueeze(0)                               # [M, B, D]

    # v_diff[j] = v_{j+1} − v₀  (how each model differs from reference)
    v_diff = vels[1:] - vels[0:1]                                   # [M-1, B, D]

    # ---- Build A  [B, M, M] ----
    # Upper block [B, M-1, M]:
    #   A[b, j, k] = 2·dσ·gs · ⟨u_diff[k,b,:], v_diff[j,b,:]⟩
    # Efficiently via batched matmul:
    #   u_diff_t : [B, M, D],  v_diff_t : [B, M-1, D]
    #   upper = v_diff_t @ u_diff_t^T → [B, M-1, M]
    u_diff_t = u_diff.permute(1, 0, 2).float()                      # [B, M, D]
    v_diff_t = v_diff.permute(1, 0, 2).float()                      # [B, M-1, D]

    A = torch.zeros(B, M, M, device=dev, dtype=torch.float32)
    A[:, :M-1, :] = (2 * dsigma * guidance_scale) * torch.bmm(
        v_diff_t, u_diff_t.transpose(1, 2)
    )                                                                # [B, M-1, M]
    A[:, M-1, :] = 1.0                                              # sum constraint

    # ---- Build b  [B, M] ----
    b = torch.zeros(B, M, device=dev, dtype=torch.float32)

    # Per-model squared norms: [M, B]
    norms_sq = (vels.float() ** 2).sum(dim=2)                       # [M, B]

    # RHS for each constraint j = 0 … M-2  (comparing model j+1 to model 0)
    #   b[j] = |dσ|·(‖v₀‖² − ‖v_{j+1}‖²) − ⟨dx_base, v_{j+1} − v₀⟩ − σ·ℓ/N
    for j in range(M - 1):
        norm_term = torch.abs(dsigma) * (norms_sq[0] - norms_sq[j + 1])  # [B]
        dot_term = (dx_base.float() * v_diff_t[:, j, :]).sum(dim=1)       # [B]
        b[:, j] = norm_term - dot_term - sigma * lift / num_inference_steps

    b[:, M-1] = 1.0                                                 # sum constraint

    # ---- Solve  A κ = b ----
    # Use least-squares for robustness (handles near-singular cases)
    kappa = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)  # [B, M]

    return kappa.to(velocities[0].dtype)


def stochastic_super_diff_multi(
        latents,
        prompts: List[str],
        scheduler: EulerDiscreteScheduler,
        unet,
        tokenizer,
        text_encoder,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 100,
        batch_size: int = 4,
        device=torch.device("cuda"),
        dtype=torch.float16,
        lift: float = 0.0,
        operation: str = "AND"  # "AND" or "OR"
):
    """
    Compose M pre-trained score models using SuperDiff (Algorithm 1).

    For AND: solves the M+1 linear system from Proposition 6 so that
             all log-densities evolve at the same rate.
    For OR:  uses softmax over log-likelihoods (Proposition 3).

    Args:
        latents: Initial latent noise
        prompts: List of M prompts to compose
        scheduler: Diffusion scheduler
        unet: UNet model
        tokenizer: Text tokenizer
        text_encoder: Text encoder
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of diffusion steps
        batch_size: Batch size
        device: Device to run on
        dtype: Data type
        lift: Lift / bias parameter ℓ for stability (Eq. 18)
        operation: "AND" or "OR" composition (Algorithm 1)

    Returns:
        latents: Final denoised latents
        kappas: Tensor of shape (num_steps+1, batch_size, M) with composition weights
        log_likelihoods: Tensor of shape (num_steps+1, batch_size, M) with log-likelihoods
    """
    M = len(prompts)

    # Get embeddings for all prompts
    embeddings_list = [
        get_text_embedding([prompt] * batch_size, tokenizer, text_encoder, device)
        for prompt in prompts
    ]
    uncond_embeddings = get_text_embedding([""] * batch_size, tokenizer, text_encoder, device)

    # Initialize kappas and log-likelihoods for M models
    kappas = torch.zeros((num_inference_steps + 1, batch_size, M), device=device, dtype=dtype)
    kappas[0] = 1.0 / M  # Initialize with uniform weights

    log_likelihoods = torch.zeros((num_inference_steps + 1, batch_size, M), device=device, dtype=dtype)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]

        # Compute velocities for all M models
        velocities = []
        for emb in embeddings_list:
            vel, _ = get_vel(unet, t, sigma, latents, [emb], device=device, dtype=dtype)
            velocities.append(vel)

        # Compute unconditional velocity
        vel_uncond, _ = get_vel(unet, t, sigma, latents, [uncond_embeddings], device=device, dtype=dtype)

        # Generate noise
        noise = torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)

        # Compute kappas based on operation type
        if operation == "OR":
            # Softmax over T·log-likelihoods + ℓ (Proposition 3 in Algorithm 1)
            temperature = 1.0
            logits = log_likelihoods[i] / temperature + lift / num_inference_steps
            kappas[i + 1] = torch.softmax(logits, dim=-1)

        elif operation == "AND":
            # Solve M+1 linear system for κ (Proposition 6)
            kappas[i + 1] = _solve_kappa_and(
                velocities, vel_uncond, dsigma, sigma, noise,
                guidance_scale, lift, num_inference_steps,
            )

        else:
            raise ValueError(f"Unknown operation: {operation}. Use 'AND' or 'OR'.")

        # Composite vector field:  u_t = Σ_m κ_m · ∇log q_m  (Algorithm 1)
        # In CFG form: vf = v_unc + gs · Σ_m κ_m · (v_m − v_unc)
        vf = vel_uncond.clone()
        for m in range(M):
            kappa_m = kappas[i + 1, :, m][:, None, None, None]      # [B,1,1,1]
            vf = vf + guidance_scale * kappa_m * (velocities[m] - vel_uncond)

        # SDE step (Proposition 1):
        # dx_τ = (−f_{1−τ} + g²_{1−τ} u_t) dτ + g_{1−τ} dW̄_τ
        dx = 2 * dsigma * vf + noise
        latents = latents + dx

        # Update log-likelihoods for each model (Theorem 1, Eq. 13)
        for m in range(M):
            vel_m = velocities[m]
            ll_update = (
                -torch.abs(dsigma) / sigma * (vel_m ** 2)
                - (dx * (vel_m / sigma))
            ).sum((1, 2, 3))
            log_likelihoods[i + 1, :, m] = log_likelihoods[i, :, m] + ll_update

    return latents, kappas, log_likelihoods
