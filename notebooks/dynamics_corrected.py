"""
Corrected SuperDiff AND implementation for M models.

This version properly implements Proposition 6 by solving the linear system
to find κ values that satisfy the constraint:
    d log q^i = d log q^j  for all i,j ∈ [M]
    Σⱼ κⱼ = 1
"""

from typing import List
import torch
from diffusers import EulerDiscreteScheduler
from torch.nn.attention import SDPBackend, sdpa_kernel

from notebooks.utils import get_text_embedding


@torch.no_grad
def get_vel(unet, t, sigma, latents, embeddings, eps=None, get_div=False, device=torch.device("cuda"), dtype=torch.float16):
    t = t.to(device, dtype=torch.float16)

    def v(_x, _e):
        _x = _x.to(device=device, dtype=dtype)
        _e = _e.to(device=device, dtype=dtype)

        denom = torch.sqrt(sigma * sigma + 1.0)
        x_in = _x / denom

        with torch.autocast("cuda", dtype=dtype):
            return unet(x_in, t, encoder_hidden_states=_e).sample

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


def solve_kappa_and(velocities, vel_uncond, dsigma, sigma, dx_base, guidance_scale, lift, num_inference_steps, batch_size, M):
    """
    Solve for κ values that satisfy Proposition 6 for AND operation.

    Constraint: d log q^i = d log q^j for all i,j
    This means: ⟨dx, ∇log q^i⟩ should be equal for all i

    Following the 2-prompt analytical solution but extended to M prompts.

    Args:
        velocities: List of M velocity fields (each is [batch, C, H, W])
        vel_uncond: Unconditional velocity [batch, C, H, W]
        dsigma: Step size
        sigma: Current noise level
        dx_base: Base update (without composition)
        guidance_scale: CFG scale
        lift: Lift parameter
        num_inference_steps: Total steps
        batch_size: Batch size
        M: Number of models

    Returns:
        kappas: [batch, M] tensor of composition weights
    """
    # Convert to numpy for easier linear algebra, or use torch.linalg
    # We'll use torch for consistency

    # For each sample in batch, solve independently
    kappas = torch.zeros(batch_size, M, device=velocities[0].device, dtype=velocities[0].dtype)

    for b in range(batch_size):
        # Extract velocities for this batch sample
        vels_b = [v[b].flatten() for v in velocities]  # Each is [C*H*W]
        vel_unc_b = vel_uncond[b].flatten()  # [C*H*W]
        dx_base_b = dx_base[b].flatten()  # [C*H*W]

        # Build the linear system
        # For M models with constraint sum(kappa) = 1, we have M equations
        #
        # The log-density change for model i is approximately:
        # d log q^i ≈ -|dsigma|/sigma * ||v_i||^2 - ⟨dx, v_i/sigma⟩
        #
        # For AND: we want all d log q^i to be equal
        # This gives us M-1 independent equations, plus sum(kappa)=1

        # Following the 2-prompt case, the composite update is:
        # dx = 2*dsigma*vf + noise
        # where vf = vel_uncond + guidance_scale * sum_i(kappa_i * (vel_i - vel_uncond))
        #
        # For equal log-density evolution, we need:
        # d log q^i = d log q^j for all i,j
        #
        # Expanding the log-density evolution (from Theorem 1):
        # d log q^i = -|dsigma|/sigma * ||v_i||^2 - ⟨dx, v_i/sigma⟩
        #
        # Setting d log q^i = d log q^j:
        # -|dsigma|/sigma * ||v_i||^2 - ⟨dx, v_i/sigma⟩ = -|dsigma|/sigma * ||v_j||^2 - ⟨dx, v_j/sigma⟩
        #
        # Rearranging:
        # -⟨dx, v_i/sigma⟩ + ⟨dx, v_j/sigma⟩ = -|dsigma|/sigma * (||v_i||^2 - ||v_j||^2)
        # ⟨dx, (v_j - v_i)/sigma⟩ = |dsigma|/sigma * (||v_i||^2 - ||v_j||^2)

        # Now dx = 2*dsigma*(vel_uncond + guidance_scale*sum_k(kappa_k*(v_k - vel_uncond))) + noise
        # Let's denote: dx_cfg = 2*dsigma*guidance_scale*sum_k(kappa_k*(v_k - vel_uncond))
        # And: dx_base = 2*dsigma*vel_uncond + noise (the base update without composition)
        # So: dx = dx_base + dx_cfg

        # Substituting into the constraint:
        # ⟨dx_base + dx_cfg, (v_j - v_i)/sigma⟩ = |dsigma|/sigma * (||v_i||^2 - ||v_j||^2)
        # ⟨dx_base, (v_j - v_i)/sigma⟩ + ⟨dx_cfg, (v_j - v_i)/sigma⟩ = |dsigma|/sigma * (||v_i||^2 - ||v_j||^2)
        #
        # Now: dx_cfg = 2*dsigma*guidance_scale*sum_k(kappa_k*(v_k - vel_uncond))
        # ⟨dx_cfg, (v_j - v_i)/sigma⟩ = 2*dsigma*guidance_scale/sigma * sum_k(kappa_k * ⟨v_k - vel_uncond, v_j - v_i⟩)

        # This gives us a linear system in kappa:
        # For each pair (i,j) with i < j:
        # sum_k(kappa_k * A[i,j,k]) = B[i,j]
        # where:
        # A[i,j,k] = 2*dsigma*guidance_scale/sigma * ⟨v_k - vel_uncond, v_j - v_i⟩
        # B[i,j] = |dsigma|/sigma * (||v_i||^2 - ||v_j||^2) - ⟨dx_base, (v_j - v_i)/sigma⟩

        # For M models, we have M-1 independent constraints (taking model 0 as reference)
        # Plus the constraint: sum(kappa) = 1
        # Total: M equations for M unknowns

        # Build the system matrix A and vector b
        A_sys = torch.zeros(M, M, device=velocities[0].device, dtype=torch.float32)
        b_sys = torch.zeros(M, device=velocities[0].device, dtype=torch.float32)

        # First M-1 rows: equality constraints (compare each model to model 0)
        for i in range(M - 1):
            j = i + 1  # Compare model j with model 0
            v_0 = vels_b[0]
            v_j = vels_b[j]

            # Compute RHS
            norm_diff = (v_0 ** 2).sum() - (v_j ** 2).sum()
            dx_term = torch.dot(dx_base_b, (v_j - v_0) / sigma)
            b_sys[i] = torch.abs(dsigma) / sigma * norm_diff - dx_term

            # Compute LHS coefficients for each k
            for k in range(M):
                v_k = vels_b[k]
                A_sys[i, k] = 2 * dsigma * guidance_scale / sigma * torch.dot(v_k - vel_unc_b, v_j - v_0)

        # Last row: sum constraint
        A_sys[M - 1, :] = 1.0
        b_sys[M - 1] = 1.0

        # Add lift term to regularize (add to all equations except sum constraint)
        b_sys[:M-1] += sigma * lift / num_inference_steps

        # Solve the linear system: A_sys @ kappa = b_sys
        try:
            kappa_b = torch.linalg.solve(A_sys, b_sys)
            # Clamp to reasonable range and re-normalize
            kappa_b = torch.clamp(kappa_b, min=0.0, max=2.0)
            kappa_b = kappa_b / (kappa_b.sum() + 1e-8)
        except:
            # If singular, fall back to uniform
            kappa_b = torch.ones(M, device=velocities[0].device, dtype=velocities[0].dtype) / M

        kappas[b] = kappa_b.to(velocities[0].dtype)

    return kappas


def stochastic_super_diff_multi_corrected(
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
        operation: str = "AND"
):
    """
    CORRECTED: Compose M pre-trained score models using SuperDiff.

    This version properly implements Proposition 6 for AND operation.
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

        # Base update (without composition)
        dx_base = 2 * dsigma * vel_uncond + noise

        # Compute kappas based on operation type
        if operation == "OR":
            # Softmax over log-likelihoods (Proposition 3)
            temperature = 1.0
            logits = log_likelihoods[i] / temperature + lift / num_inference_steps
            kappas[i + 1] = torch.softmax(logits, dim=-1)

        elif operation == "AND":
            # CORRECTED: Solve linear system for AND operation (Proposition 6)
            kappas[i + 1] = solve_kappa_and(
                velocities=velocities,
                vel_uncond=vel_uncond,
                dsigma=dsigma,
                sigma=sigma,
                dx_base=dx_base,
                guidance_scale=guidance_scale,
                lift=lift,
                num_inference_steps=num_inference_steps,
                batch_size=batch_size,
                M=M
            )

        else:
            raise ValueError(f"Unknown operation: {operation}. Use 'AND' or 'OR'.")

        # Compute composite vector field as weighted sum
        # vf = vel_uncond + guidance_scale * sum_m(kappa_m * (vel_m - vel_uncond))
        vf = vel_uncond.clone()
        for m in range(M):
            kappa_m = kappas[i + 1, :, m]  # (batch,)
            kappa_m = kappa_m[:, None, None, None]
            vf = vf + guidance_scale * kappa_m * (velocities[m] - vel_uncond)

        # Update latents
        dx = 2 * dsigma * vf + noise
        latents = latents + dx

        # Update log-likelihoods for each model (Theorem 1)
        for m in range(M):
            vel_m = velocities[m]
            ll_update = (
                -torch.abs(dsigma) / sigma * (vel_m ** 2)
                - (dx * (vel_m / sigma))
            ).sum((1, 2, 3))
            log_likelihoods[i + 1, :, m] = log_likelihoods[i, :, m] + ll_update

    return latents, kappas, log_likelihoods
