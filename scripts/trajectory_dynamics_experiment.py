"""
Trajectory Dynamics Experiment — SD3.5 Medium

Compares reverse diffusion trajectories under different conditioning:
  1. Prompt A (e.g., "a dog")
  2. Prompt B (e.g., "a cat")
  3. CLIP Monolithic AND (e.g., "a dog and a cat")
  4. SuperDIFF AND (prompt A ∧ prompt B)

All conditions start from the exact same initial Gaussian noise x_T.
Trajectories are recorded at every timestep, jointly projected via PCA,
and visualized as time-gradient-colored curves in 2D.
"""

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# Add project root to path so notebooks package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler
from notebooks.utils import (
    get_sd_models,
    get_sd3_models,
    get_gligen_models,
    get_image,
    get_text_embedding,
)
from notebooks.composition_experiments import (
    LatentTrajectoryCollector,
    sample_with_trajectory_tracking,
    superdiff_with_trajectory_tracking,
    get_vel_sd3,
    _get_sd3_conditioning,
    sample_sd3_with_trajectory_tracking,
    superdiff_sd3_with_trajectory_tracking,
    sample_gligen_with_trajectory_tracking,
    poe_gligen_with_trajectory_tracking,
)
from notebooks.dynamics import get_latents, get_vel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUTHOR_STOCH_NOTEBOOK_MODEL_ID = "CompVis/stable-diffusion-v1-4"


@dataclass
class TrajectoryExperimentConfig:
    prompt_a: str = "a dog"
    prompt_b: str = "a cat"
    monolithic_prompt: str = ""  # auto-derived from prompt_a + prompt_b if empty
    model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    num_inference_steps: int = 50
    guidance_scale: float = 4.5
    seed: int = 42
    batch_size: int = 1
    z_channels: int = 16
    latent_height: int = 128
    latent_width: int = 128
    lift: float = 0.0
    projection: str = "pca"  # "pca" or "mds"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    output_dir: str = ""
    # SuperDIFF variant: "ours", "fm_ode", "author_det", "author_stoch", "all"
    # - "fm_ode": Flow matching ODE adaptation (recommended for SD3.5)
    # - "ours": Original kappa solver from composition_experiments.py
    # - "author_det": Author's deterministic with JVP (~2x VRAM, may OOM on SD3.5)
    # - "author_stoch": Author's stochastic notebook variant on CompVis/stable-diffusion-v1-4
    # - "all": Run all variants
    superdiff_variant: str = "fm_ode"
    # Classifier probe
    no_clip_probe: bool = False
    no_poe: bool = False
    # SuperDIFF-Guided hybrid: (1-α)·v_mono + α·v_superdiff
    guided: bool = False
    alphas: List[float] = field(default_factory=lambda: [0.3])
    num_seeds: int = 1
    # Prompt list mode (M >= 2 prompts). If provided, this list overrides
    # --prompt-a/--prompt-b and drives either 2-prompt or multi-prompt flow.
    multi_prompts: List[str] = field(default_factory=list)
    # Concept negation (NOT)
    neg_prompt: str = ""  # e.g., "glasses" to suppress
    neg_scale: float = 1.0  # fixed weight for Composable NOT (Eq 13)
    neg_lambda: float = 1.0  # suppression strength for SuperDIFF NOT
    # Uniform color: all trajectories use the same time colormap, differentiated by line style
    uniform_color: bool = False
    # Solo mode: run a single prompt through standard SD3.5 (no composition)
    solo: bool = False
    # GLIGEN mode: switches to DDPM noise-prediction paradigm with bounding-box grounding
    gligen: bool = False
    gligen_model_id: str = "gligen/gligen-generation-text-box"
    gligen_phrases: List[str] = field(default_factory=list)
    gligen_boxes: List[List[float]] = field(default_factory=list)
    gligen_scheduled_sampling_beta: float = 0.3
    # Spatial extension
    spatial: bool = False
    spatial_prompt_a: str = "a dog on the left"
    spatial_prompt_b: str = "a cat on the right"
    spatial_monolithic: str = "a dog on the left and a cat on the right"


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------
CONDITION_COLORS = {
    "prompt_a": "#e63946",       # red
    "prompt_b": "#457b9d",       # blue
    "monolithic": "#2a9d8f",     # teal
    "superdiff": "#e9c46a",      # gold
    "superdiff_det": "#f4a261",  # orange
    "superdiff_stoch": "#bc6c25",# brown
    "superdiff_fm_ode": "#d4a373",# tan
    "superdiff_multi": "#e76f51", # coral
    "superdiff_guided": "#264653",# dark teal
    "poe": "#9b5de5",            # purple
    "composable_not": "#c1121f",  # crimson
    "superdiff_not": "#780000",   # dark red
}

CONDITION_CMAPS = {
    "prompt_a": "Reds",
    "prompt_b": "Blues",
    "monolithic": "Greens",
    "superdiff": "Oranges",
    "superdiff_det": "YlOrBr",
    "superdiff_stoch": "YlGn",
    "superdiff_fm_ode": "copper",
    "superdiff_multi": "RdPu",
    "superdiff_guided": "BuGn",
    "poe": "Purples",
    "composable_not": "RdPu",
    "superdiff_not": "Reds",
}

CONDITION_LABELS = {
    "prompt_a": None,
    "prompt_b": None,
    "monolithic": None,
    "superdiff": "SuperDIFF (ours)",
    "superdiff_det": "SuperDIFF (author, det)",
    "superdiff_stoch": "SuperDIFF (author, stoch)",
    "superdiff_fm_ode": "SuperDIFF (FM-ODE)",
    "superdiff_multi": "SuperDIFF Multi-AND",
    "superdiff_guided": "SuperDIFF-Guided",
    "poe": "PoE (score addition)",
    "composable_not": "Composable NOT (Eq 13)",
    "superdiff_not": "SuperDIFF NOT",
}


def _join_prompts_with_and(prompts: List[str]) -> str:
    """Create a monolithic prompt by joining all prompts with 'and'."""
    if len(prompts) < 2:
        raise ValueError(f"Need at least 2 prompts, got {len(prompts)}")
    return " and ".join(prompts)


def _build_clip_class_prompts(prompts: List[str], monolithic_prompt: str) -> Dict[str, str]:
    """
    Build CLIP text prototypes from active prompts instead of hardcoded classes.
    """
    class_prompts: Dict[str, str] = {}
    for idx, prompt in enumerate(prompts, start=1):
        class_prompts[f"[{idx}] {prompt}"] = prompt
    class_prompts[f"[ALL] {monolithic_prompt}"] = monolithic_prompt
    return class_prompts


def _symbol_for_index(idx: int) -> str:
    """0-based index -> spreadsheet-style symbol (A..Z, AA..)."""
    if idx < 0:
        raise ValueError(f"idx must be >= 0, got {idx}")
    chars: List[str] = []
    n = idx
    while True:
        n, rem = divmod(n, 26)
        chars.append(chr(ord("A") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(chars))


def _format_prompt_key_text(key_to_prompt: Dict[str, str], width: int = 120) -> str:
    """Format prompt key legend text for figure headers."""
    entries = [f"{k} = {v}" for k, v in key_to_prompt.items()]
    joined = " | ".join(entries)
    return textwrap.fill(
        joined,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _build_compact_visual_labels(
    conditions: Dict[str, dict],
    base_prompts: List[str],
    monolithic_prompt: str,
    model_tag: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build compact labels (e.g., "SuperDIFF: A ^ B") plus prompt key map.

    This keeps figure titles and annotations readable when prompts are long.
    """
    prompt_to_key: Dict[str, str] = {}
    key_to_prompt: Dict[str, str] = {}

    def add_prompt(prompt: str):
        p = (prompt or "").strip()
        if not p or p in prompt_to_key:
            return
        key = _symbol_for_index(len(prompt_to_key))
        prompt_to_key[p] = key
        key_to_prompt[key] = p

    def key_for(prompt: str) -> str:
        p = (prompt or "").strip()
        if not p:
            return "?"
        add_prompt(p)
        return prompt_to_key[p]

    def expr_for_prompts(prompts: List[str]) -> str:
        keys = [key_for(p) for p in prompts if (p or "").strip()]
        return " ^ ".join(keys) if keys else "?"

    mono_norm = (monolithic_prompt or "").strip()

    for p in base_prompts:
        add_prompt(p)

    # Pre-register prompt tokens used by conditions.
    for cond in conditions.values():
        ctype = cond.get("type")
        if ctype == "standard":
            p = (cond.get("prompt", "") or "").strip()
            if mono_norm and p == mono_norm and len(base_prompts) >= 2:
                continue
            if " without " in p:
                lhs, rhs = p.split(" without ", 1)
                add_prompt(lhs.strip())
                add_prompt(rhs.strip())
            else:
                add_prompt(p)
        elif ctype in ("superdiff", "superdiff_author_det", "superdiff_author_stoch", "superdiff_fm_ode"):
            add_prompt(cond.get("obj_prompt", ""))
            add_prompt(cond.get("bg_prompt", ""))
        elif ctype in ("superdiff_multi", "superdiff_guided"):
            for p in cond.get("prompts", []):
                add_prompt(p)
        elif ctype == "poe":
            add_prompt(cond.get("prompt_a", ""))
            add_prompt(cond.get("prompt_b", ""))
        elif ctype in ("composable_not", "superdiff_not"):
            add_prompt(cond.get("pos_prompt", ""))
            add_prompt(cond.get("neg_prompt", ""))

    base_expr = expr_for_prompts(base_prompts) if base_prompts else "?"
    compact_labels: Dict[str, str] = {}

    for name, cond in conditions.items():
        ctype = cond.get("type")
        if ctype == "standard":
            p = (cond.get("prompt", "") or "").strip()
            if mono_norm and p == mono_norm and len(base_prompts) >= 2:
                expr = base_expr
            elif " without " in p:
                lhs, rhs = p.split(" without ", 1)
                expr = f"{key_for(lhs)} without {key_for(rhs)}"
            else:
                expr = key_for(p)
            compact_labels[name] = f"{model_tag}: {expr}"
        elif ctype == "poe":
            expr = f"{key_for(cond.get('prompt_a', ''))} x {key_for(cond.get('prompt_b', ''))}"
            compact_labels[name] = f"PoE: {expr}"
        elif ctype == "superdiff":
            expr = f"{key_for(cond.get('obj_prompt', ''))} ^ {key_for(cond.get('bg_prompt', ''))}"
            compact_labels[name] = f"SuperDIFF (ours): {expr}"
        elif ctype == "superdiff_author_det":
            expr = f"{key_for(cond.get('obj_prompt', ''))} ^ {key_for(cond.get('bg_prompt', ''))}"
            compact_labels[name] = f"SuperDIFF (author, det): {expr}"
        elif ctype == "superdiff_author_stoch":
            expr = f"{key_for(cond.get('obj_prompt', ''))} ^ {key_for(cond.get('bg_prompt', ''))}"
            compact_labels[name] = f"SuperDIFF (author, stoch): {expr}"
        elif ctype == "superdiff_fm_ode":
            expr = f"{key_for(cond.get('obj_prompt', ''))} ^ {key_for(cond.get('bg_prompt', ''))}"
            compact_labels[name] = f"SuperDIFF (FM-ODE): {expr}"
        elif ctype == "superdiff_multi":
            expr = expr_for_prompts(cond.get("prompts", []))
            compact_labels[name] = f"SuperDIFF Multi-AND: {expr}"
        elif ctype == "superdiff_guided":
            expr = expr_for_prompts(cond.get("prompts", []))
            alpha = cond.get("alpha", 0.3)
            compact_labels[name] = f"Guided (alpha={alpha:g}): {expr}"
        elif ctype == "composable_not":
            expr = f"{key_for(cond.get('pos_prompt', ''))} ~{key_for(cond.get('neg_prompt', ''))}"
            compact_labels[name] = f"Composable NOT: {expr}"
        elif ctype == "superdiff_not":
            lam = cond.get("neg_lambda", 1.0)
            expr = f"{key_for(cond.get('pos_prompt', ''))} ~{key_for(cond.get('neg_prompt', ''))}"
            compact_labels[name] = f"SuperDIFF NOT (lambda={lam:g}): {expr}"
        else:
            compact_labels[name] = name

    return compact_labels, key_to_prompt


# Sequential colormaps for alpha sweep — visually distinct, dark-to-light progression
_GUIDED_CMAPS = ["YlGn", "BuPu", "OrRd", "YlGnBu", "PuRd", "GnBu", "YlOrRd"]
_GUIDED_COLORS = ["#264653", "#7b2d8e", "#c1440e", "#1d7874", "#a4133c", "#2d6a4f", "#e76f51"]


def _register_guided_conditions(
    conditions: Dict[str, dict],
    labels: Dict[str, str],
    alphas: List[float],
    prompts: List[str],
    monolithic_prompt: str,
):
    """Register one SuperDIFF-Guided condition per alpha value."""
    prompt_str = " \u2227 ".join(f'"{p}"' for p in prompts)
    for i, alpha in enumerate(alphas):
        key = f"guided_a{alpha:.2f}".replace(".", "")
        conditions[key] = {
            "type": "superdiff_guided",
            "prompts": prompts,
            "alpha": alpha,
            "monolithic_prompt": monolithic_prompt,
        }
        labels[key] = f"Guided (\u03b1={alpha})"
        CONDITION_CMAPS[key] = _GUIDED_CMAPS[i % len(_GUIDED_CMAPS)]
        CONDITION_COLORS[key] = _GUIDED_COLORS[i % len(_GUIDED_COLORS)]


# ---------------------------------------------------------------------------
# Author's exact SuperDIFF AND algorithms (adapted for SD3 flow matching)
# From: notebooks/superposition_AND.ipynb
# ---------------------------------------------------------------------------
def _get_vel_sd3_with_div(transformer, t, latents, prompt_embeds, pooled_embeds,
                          eps, device=torch.device("cuda"), dtype=torch.float16):
    """
    Get velocity prediction AND Hutchinson divergence estimate from SD3 transformer.

    This is the SD3 adaptation of the author's get_vel(..., get_div=True).
    Uses torch.func.jvp to compute ∂v/∂x · eps for the trace estimator:
        div(v) ≈ -eps^T · (∂v/∂x · eps)

    Args:
        eps: Rademacher random vector (±1), same shape as latents
    Returns:
        vel: velocity prediction
        dlog: Hutchinson divergence estimate = -(eps * jvp_output).sum((1,2,3))
    """
    latents_in = latents.to(device=device, dtype=dtype)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_embeds = pooled_embeds.to(device=device, dtype=dtype)
    timestep = t.expand(latents_in.shape[0]).to(device=device)

    def v_fn(_x):
        with torch.autocast("cuda", dtype=dtype):
            return transformer(
                hidden_states=_x,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

    from torch.nn.attention import SDPBackend, sdpa_kernel
    with torch.enable_grad():
        with sdpa_kernel(SDPBackend.MATH):
            vel, jvp_out = torch.func.jvp(v_fn, (latents_in,), (eps,))
            dlog = -(eps * jvp_out).sum((1, 2, 3))

    return vel, dlog


def superdiff_author_deterministic_sd3(
    latents, obj_prompt, bg_prompt, scheduler, transformer,
    tokenizer, text_encoder, tokenizer_2, text_encoder_2,
    tokenizer_3, text_encoder_3,
    guidance_scale=4.5, num_inference_steps=50, batch_size=1,
    device=torch.device("cuda"), dtype=torch.float16, lift=0.0,
):
    """
    Author's DETERMINISTIC SuperDIFF AND for SD3, direct port of cell-5.

    Key features:
    - Uses JVP divergence estimator (Hutchinson trace) for kappa computation
    - NO noise injection — pure ODE step: latents += dt * vf
    - Kappa includes sigma*(dlog_obj - dlog_bg) divergence term

    This is the "exact" version from the paper, Eq. 16-18.
    """
    obj_embeds, obj_pooled = _get_sd3_conditioning(
        obj_prompt, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    bg_embeds, bg_pooled = _get_sd3_conditioning(
        bg_prompt, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    uncond_embeds, uncond_pooled = _get_sd3_conditioning(
        "", batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )

    tracker = LatentTrajectoryCollector(
        num_inference_steps, batch_size,
        latents.shape[1], latents.shape[2], latents.shape[3],
    )
    ll_obj = torch.zeros((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)
    ll_bg = torch.zeros((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)
    kappa = 0.5 * torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        dt = scheduler.sigmas[i + 1] - sigma

        # Rademacher random vector for Hutchinson estimator
        eps = (torch.randint_like(latents, 2, dtype=latents.dtype) * 2 - 1)

        # Get velocities WITH divergence for obj and bg
        vel_obj, dlog_obj = _get_vel_sd3_with_div(
            transformer, t, latents, obj_embeds, obj_pooled, eps,
            device=device, dtype=dtype,
        )
        vel_bg, dlog_bg = _get_vel_sd3_with_div(
            transformer, t, latents, bg_embeds, bg_pooled, eps,
            device=device, dtype=dtype,
        )
        # Unconditional: no divergence needed
        vel_uncond = get_vel_sd3(transformer, t, latents, uncond_embeds, uncond_pooled,
                                 device=device, dtype=dtype)

        # Author's exact kappa formula (deterministic, cell-5):
        #   kappa = sigma*(dlog_obj - dlog_bg)
        #         + ((vel_obj-vel_bg)*(vel_obj+vel_bg)).sum()
        #         + lift/dt*sigma/N
        #         - ((vel_obj-vel_bg)*(vel_uncond + gs*(vel_bg-vel_uncond))).sum()
        #         / gs*((vel_obj-vel_bg)²).sum()
        diff = vel_obj - vel_bg
        kappa_num = (
            float(sigma) * (dlog_obj - dlog_bg)
            + (diff * (vel_obj + vel_bg)).sum((1, 2, 3))
            + lift / (float(dt) + 1e-12) * float(sigma) / num_inference_steps
            - (diff * (vel_uncond + guidance_scale * (vel_bg - vel_uncond))).sum((1, 2, 3))
        )
        kappa_den = guidance_scale * (diff ** 2).sum((1, 2, 3))
        kappa[i + 1] = kappa_num / (kappa_den + 1e-8)

        # Composite velocity field
        vf = vel_uncond + guidance_scale * (
            (vel_bg - vel_uncond) + kappa[i + 1][:, None, None, None] * diff
        )

        # Store BEFORE step (consistent with standard/PoE samplers)
        tracker.store_step(i, latents, vf, float(sigma), t.item())

        # Deterministic ODE step (author: latents += dsigma * vf)
        latents = latents + dt * vf

        # Update log-likelihoods (author's exact formula, cell-5):
        #   ll_obj[i+1] = ll_obj[i] + dsigma*(dlog_obj - ((-vel_obj/sigma)*(vel_obj-vf)).sum())
        sigma_safe = max(float(sigma), 1e-6)
        ll_obj[i + 1] = ll_obj[i] + float(dt) * (
            dlog_obj - ((-vel_obj / sigma_safe) * (vel_obj - vf)).sum((1, 2, 3))
        )
        ll_bg[i + 1] = ll_bg[i] + float(dt) * (
            dlog_bg - ((-vel_bg / sigma_safe) * (vel_bg - vf)).sum((1, 2, 3))
        )

    tracker.store_final(latents)
    return latents, tracker, kappa, ll_obj, ll_bg


def superdiff_author_stochastic_sd3(
    latents, obj_prompt, bg_prompt, scheduler, transformer,
    tokenizer, text_encoder, tokenizer_2, text_encoder_2,
    tokenizer_3, text_encoder_3,
    guidance_scale=4.5, num_inference_steps=50, batch_size=1,
    device=torch.device("cuda"), dtype=torch.float16, lift=0.0,
):
    """
    Author's STOCHASTIC SuperDIFF AND for SD3, ported from superposition_AND.ipynb.

    Implements the notebook update rule:
        noise  = sqrt(2*|dt|*sigma) * N(0, I)
        dx_ind = 2*dt*(v_unc + gs*(v_bg - v_unc)) + noise
        kappa  = [|dt|*(v_bg-v_obj)*(v_bg+v_obj) - dx_ind*(v_obj-v_bg) + sigma*lift/N]
                 / [2*dt*gs*||v_obj-v_bg||^2]
        dx     = 2*dt*v_comp + noise
        x      = x + dx

    NOTE:
    This injects stochastic VE-style noise each step. It matches the author's
    notebook behavior but is not a pure flow-matching ODE trajectory.
    """
    obj_embeds, obj_pooled = _get_sd3_conditioning(
        obj_prompt, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    bg_embeds, bg_pooled = _get_sd3_conditioning(
        bg_prompt, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    uncond_embeds, uncond_pooled = _get_sd3_conditioning(
        "", batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )

    tracker = LatentTrajectoryCollector(
        num_inference_steps, batch_size,
        latents.shape[1], latents.shape[2], latents.shape[3],
    )
    ll_obj = torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)
    ll_bg = torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)
    kappa = 0.5 * torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        dt = scheduler.sigmas[i + 1] - sigma
        sigma_f = float(sigma)
        dt_f = float(dt)
        sigma_safe = max(sigma_f, 1e-6)

        vel_obj = get_vel_sd3(transformer, t, latents, obj_embeds, obj_pooled,
                              device=device, dtype=dtype)
        vel_bg = get_vel_sd3(transformer, t, latents, bg_embeds, bg_pooled,
                             device=device, dtype=dtype)
        vel_uncond = get_vel_sd3(transformer, t, latents, uncond_embeds, uncond_pooled,
                                 device=device, dtype=dtype)

        noise_scale = (2.0 * abs(dt_f) * max(sigma_f, 0.0)) ** 0.5
        noise = noise_scale * torch.randn_like(latents)

        diff = vel_obj - vel_bg
        dx_ind = 2.0 * dt_f * (vel_uncond + guidance_scale * (vel_bg - vel_uncond)) + noise

        kappa_num = (
            (abs(dt_f) * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
            - (dx_ind * diff).sum((1, 2, 3))
            + sigma_f * lift / num_inference_steps
        )
        kappa_den = 2.0 * dt_f * guidance_scale * (diff ** 2).sum((1, 2, 3))
        kappa[i + 1] = kappa_num / (kappa_den + 1e-8)

        vf = vel_uncond + guidance_scale * (
            (vel_bg - vel_uncond) + kappa[i + 1][:, None, None, None] * diff
        )

        # Store BEFORE step (consistent with standard/PoE samplers)
        tracker.store_step(i, latents, vf, sigma_f, t.item())

        dx = 2.0 * dt_f * vf + noise
        latents = latents + dx

        ll_obj[i + 1] = ll_obj[i] + (
            -abs(dt_f) / sigma_safe * (vel_obj ** 2)
            - (dx * (vel_obj / sigma_safe))
        ).sum((1, 2, 3))
        ll_bg[i + 1] = ll_bg[i] + (
            -abs(dt_f) / sigma_safe * (vel_bg ** 2)
            - (dx * (vel_bg / sigma_safe))
        ).sum((1, 2, 3))

    tracker.store_final(latents)
    return latents, tracker, kappa, ll_obj, ll_bg


def superdiff_fm_ode_sd3(
    latents, obj_prompt, bg_prompt, scheduler, transformer,
    tokenizer, text_encoder, tokenizer_2, text_encoder_2,
    tokenizer_3, text_encoder_3,
    guidance_scale=4.5, num_inference_steps=50, batch_size=1,
    device=torch.device("cuda"), dtype=torch.float16, lift=0.0,
):
    """
    SuperDIFF AND adapted for Flow Matching (SD3) — deterministic ODE.

    The author's stochastic variant injects noise to estimate log-densities
    without JVP. But SD3 uses flow matching (an ODE, not SDE), so injecting
    VE-SDE noise corrupts the trajectory (causes pixelation).

    This variant uses the deterministic kappa formula WITHOUT divergence:
    when noise=0, the stochastic kappa reduces to the deterministic one
    minus the divergence correction term (which is a small correction).

    Kappa formula (deterministic, no JVP):
        diff = v_obj - v_bg
        dx_bg = dt · (v_unc + gs·(v_bg - v_unc))        # bg-only ODE step
        κ = [|dt|·(-diff)·(v_bg+v_obj) - dx_bg·diff + σ·ℓ/N]
            / [dt · gs · ‖diff‖²]

    This ensures equal log-density evolution (Proposition 6) up to the
    divergence correction, which is typically small.

    Step: pure ODE  →  latents += dt · v_composite
    """
    obj_embeds, obj_pooled = _get_sd3_conditioning(
        obj_prompt, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    bg_embeds, bg_pooled = _get_sd3_conditioning(
        bg_prompt, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    uncond_embeds, uncond_pooled = _get_sd3_conditioning(
        "", batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )

    tracker = LatentTrajectoryCollector(
        num_inference_steps, batch_size,
        latents.shape[1], latents.shape[2], latents.shape[3],
    )
    ll_obj = torch.zeros((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)
    ll_bg = torch.zeros((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)
    kappa = 0.5 * torch.ones((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        dt = scheduler.sigmas[i + 1] - sigma
        sigma_safe = max(float(sigma), 1e-6)

        vel_obj = get_vel_sd3(transformer, t, latents, obj_embeds, obj_pooled,
                              device=device, dtype=dtype)
        vel_bg = get_vel_sd3(transformer, t, latents, bg_embeds, bg_pooled,
                             device=device, dtype=dtype)
        vel_uncond = get_vel_sd3(transformer, t, latents, uncond_embeds, uncond_pooled,
                                 device=device, dtype=dtype)

        # Deterministic kappa (no noise, no JVP divergence):
        # This is the stochastic formula with noise=0, which reduces to
        # the deterministic formula minus the divergence correction.
        diff = vel_obj - vel_bg
        dx_bg = dt * (vel_uncond + guidance_scale * (vel_bg - vel_uncond))

        kappa_num = (
            (torch.abs(dt) * (vel_bg - vel_obj) * (vel_bg + vel_obj)).sum((1, 2, 3))
            - (dx_bg * diff).sum((1, 2, 3))
            + float(sigma) * lift / num_inference_steps
        )
        kappa_den = dt * guidance_scale * (diff ** 2).sum((1, 2, 3))
        kappa[i + 1] = kappa_num / (kappa_den + 1e-8)

        # Composite velocity field
        vf = vel_uncond + guidance_scale * (
            (vel_bg - vel_uncond) + kappa[i + 1][:, None, None, None] * diff
        )

        # Store BEFORE step (consistent with standard/PoE samplers)
        tracker.store_step(i, latents, vf, float(sigma), t.item())

        # Pure ODE step (no noise — flow matching is an ODE)
        dx = dt * vf
        latents = latents + dx

        # Approximate log-likelihoods via ODE density estimator
        # (without divergence term, these track relative changes only)
        ll_obj[i + 1] = ll_obj[i] + (
            -torch.abs(dt) / sigma_safe * (vel_obj ** 2)
            - (dx * (vel_obj / sigma_safe))
        ).sum((1, 2, 3))
        ll_bg[i + 1] = ll_bg[i] + (
            -torch.abs(dt) / sigma_safe * (vel_bg ** 2)
            - (dx * (vel_bg / sigma_safe))
        ).sum((1, 2, 3))

    tracker.store_final(latents)
    return latents, tracker, kappa, ll_obj, ll_bg


# ---------------------------------------------------------------------------
# Multi-prompt SuperDIFF AND for Flow Matching (SD3) — M ≥ 2 prompts
# ---------------------------------------------------------------------------

def _solve_kappa_multi_fm_ode(velocities, vel_uncond, dt, sigma,
                               guidance_scale, lift, num_inference_steps):
    """
    Solve the M×M linear system from Proposition 6 for AND composition,
    adapted for flow matching ODE (no noise, no JVP divergence).

    Finds κ = [κ₁, ..., κₘ] with Σκ = 1 such that all log-densities
    evolve at the same rate:  d log qⁱ = d log qʲ  ∀ i,j ∈ [M].

    Flow matching adaptation:
    - dx_base = dt · v_unc  (no noise)
    - A[j,k] = dt · gs · ⟨vₖ − v_unc, vⱼ₊₁ − v₀⟩
    - b[j] = |dt| · (‖v₀‖² − ‖vⱼ₊₁‖²) − ⟨dx_base, vⱼ₊₁ − v₀⟩ − σ·ℓ/N
    - Plus Σₖ κₖ = 1

    For M=2, this is equivalent to the closed-form in superdiff_fm_ode_sd3.
    """
    M = len(velocities)
    B = velocities[0].shape[0]
    dev = velocities[0].device

    # Flatten spatial dims: [M, B, D]
    vels = torch.stack([v.flatten(1) for v in velocities])          # [M, B, D]
    v_unc = vel_uncond.flatten(1)                                    # [B, D]
    dx_base = (dt * vel_uncond).flatten(1)                           # [B, D]

    # u_diff[k] = vₖ − v_unc
    u_diff = vels - v_unc.unsqueeze(0)                               # [M, B, D]
    # v_diff[j] = v_{j+1} − v₀
    v_diff = vels[1:] - vels[0:1]                                   # [M-1, B, D]

    # Build A [B, M, M]
    u_diff_t = u_diff.permute(1, 0, 2).float()                      # [B, M, D]
    v_diff_t = v_diff.permute(1, 0, 2).float()                      # [B, M-1, D]

    A = torch.zeros(B, M, M, device=dev, dtype=torch.float32)
    A[:, :M-1, :] = (dt * guidance_scale) * torch.bmm(
        v_diff_t, u_diff_t.transpose(1, 2)
    )                                                                # [B, M-1, M]
    A[:, M-1, :] = 1.0                                              # sum constraint

    # Build b [B, M]
    b = torch.zeros(B, M, device=dev, dtype=torch.float32)
    norms_sq = (vels.float() ** 2).sum(dim=2)                       # [M, B]

    for j in range(M - 1):
        norm_term = torch.abs(dt) * (norms_sq[0] - norms_sq[j + 1])
        dot_term = (dx_base.float() * v_diff_t[:, j, :]).sum(dim=1)
        b[:, j] = norm_term - dot_term - sigma * lift / num_inference_steps

    b[:, M-1] = 1.0                                                 # sum constraint

    # Solve A κ = b (least-squares for robustness)
    kappa = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)  # [B, M]

    return kappa.to(velocities[0].dtype)


def superdiff_multi_fm_ode_sd3(
    latents, prompts, scheduler, transformer,
    tokenizer, text_encoder, tokenizer_2, text_encoder_2,
    tokenizer_3, text_encoder_3,
    guidance_scale=4.5, num_inference_steps=50, batch_size=1,
    device=torch.device("cuda"), dtype=torch.float16, lift=0.0,
):
    """
    Multi-prompt SuperDIFF AND for SD3 Flow Matching — pure ODE, M ≥ 2 prompts.

    Composes M prompts by solving the linear system from Proposition 6 so that
    all M log-densities evolve at equal rates. No noise injection, no JVP.

    Composite velocity:  vf = v_unc + gs · Σₘ κₘ · (vₘ − v_unc)
    Step:                latents += dt · vf

    Args:
        prompts: List of M prompt strings to AND-compose

    Returns:
        latents: Final denoised latents
        tracker: LatentTrajectoryCollector
        kappas: (num_steps+1, batch, M) composition weights
        log_likelihoods: (num_steps+1, batch, M) approximate log-likelihoods
    """
    M = len(prompts)
    assert M >= 2, f"Need at least 2 prompts for AND composition, got {M}"

    # Get embeddings for all prompts
    embeddings_list = []
    for prompt in prompts:
        embeds, pooled = _get_sd3_conditioning(
            prompt, batch_size, tokenizer, text_encoder,
            tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
        )
        embeddings_list.append((embeds, pooled))

    uncond_embeds, uncond_pooled = _get_sd3_conditioning(
        "", batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )

    tracker = LatentTrajectoryCollector(
        num_inference_steps, batch_size,
        latents.shape[1], latents.shape[2], latents.shape[3],
    )
    kappas = torch.zeros((num_inference_steps + 1, batch_size, M), device=device, dtype=torch.float32)
    kappas[0] = 1.0 / M  # uniform init
    log_likelihoods = torch.zeros((num_inference_steps + 1, batch_size, M), device=device, dtype=torch.float32)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        dt_val = scheduler.sigmas[i + 1] - sigma
        sigma_safe = max(float(sigma), 1e-6)

        # Compute velocities for all M models
        velocities = []
        for emb, pooled in embeddings_list:
            vel = get_vel_sd3(transformer, t, latents, emb, pooled,
                              device=device, dtype=dtype)
            velocities.append(vel)

        vel_uncond = get_vel_sd3(transformer, t, latents, uncond_embeds, uncond_pooled,
                                 device=device, dtype=dtype)

        # Solve M×M linear system for kappas
        kappas[i + 1] = _solve_kappa_multi_fm_ode(
            velocities, vel_uncond, dt_val, sigma,
            guidance_scale, lift, num_inference_steps,
        )

        # Composite velocity field: vf = v_unc + gs · Σₘ κₘ · (vₘ − v_unc)
        vf = vel_uncond.clone()
        for m in range(M):
            kappa_m = kappas[i + 1, :, m][:, None, None, None]
            vf = vf + guidance_scale * kappa_m * (velocities[m] - vel_uncond)

        # Store BEFORE step (consistent with standard/PoE samplers)
        tracker.store_step(i, latents, vf, float(sigma), t.item())

        # Pure ODE step
        dx = dt_val * vf
        latents = latents + dx

        # Approximate log-likelihoods (no divergence correction)
        for m in range(M):
            vel_m = velocities[m]
            ll_update = (
                -torch.abs(dt_val) / sigma_safe * (vel_m ** 2)
                - (dx * (vel_m / sigma_safe))
            ).sum((1, 2, 3))
            log_likelihoods[i + 1, :, m] = log_likelihoods[i, :, m] + ll_update

    tracker.store_final(latents)
    return latents, tracker, kappas, log_likelihoods


# ---------------------------------------------------------------------------
# SuperDIFF-Guided SD3: Hybrid monolithic + kappa-corrected composition
# ---------------------------------------------------------------------------

def superdiff_guided_fm_ode_sd3(
    latents, prompts, scheduler, transformer,
    tokenizer, text_encoder, tokenizer_2, text_encoder_2,
    tokenizer_3, text_encoder_3,
    guidance_scale=4.5, num_inference_steps=50, batch_size=1,
    device=torch.device("cuda"), dtype=torch.float16,
    lift=0.0, alpha=0.3, monolithic_prompt=None,
):
    """
    Hybrid sampler: SD3.5 monolithic CFG + SuperDIFF kappa correction.

    Standard CFG applies one fixed guidance scale to the entire prompt.
    SuperDIFF provides per-concept, per-timestep guidance (kappa) derived
    from the density equalization constraint.

    This hybrid blends the two:
        v_mono  = v_unc + gs · (v_monolithic − v_unc)      # SD3.5's own composition
        v_sd    = v_unc + gs · Σₘ κₘ · (vₘ − v_unc)       # SuperDIFF's balanced composite
        v_final = (1 − α) · v_mono + α · v_sd              # blend

    When α=0: pure SD3.5 monolithic (standard CFG)
    When α=1: pure SuperDIFF multi-AND
    When α∈(0,1): SD3.5 anchored with SuperDIFF rebalancing correction

    The correction only matters when SD3.5's attention allocation is imbalanced
    (e.g., one concept dominates). At each timestep, kappa adapts to boost
    underrepresented concepts.

    Args:
        prompts: List of M sub-prompts to compose
        alpha: Blending strength for SuperDIFF correction (0=pure SD3.5, 1=pure SuperDIFF)
        monolithic_prompt: Full composed prompt for SD3.5 CFG. If None, auto-joins with "and".

    Returns:
        latents, tracker, kappas, log_likelihoods
    """
    M = len(prompts)
    assert M >= 2, f"Need at least 2 prompts, got {M}"

    if monolithic_prompt is None:
        monolithic_prompt = " and ".join(prompts)

    # Embeddings for all sub-prompts
    embeddings_list = []
    for prompt in prompts:
        embeds, pooled = _get_sd3_conditioning(
            prompt, batch_size, tokenizer, text_encoder,
            tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
        )
        embeddings_list.append((embeds, pooled))

    # Monolithic prompt embedding
    mono_embeds, mono_pooled = _get_sd3_conditioning(
        monolithic_prompt, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )

    # Unconditional
    uncond_embeds, uncond_pooled = _get_sd3_conditioning(
        "", batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )

    tracker = LatentTrajectoryCollector(
        num_inference_steps, batch_size,
        latents.shape[1], latents.shape[2], latents.shape[3],
    )
    kappas = torch.zeros((num_inference_steps + 1, batch_size, M), device=device, dtype=torch.float32)
    kappas[0] = 1.0 / M
    log_likelihoods = torch.zeros((num_inference_steps + 1, batch_size, M), device=device, dtype=torch.float32)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        dt_val = scheduler.sigmas[i + 1] - sigma
        sigma_safe = max(float(sigma), 1e-6)

        # Sub-prompt velocities
        velocities = []
        for emb, pooled in embeddings_list:
            vel = get_vel_sd3(transformer, t, latents, emb, pooled,
                              device=device, dtype=dtype)
            velocities.append(vel)

        # Monolithic velocity
        vel_mono = get_vel_sd3(transformer, t, latents, mono_embeds, mono_pooled,
                               device=device, dtype=dtype)

        # Unconditional velocity
        vel_uncond = get_vel_sd3(transformer, t, latents, uncond_embeds, uncond_pooled,
                                 device=device, dtype=dtype)

        # SuperDIFF kappa (density-equalizing weights)
        kappas[i + 1] = _solve_kappa_multi_fm_ode(
            velocities, vel_uncond, dt_val, sigma,
            guidance_scale, lift, num_inference_steps,
        )

        # SD3.5 monolithic CFG velocity
        v_mono_cfg = vel_uncond + guidance_scale * (vel_mono - vel_uncond)

        # SuperDIFF composite velocity
        v_sd = vel_uncond.clone()
        for m in range(M):
            kappa_m = kappas[i + 1, :, m][:, None, None, None]
            v_sd = v_sd + guidance_scale * kappa_m * (velocities[m] - vel_uncond)

        # Hybrid blend: anchor on SD3.5, correct with SuperDIFF
        vf = (1.0 - alpha) * v_mono_cfg + alpha * v_sd

        # Store BEFORE step (consistent with standard/PoE samplers)
        tracker.store_step(i, latents, vf, float(sigma), t.item())

        # Pure ODE step
        dx = dt_val * vf
        latents = latents + dx

        # Log-likelihoods per sub-concept
        for m in range(M):
            vel_m = velocities[m]
            ll_update = (
                -torch.abs(dt_val) / sigma_safe * (vel_m ** 2)
                - (dx * (vel_m / sigma_safe))
            ).sum((1, 2, 3))
            log_likelihoods[i + 1, :, m] = log_likelihoods[i, :, m] + ll_update

    tracker.store_final(latents)
    return latents, tracker, kappas, log_likelihoods


# ---------------------------------------------------------------------------
# Composable NOT (Liu et al., Eq 13): fixed negative guidance weight
# ---------------------------------------------------------------------------

def composable_not_sd3(
    latents, pos_prompt, neg_prompt, scheduler, transformer,
    tokenizer, text_encoder, tokenizer_2, text_encoder_2,
    tokenizer_3, text_encoder_3,
    guidance_scale=4.5, num_inference_steps=50, batch_size=1,
    device=torch.device("cuda"), dtype=torch.float16, neg_scale=1.0,
):
    """
    Composable NOT from "Composable Diffusion Models" (Liu et al., Eq 13).

    Concept negation via score subtraction:
        p(x | not c̃_j, c_i) ∝ p(x) · p(c_i|x) / p(c̃_j|x)

    In velocity space (flow matching):
        vf = v_unc + gs·(v_pos - v_unc) - neg_scale·(v_neg - v_unc)

    Args:
        pos_prompt: The concept we want (e.g., "a woman wearing a hat")
        neg_prompt: The concept to suppress (e.g., "glasses")
        neg_scale: Fixed negative guidance weight (default 1.0)

    Returns:
        latents, tracker
    """
    pos_embeds, pos_pooled = _get_sd3_conditioning(
        pos_prompt, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    neg_embeds, neg_pooled = _get_sd3_conditioning(
        neg_prompt, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    uncond_embeds, uncond_pooled = _get_sd3_conditioning(
        "", batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )

    tracker = LatentTrajectoryCollector(
        num_inference_steps, batch_size,
        latents.shape[1], latents.shape[2], latents.shape[3],
    )

    # Work in fp32 to avoid overflow from negative guidance subtraction
    latents = latents.float()

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        dt = scheduler.sigmas[i + 1] - sigma

        vel_pos = get_vel_sd3(transformer, t, latents.to(dtype), pos_embeds, pos_pooled,
                              device=device, dtype=dtype).float()
        vel_neg = get_vel_sd3(transformer, t, latents.to(dtype), neg_embeds, neg_pooled,
                              device=device, dtype=dtype).float()
        vel_uncond = get_vel_sd3(transformer, t, latents.to(dtype), uncond_embeds, uncond_pooled,
                                  device=device, dtype=dtype).float()

        # Eq 13: positive guidance + negative suppression (fp32)
        vf = (vel_uncond
              + guidance_scale * (vel_pos - vel_uncond)
              - neg_scale * (vel_neg - vel_uncond))

        # Store BEFORE step
        tracker.store_step(i, latents, vf, float(sigma), t.item())

        # Pure ODE step (fp32)
        latents = latents + dt * vf

    tracker.store_final(latents)
    return latents, tracker


# ---------------------------------------------------------------------------
# SuperDIFF NOT: Dynamic kappa negation for Flow Matching (SD3)
# ---------------------------------------------------------------------------

def superdiff_not_fm_ode_sd3(
    latents, pos_prompt, neg_prompt, scheduler, transformer,
    tokenizer, text_encoder, tokenizer_2, text_encoder_2,
    tokenizer_3, text_encoder_3,
    guidance_scale=4.5, num_inference_steps=50, batch_size=1,
    device=torch.device("cuda"), dtype=torch.float16,
    lift=0.0, neg_lambda=1.0,
):
    """
    SuperDIFF NOT — dynamic concept negation for Flow Matching ODE.

    Unlike Composable NOT (Eq 13) which uses a fixed negative weight,
    SuperDIFF NOT solves for κ_neg(t) at each timestep from the constraint:

        d/dt log q^neg = -λ · d/dt log q^pos

    i.e., the negated concept's log-density DECREASES at rate λ relative
    to the positive concept's increase.

    Composite velocity (same form as AND):
        vf = v_unc + gs·[(v_pos - v_unc) + κ·(v_neg - v_pos)]

    Kappa derivation from NOT constraint:
        κ_NOT = -[|dt|·(‖v_neg‖² + λ·‖v_pos‖²) + ⟨dx_pos, v_neg + λ·v_pos⟩]
                / [dt · gs · ⟨v_neg - v_pos, v_neg + λ·v_pos⟩]

    κ_NOT comes out NEGATIVE, steering the trajectory away from the negated
    concept in score space. The magnitude adapts per-timestep to maintain the
    suppression constraint throughout the diffusion process.

    Args:
        pos_prompt: The concept we want (e.g., "a woman wearing a hat")
        neg_prompt: The concept to suppress (e.g., "glasses")
        neg_lambda: Suppression strength (0=ignore neg, 1=symmetric, >1=aggressive)

    Returns:
        latents, tracker, kappa, ll_pos, ll_neg
    """
    pos_embeds, pos_pooled = _get_sd3_conditioning(
        pos_prompt, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    neg_embeds, neg_pooled = _get_sd3_conditioning(
        neg_prompt, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    uncond_embeds, uncond_pooled = _get_sd3_conditioning(
        "", batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )

    tracker = LatentTrajectoryCollector(
        num_inference_steps, batch_size,
        latents.shape[1], latents.shape[2], latents.shape[3],
    )
    ll_pos = torch.zeros((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)
    ll_neg = torch.zeros((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)
    kappa = torch.zeros((num_inference_steps + 1, batch_size), device=device, dtype=torch.float32)

    lam = neg_lambda

    # Work in fp32 to prevent overflow from negative kappa amplification
    latents = latents.float()

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        dt = scheduler.sigmas[i + 1] - sigma
        sigma_safe = max(float(sigma), 1e-6)

        # fp32 velocities for numerical stability in kappa computation
        vel_pos = get_vel_sd3(transformer, t, latents.to(dtype), pos_embeds, pos_pooled,
                              device=device, dtype=dtype).float()
        vel_neg = get_vel_sd3(transformer, t, latents.to(dtype), neg_embeds, neg_pooled,
                              device=device, dtype=dtype).float()
        vel_uncond = get_vel_sd3(transformer, t, latents.to(dtype), uncond_embeds, uncond_pooled,
                                  device=device, dtype=dtype).float()

        # NOT kappa derivation:
        # Constraint: d log q^neg + λ·d log q^pos = 0
        #
        # diff = v_neg - v_pos
        # w    = v_neg + λ·v_pos  (weighted target)
        # dx_pos = dt·(v_unc + gs·(v_pos - v_unc))  (base ODE step, positive only)
        #
        # κ = -[|dt|·(‖v_neg‖² + λ·‖v_pos‖²) + ⟨dx_pos, w⟩]
        #     / [dt · gs · ⟨diff, w⟩]
        diff = vel_neg - vel_pos
        w = vel_neg + lam * vel_pos
        dx_pos = dt * (vel_uncond + guidance_scale * (vel_pos - vel_uncond))

        kappa_num = -(
            torch.abs(dt) * (vel_neg ** 2 + lam * vel_pos ** 2).sum((1, 2, 3))
            + (dx_pos * w).sum((1, 2, 3))
        ) + float(sigma) * lift / num_inference_steps

        kappa_den = dt * guidance_scale * (diff * w).sum((1, 2, 3))

        # The denominator ⟨diff, w⟩ can cross zero (unlike AND's ‖diff‖²),
        # which causes kappa to spike. Use safe division with fallback.
        den_abs = kappa_den.abs()
        den_safe = torch.where(den_abs > 1e-4, kappa_den, torch.ones_like(kappa_den))
        kappa_raw = torch.where(den_abs > 1e-4, kappa_num / den_safe, torch.zeros_like(kappa_num))
        kappa[i + 1] = kappa_raw.clamp(-2.0, 0.5)

        # Composite velocity: same form as AND, but κ < 0 suppresses neg concept
        vf = vel_uncond + guidance_scale * (
            (vel_pos - vel_uncond) + kappa[i + 1][:, None, None, None] * diff
        )

        # Store BEFORE step
        tracker.store_step(i, latents, vf, float(sigma), t.item())

        # Pure ODE step
        dx = dt * vf
        latents = latents + dx

        # Log-likelihoods
        ll_pos[i + 1] = ll_pos[i] + (
            -torch.abs(dt) / sigma_safe * (vel_pos ** 2)
            - (dx * (vel_pos / sigma_safe))
        ).sum((1, 2, 3))
        ll_neg[i + 1] = ll_neg[i] + (
            -torch.abs(dt) / sigma_safe * (vel_neg ** 2)
            - (dx * (vel_neg / sigma_safe))
        ).sum((1, 2, 3))

    tracker.store_final(latents)
    return latents, tracker, kappa, ll_pos, ll_neg


# ---------------------------------------------------------------------------
# PoE (Product of Experts) via score addition
# ---------------------------------------------------------------------------
def poe_sd_with_trajectory_tracking(
    latents, prompt_a, prompt_b, scheduler, unet,
    tokenizer, text_encoder,
    guidance_scale=4.5, num_inference_steps=50, batch_size=1,
    device=torch.device("cuda"), dtype=torch.float16,
):
    """
    PoE for notebook-style SD backends (Euler + UNet score model).
    """
    a_embeds = get_text_embedding([prompt_a] * batch_size, tokenizer, text_encoder, device)
    b_embeds = get_text_embedding([prompt_b] * batch_size, tokenizer, text_encoder, device)
    uncond_embeds = get_text_embedding([""] * batch_size, tokenizer, text_encoder, device)

    tracker = LatentTrajectoryCollector(
        num_inference_steps, batch_size,
        latents.shape[1], latents.shape[2], latents.shape[3],
    )

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
        sigma = scheduler.sigmas[i]

        vel_a, _ = get_vel(unet, t, sigma, latents, [a_embeds], device=device, dtype=dtype)
        vel_b, _ = get_vel(unet, t, sigma, latents, [b_embeds], device=device, dtype=dtype)
        vel_uncond, _ = get_vel(unet, t, sigma, latents, [uncond_embeds], device=device, dtype=dtype)

        # PoE score addition: s_a + s_b - s_unc
        vf = vel_uncond + guidance_scale * ((vel_a - vel_uncond) + (vel_b - vel_uncond))

        noise = torch.sqrt(2 * torch.abs(dsigma) * sigma) * torch.randn_like(latents)
        dx = 2 * dsigma * vf + noise

        tracker.store_step(i, latents, vf, sigma.item(), t.item())
        latents = latents + dx

    tracker.store_final(latents)
    return latents, tracker


def poe_sd3_with_trajectory_tracking(
    latents, prompt_a, prompt_b, scheduler, transformer,
    tokenizer, text_encoder, tokenizer_2, text_encoder_2,
    tokenizer_3, text_encoder_3,
    guidance_scale=4.5, num_inference_steps=50, batch_size=1,
    device=torch.device("cuda"), dtype=torch.float16,
):
    """
    Product of Experts composition for SD3 flow matching.

    PoE corresponds to p(x) ∝ p_A(x) · p_B(x), so in score space:
        ∇log p_PoE = ∇log p_A + ∇log p_B

    In the CFG framework this becomes:
        v_PoE = v_unc + gs·((v_A − v_unc) + (v_B − v_unc))

    This finds latent regions with high probability under BOTH models
    simultaneously, without any dynamic kappa reweighting.

    Returns: (final_latents, tracker)
    """
    a_embeds, a_pooled = _get_sd3_conditioning(
        prompt_a, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    b_embeds, b_pooled = _get_sd3_conditioning(
        prompt_b, batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )
    uncond_embeds, uncond_pooled = _get_sd3_conditioning(
        "", batch_size, tokenizer, text_encoder,
        tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, device,
    )

    tracker = LatentTrajectoryCollector(
        num_inference_steps, batch_size,
        latents.shape[1], latents.shape[2], latents.shape[3],
    )

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[i]
        dt = scheduler.sigmas[i + 1] - sigma

        vel_a = get_vel_sd3(transformer, t, latents, a_embeds, a_pooled,
                            device=device, dtype=dtype)
        vel_b = get_vel_sd3(transformer, t, latents, b_embeds, b_pooled,
                            device=device, dtype=dtype)
        vel_uncond = get_vel_sd3(transformer, t, latents, uncond_embeds, uncond_pooled,
                                 device=device, dtype=dtype)

        # PoE: sum of conditional scores (each relative to unconditional)
        # v_PoE = v_unc + gs * ((v_A - v_unc) + (v_B - v_unc))
        vf = vel_uncond + guidance_scale * (
            (vel_a - vel_uncond) + (vel_b - vel_uncond)
        )

        dx = dt * vf
        tracker.store_step(i, latents, vf, float(sigma), t.item())
        latents = latents + dx

    tracker.store_final(latents)
    return latents, tracker


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------
def collect_trajectories(
    cfg: TrajectoryExperimentConfig,
    models: dict,
    conditions: Dict[str, dict],
) -> Dict[str, LatentTrajectoryCollector]:
    """Run all conditions and return trajectory collectors."""

    device = torch.device(cfg.device)
    dtype = cfg.dtype
    using_sd_legacy = (not cfg.gligen) and ("unet" in models) and ("transformer" not in models)

    if cfg.gligen:
        scheduler = models["scheduler"]
    elif using_sd_legacy:
        scheduler = EulerDiscreteScheduler.from_pretrained(
            cfg.model_id, subfolder="scheduler"
        )
    else:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            cfg.model_id, subfolder="scheduler"
        )

    # Shared initial noise
    x_T = get_latents(
        scheduler,
        z_channels=cfg.z_channels,
        device=device,
        dtype=dtype,
        num_inference_steps=cfg.num_inference_steps,
        batch_size=cfg.batch_size,
        latent_width=cfg.latent_width,
        latent_height=cfg.latent_height,
        seed=cfg.seed,
    )

    if cfg.gligen:
        common_kwargs = dict(
            scheduler=scheduler,
            unet=models["unet"],
            tokenizer=models["tokenizer"],
            text_encoder=models["text_encoder"],
            gligen_phrases=cfg.gligen_phrases or None,
            gligen_boxes=cfg.gligen_boxes or None,
            gligen_scheduled_sampling_beta=cfg.gligen_scheduled_sampling_beta,
            guidance_scale=cfg.guidance_scale,
            num_inference_steps=cfg.num_inference_steps,
            batch_size=cfg.batch_size,
            device=device,
            dtype=dtype,
        )
    elif using_sd_legacy:
        common_kwargs = dict(
            scheduler=scheduler,
            unet=models["unet"],
            tokenizer=models["tokenizer"],
            text_encoder=models["text_encoder"],
            guidance_scale=cfg.guidance_scale,
            num_inference_steps=cfg.num_inference_steps,
            batch_size=cfg.batch_size,
            device=device,
            dtype=dtype,
            height=cfg.latent_height * 8,
            width=cfg.latent_width * 8,
        )
    else:
        common_kwargs = dict(
            scheduler=scheduler,
            transformer=models["transformer"],
            tokenizer=models["tokenizer"],
            text_encoder=models["text_encoder"],
            tokenizer_2=models["tokenizer_2"],
            text_encoder_2=models["text_encoder_2"],
            tokenizer_3=models["tokenizer_3"],
            text_encoder_3=models["text_encoder_3"],
            guidance_scale=cfg.guidance_scale,
            num_inference_steps=cfg.num_inference_steps,
            batch_size=cfg.batch_size,
            device=device,
            dtype=dtype,
        )

    trackers = {}
    final_latents = {}
    kappa_data = {}          # name -> {kappa, log_likelihoods, concept_labels}

    for name, cond in conditions.items():
        print(f"  Running condition: {name} ...")
        latents_clone = x_T.clone()

        # --- GLIGEN dispatch ---
        if cfg.gligen:
            if cond["type"] == "standard":
                latents_out, tracker = sample_gligen_with_trajectory_tracking(
                    latents_clone, cond["prompt"], **common_kwargs,
                )
            elif cond["type"] == "poe":
                latents_out, tracker = poe_gligen_with_trajectory_tracking(
                    latents_clone,
                    cond["prompt_a"],
                    cond["prompt_b"],
                    **common_kwargs,
                )
            else:
                raise ValueError(
                    f"GLIGEN mode only supports 'standard' and 'poe' conditions, "
                    f"got '{cond['type']}'"
                )
        # --- Legacy SD dispatch (notebook-compatible author_stoch backend) ---
        elif using_sd_legacy:
            if cond["type"] == "standard":
                latents_out, tracker = sample_with_trajectory_tracking(
                    latents_clone, cond["prompt"], **common_kwargs,
                )
            elif cond["type"] == "superdiff_author_stoch":
                latents_out, tracker, kappa, ll_obj, ll_bg = (
                    superdiff_with_trajectory_tracking(
                        latents_clone,
                        cond["obj_prompt"],
                        cond["bg_prompt"],
                        **common_kwargs,
                        lift=cfg.lift,
                    )
                )
            kappa_data[name] = {
                    "kappa": [[round(float(1 - kappa[t, 0]), 6), round(float(kappa[t, 0]), 6)]
                               for t in range(kappa.shape[0])],
                    "log_likelihoods": [[round(float(ll_obj[t, 0]), 6), round(float(ll_bg[t, 0]), 6)]
                                         for t in range(ll_obj.shape[0])],
                    "concept_labels": [cond["obj_prompt"], cond["bg_prompt"]],
                }
            elif cond["type"] == "poe":
                latents_out, tracker = poe_sd_with_trajectory_tracking(
                    latents_clone,
                    cond["prompt_a"],
                    cond["prompt_b"],
                    scheduler=scheduler,
                    unet=models["unet"],
                    tokenizer=models["tokenizer"],
                    text_encoder=models["text_encoder"],
                    guidance_scale=cfg.guidance_scale,
                    num_inference_steps=cfg.num_inference_steps,
                    batch_size=cfg.batch_size,
                    device=device,
                    dtype=dtype,
                )
            else:
                raise ValueError(
                    "Notebook author_stoch backend supports only condition types "
                    f"'standard', 'poe', and 'superdiff_author_stoch'; got '{cond['type']}'"
                )
        # --- SD3.5 dispatch ---
        elif cond["type"] == "standard":
            latents_out, tracker = sample_sd3_with_trajectory_tracking(
                latents_clone, cond["prompt"], **common_kwargs,
            )
        elif cond["type"] == "superdiff":
            latents_out, tracker, kappa, ll_obj, ll_bg = (
                superdiff_sd3_with_trajectory_tracking(
                    latents_clone,
                    cond["obj_prompt"],
                    cond["bg_prompt"],
                    **common_kwargs,
                    lift=cfg.lift,
                )
            )
            kappa_data[name] = {
                "kappa": [[round(float(1 - kappa[t, 0]), 6), round(float(kappa[t, 0]), 6)]
                           for t in range(kappa.shape[0])],
                "log_likelihoods": [[round(float(ll_obj[t, 0]), 6), round(float(ll_bg[t, 0]), 6)]
                                     for t in range(ll_obj.shape[0])],
                "concept_labels": [cond["obj_prompt"], cond["bg_prompt"]],
            }
        elif cond["type"] == "superdiff_author_det":
            latents_out, tracker, kappa, ll_obj, ll_bg = (
                superdiff_author_deterministic_sd3(
                    latents_clone,
                    cond["obj_prompt"],
                    cond["bg_prompt"],
                    **common_kwargs,
                    lift=cfg.lift,
                )
            )
            kappa_data[name] = {
                "kappa": [[round(float(1 - kappa[t, 0]), 6), round(float(kappa[t, 0]), 6)]
                           for t in range(kappa.shape[0])],
                "log_likelihoods": [[round(float(ll_obj[t, 0]), 6), round(float(ll_bg[t, 0]), 6)]
                                     for t in range(ll_obj.shape[0])],
                "concept_labels": [cond["obj_prompt"], cond["bg_prompt"]],
            }
        elif cond["type"] == "superdiff_fm_ode":
            latents_out, tracker, kappa, ll_obj, ll_bg = (
                superdiff_fm_ode_sd3(
                    latents_clone,
                    cond["obj_prompt"],
                    cond["bg_prompt"],
                    **common_kwargs,
                    lift=cfg.lift,
                )
            )
            kappa_data[name] = {
                "kappa": [[round(float(1 - kappa[t, 0]), 6), round(float(kappa[t, 0]), 6)]
                           for t in range(kappa.shape[0])],
                "log_likelihoods": [[round(float(ll_obj[t, 0]), 6), round(float(ll_bg[t, 0]), 6)]
                                     for t in range(ll_obj.shape[0])],
                "concept_labels": [cond["obj_prompt"], cond["bg_prompt"]],
            }
        elif cond["type"] == "superdiff_multi":
            latents_out, tracker, kappas_m, ll_m = (
                superdiff_multi_fm_ode_sd3(
                    latents_clone,
                    cond["prompts"],
                    **common_kwargs,
                    lift=cfg.lift,
                )
            )
            kappa_data[name] = {
                "kappa": [[round(float(kappas_m[t, 0, m]), 6) for m in range(kappas_m.shape[2])]
                           for t in range(kappas_m.shape[0])],
                "log_likelihoods": [[round(float(ll_m[t, 0, m]), 6) for m in range(ll_m.shape[2])]
                                     for t in range(ll_m.shape[0])],
                "concept_labels": cond["prompts"],
            }
        elif cond["type"] == "superdiff_guided":
            latents_out, tracker, kappas_m, ll_m = (
                superdiff_guided_fm_ode_sd3(
                    latents_clone,
                    cond["prompts"],
                    **common_kwargs,
                    lift=cfg.lift,
                    alpha=cond.get("alpha", 0.3),
                    monolithic_prompt=cond.get("monolithic_prompt"),
                )
            )
            kappa_data[name] = {
                "kappa": [[round(float(kappas_m[t, 0, m]), 6) for m in range(kappas_m.shape[2])]
                           for t in range(kappas_m.shape[0])],
                "log_likelihoods": [[round(float(ll_m[t, 0, m]), 6) for m in range(ll_m.shape[2])]
                                     for t in range(ll_m.shape[0])],
                "concept_labels": cond["prompts"],
            }
        elif cond["type"] == "composable_not":
            latents_out, tracker = composable_not_sd3(
                latents_clone,
                cond["pos_prompt"],
                cond["neg_prompt"],
                **common_kwargs,
                neg_scale=cond.get("neg_scale", 1.0),
            )
        elif cond["type"] == "superdiff_not":
            latents_out, tracker, kappa_neg, ll_pos, ll_neg = (
                superdiff_not_fm_ode_sd3(
                    latents_clone,
                    cond["pos_prompt"],
                    cond["neg_prompt"],
                    **common_kwargs,
                    lift=cfg.lift,
                    neg_lambda=cond.get("neg_lambda", 1.0),
                )
            )
            kappa_data[name] = {
                "kappa": [[round(float(kappa_neg[t, 0]), 6)] for t in range(kappa_neg.shape[0])],
                "log_likelihoods": [[round(float(ll_pos[t, 0]), 6), round(float(ll_neg[t, 0]), 6)]
                                     for t in range(ll_pos.shape[0])],
                "concept_labels": [cond["pos_prompt"], cond["neg_prompt"]],
            }
        elif cond["type"] == "poe":
            latents_out, tracker = poe_sd3_with_trajectory_tracking(
                latents_clone,
                cond["prompt_a"],
                cond["prompt_b"],
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unknown condition type: {cond['type']}")

        trackers[name] = tracker
        final_latents[name] = latents_out
        print(f"    Done. Final latent norm: {latents_out.norm().item():.2f}")

        # Free intermediate VRAM between conditions
        torch.cuda.empty_cache()

    return trackers, final_latents, kappa_data


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def _flatten_trackers_padded(
    trackers: Dict[str, LatentTrajectoryCollector],
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Flatten trajectories to (T+1, D) and zero-pad to common D if needed.

    This enables joint analysis when conditions come from different backends
    (e.g., SD3 16x128x128 vs SD1.4 4x64x64 latents).
    """
    flat_raw: Dict[str, np.ndarray] = {}
    dims = {}
    n_steps = None
    max_dim = 0

    for name, tracker in trackers.items():
        traj = tracker.trajectories  # (T+1, B, C, H, W)
        T_plus_1 = traj.shape[0]
        n_steps = T_plus_1
        flat = traj[:, 0].reshape(T_plus_1, -1).numpy().astype(np.float32)
        flat_raw[name] = flat
        dims[name] = flat.shape[1]
        max_dim = max(max_dim, flat.shape[1])

    if len(set(dims.values())) > 1:
        dim_str = ", ".join(f"{k}:{v}" for k, v in dims.items())
        print("  NOTE: mixed latent dimensions detected; zero-padding for joint analysis.")
        print(f"        dims = {dim_str}")

    flat_padded: Dict[str, np.ndarray] = {}
    for name, flat in flat_raw.items():
        if flat.shape[1] == max_dim:
            flat_padded[name] = flat
        else:
            pad = np.zeros((flat.shape[0], max_dim - flat.shape[1]), dtype=np.float32)
            flat_padded[name] = np.concatenate([flat, pad], axis=1)

    return flat_padded, n_steps


def project_trajectories(
    trackers: Dict[str, LatentTrajectoryCollector],
    method: str = "pca",
) -> Dict:
    """Flatten trajectory latents and project jointly to 2D."""

    flat_trajs, n_steps = _flatten_trackers_padded(trackers)
    condition_names = list(trackers.keys())
    all_flat = [flat_trajs[name] for name in condition_names]

    # Stack: (num_conditions * (T+1), D)
    stacked = np.vstack(all_flat).astype(np.float32)

    if method == "pca":
        projector = PCA(n_components=2)
        projected = projector.fit_transform(stacked)
        explained = projector.explained_variance_ratio_
    elif method == "mds":
        # Use precomputed Euclidean distance matrix for efficiency.
        # Direct MDS on 262K-dim vectors is slow; distance matrix is N×N
        # where N = num_conditions * (T+1), typically ~200-400 points.
        from sklearn.metrics import pairwise_distances
        dist_matrix = pairwise_distances(stacked, metric="euclidean")
        projector = MDS(
            n_components=2, random_state=42,
            dissimilarity="precomputed",
            normalized_stress="auto",
        )
        projected = projector.fit_transform(dist_matrix)
        stress = projector.stress_
        print(f"  MDS stress: {stress:.4f}")
        explained = None
    else:
        raise ValueError(f"Unknown projection method: {method}")

    # Split back per condition
    result = {}
    for i, name in enumerate(condition_names):
        start = i * n_steps
        end = start + n_steps
        result[name] = projected[start:end]  # (T+1, 2)

    return result, explained, n_steps


# ---------------------------------------------------------------------------
# Visualization: Main trajectory manifold plot (reference image style)
# ---------------------------------------------------------------------------
# Endpoint markers and colors cycled when uniform_color is enabled.
# All lines stay solid — only endpoints differ.
_UNIFORM_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
_UNIFORM_ENDPOINT_COLORS = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
    "#9b5de5", "#f4a261", "#264653", "#d62828",
]


def _compact_manifold_annotation_label(name: str, label: str) -> str:
    """Keep guided endpoint annotations compact; leave legend labels unchanged."""
    if not name.startswith("guided_"):
        return label

    alpha_val = None

    # Preferred parse path from label text.
    for token in ("alpha=", "α="):
        if token in label:
            start = label.find(token) + len(token)
            end = start
            while end < len(label) and (label[end].isdigit() or label[end] in ".-+"):
                end += 1
            raw = label[start:end]
            try:
                alpha_val = float(raw)
            except ValueError:
                alpha_val = None
            break

    # Fallback parse from condition key: guided_a030 -> 0.30
    if alpha_val is None and name.startswith("guided_a"):
        raw = name[len("guided_a"):]
        if raw.isdigit():
            try:
                alpha_val = float(raw) / 100.0
            except ValueError:
                alpha_val = None

    if alpha_val is None:
        return "G"
    return f"G(α={alpha_val:g})"


def plot_trajectory_manifold(
    projected: Dict[str, np.ndarray],
    labels: Dict[str, str],
    n_steps: int,
    output_path: str,
    title: str = "Latent Trajectory Dynamics",
    projection_method: str = "pca",
    uniform_color: bool = False,
    prompt_key_map: Optional[Dict[str, str]] = None,
):
    """Plot all trajectories on one figure with time-gradient coloring.

    Args:
        uniform_color: If True, all conditions share the same colormap (viridis)
            for time progression. Conditions are differentiated by endpoint
            marker shapes and colors (lines are all solid).
    """

    fig, ax = plt.subplots(figsize=(10, 8))
    if prompt_key_map:
        key_text = _format_prompt_key_text(prompt_key_map)
        fig.text(0.5, 0.99, f"Prompt Key: {key_text}", ha="center", va="top", fontsize=9)

    norm = Normalize(vmin=0, vmax=n_steps - 1)
    condition_names = list(projected.keys())
    shared_cmap_name = "viridis"

    for idx, name in enumerate(condition_names):
        pts = projected[name]  # (T+1, 2)

        if uniform_color:
            cmap = cm.get_cmap(shared_cmap_name)
        else:
            cmap = cm.get_cmap(CONDITION_CMAPS.get(name, "viridis"))

        # Build line segments for LineCollection (always solid)
        points = pts.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=2.5,
            alpha=0.9,
        )
        lc.set_array(np.arange(len(segments)))
        ax.add_collection(lc)

        # Endpoint marker + label
        label = labels.get(name, name)
        ann_label = _compact_manifold_annotation_label(name, label)
        if uniform_color:
            marker = _UNIFORM_MARKERS[idx % len(_UNIFORM_MARKERS)]
            ep_color = _UNIFORM_ENDPOINT_COLORS[idx % len(_UNIFORM_ENDPOINT_COLORS)]
            ax.plot(pts[-1, 0], pts[-1, 1], marker=marker, color=ep_color,
                    markersize=12, markeredgecolor="black", markeredgewidth=1.0,
                    zorder=6, linestyle="none")
            ann_color = ep_color
            arrow_color = ep_color
        else:
            ann_color = cmap(0.85)
            arrow_color = cmap(0.6)

        ax.annotate(
            ann_label,
            xy=(pts[-1, 0], pts[-1, 1]),
            fontsize=10,
            fontweight="bold",
            color=ann_color,
            textcoords="offset points",
            xytext=(8, 4),
            arrowprops=dict(arrowstyle="-", color=arrow_color, lw=0.8),
        )

    # Mark shared origin
    origin = projected[condition_names[0]][0]
    ax.plot(origin[0], origin[1], "ko", markersize=8, zorder=5)
    ax.annotate(
        r"$x_T$",
        xy=(origin[0], origin[1]),
        fontsize=11,
        fontweight="bold",
        textcoords="offset points",
        xytext=(-15, -12),
    )

    # Shared colorbar for time
    cbar_cmap = shared_cmap_name if uniform_color else "coolwarm"
    sm = cm.ScalarMappable(cmap=cbar_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Time (steps)", fontsize=11)

    # Axis labels based on projection method
    ax_prefix = "MDS" if projection_method == "mds" else "PC"
    ax.autoscale()
    ax.set_xlabel(f"{ax_prefix} 1", fontsize=12)
    ax.set_ylabel(f"{ax_prefix} 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = []
    for idx, name in enumerate(condition_names):
        if uniform_color:
            marker = _UNIFORM_MARKERS[idx % len(_UNIFORM_MARKERS)]
            ep_color = _UNIFORM_ENDPOINT_COLORS[idx % len(_UNIFORM_ENDPOINT_COLORS)]
            legend_handles.append(
                Line2D([0], [0], color=cm.get_cmap(shared_cmap_name)(0.5),
                       lw=2.5, marker=marker, markerfacecolor=ep_color,
                       markeredgecolor="black", markersize=8,
                       label=labels.get(name, name))
            )
        else:
            cmap = cm.get_cmap(CONDITION_CMAPS.get(name, "viridis"))
            legend_handles.append(
                Line2D([0], [0], color=cmap(0.7), lw=2.5,
                       label=labels.get(name, name))
            )
    ax.legend(handles=legend_handles, loc="best", fontsize=9)

    top = 0.90 if prompt_key_map else 0.98
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Visualization: Per-condition subplots
# ---------------------------------------------------------------------------
def plot_trajectory_subplots(
    projected: Dict[str, np.ndarray],
    labels: Dict[str, str],
    n_steps: int,
    output_path: str,
    projection_method: str = "pca",
    uniform_color: bool = False,
    prompt_key_map: Optional[Dict[str, str]] = None,
):
    """One subplot per condition, all sharing the same axis range."""

    condition_names = list(projected.keys())
    n_cond = len(condition_names)
    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 5))
    if n_cond == 1:
        axes = [axes]

    norm = Normalize(vmin=0, vmax=n_steps - 1)

    # Compute shared axis limits
    all_pts = np.vstack(list(projected.values()))
    margin = 0.1 * (all_pts.max(0) - all_pts.min(0))
    xlim = (all_pts[:, 0].min() - margin[0], all_pts[:, 0].max() + margin[0])
    ylim = (all_pts[:, 1].min() - margin[1], all_pts[:, 1].max() + margin[1])

    ax_prefix = "MDS" if projection_method == "mds" else "PC"
    shared_cmap_name = "viridis"

    for idx, (ax, name) in enumerate(zip(axes, condition_names)):
        pts = projected[name]

        if uniform_color:
            cmap = cm.get_cmap(shared_cmap_name)
        else:
            cmap = cm.get_cmap(CONDITION_CMAPS.get(name, "viridis"))

        points = pts.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.0)
        lc.set_array(np.arange(len(segments)))
        ax.add_collection(lc)

        # Origin and endpoint
        ax.plot(pts[0, 0], pts[0, 1], "ko", markersize=6, zorder=5)
        ax.plot(pts[-1, 0], pts[-1, 1], "s", color=cmap(0.85),
                markersize=8, zorder=5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(labels.get(name, name), fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{ax_prefix} 1", fontsize=10)
        ax.set_ylabel(f"{ax_prefix} 2", fontsize=10)
        ax.grid(True, alpha=0.3)

    if prompt_key_map:
        key_text = _format_prompt_key_text(prompt_key_map)
        fig.text(0.5, 0.99, f"Prompt Key: {key_text}", ha="center", va="top", fontsize=9)
    suptitle_y = 0.92 if prompt_key_map else 0.98
    top = 0.84 if prompt_key_map else 0.93
    fig.suptitle("Per-Condition Trajectory Dynamics", fontsize=13, fontweight="bold", y=suptitle_y)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Visualization: Pairwise distances over time
# ---------------------------------------------------------------------------
def plot_pairwise_distances(
    trackers: Dict[str, LatentTrajectoryCollector],
    labels: Dict[str, str],
    output_path: str,
):
    """Plot L2 distance between each pair of trajectories over time."""

    condition_names = list(trackers.keys())
    n_cond = len(condition_names)

    # Flatten trajectories: (T+1, D) per condition, padded to common D if needed
    flat_np, _ = _flatten_trackers_padded(trackers)
    flat_trajs = {
        name: torch.from_numpy(flat_np[name]).float()
        for name in condition_names
    }

    n_steps = flat_trajs[condition_names[0]].shape[0]
    pairs = [(i, j) for i in range(n_cond) for j in range(i + 1, n_cond)]

    fig, ax = plt.subplots(figsize=(10, 6))
    pair_colors = plt.cm.Set2(np.linspace(0, 1, len(pairs)))

    for idx, (i, j) in enumerate(pairs):
        name_i, name_j = condition_names[i], condition_names[j]
        dist = torch.norm(flat_trajs[name_i] - flat_trajs[name_j], dim=1).numpy()
        label_pair = f"{labels.get(name_i, name_i)} vs {labels.get(name_j, name_j)}"
        ax.plot(dist, color=pair_colors[idx], linewidth=1.8, label=label_pair)

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("L2 Distance (latent space)", fontsize=12)
    ax.set_title("Pairwise Trajectory Distances Over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Visualization: Decoded image strip
# ---------------------------------------------------------------------------
def plot_decoded_images(
    final_latents: Dict[str, torch.Tensor],
    labels: Dict[str, str],
    vae_or_map,
    output_path: str,
    prompt_key_map: Optional[Dict[str, str]] = None,
):
    """Decode final latents and show as a strip."""
    from PIL import Image as PILImage

    condition_names = list(final_latents.keys())
    images = []
    for name in condition_names:
        vae = vae_or_map[name] if isinstance(vae_or_map, dict) else vae_or_map
        img = get_image(vae, final_latents[name], 1, 1)
        images.append(img)

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, img, name in zip(axes, images, condition_names):
        ax.imshow(img)
        ax.set_title(labels.get(name, name), fontsize=10, fontweight="bold")
        ax.axis("off")

    if prompt_key_map:
        key_text = _format_prompt_key_text(prompt_key_map)
        fig.text(0.5, 0.99, f"Prompt Key: {key_text}", ha="center", va="top", fontsize=9)
    suptitle_y = 0.92 if prompt_key_map else 0.98
    top = 0.85 if prompt_key_map else 0.93
    fig.suptitle("Decoded Final Latents", fontsize=13, fontweight="bold", y=suptitle_y)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Quantitative summary
# ---------------------------------------------------------------------------
def compute_summary(
    trackers: Dict[str, LatentTrajectoryCollector],
    projected: Dict[str, np.ndarray],
    explained_variance: Optional[np.ndarray],
    labels: Dict[str, str],
) -> dict:
    """Compute quantitative metrics and return as dict."""

    condition_names = list(trackers.keys())

    # Flatten trajectories (padded to common D if needed)
    flat_np, _ = _flatten_trackers_padded(trackers)
    flat_trajs = {
        name: torch.from_numpy(flat_np[name]).float()
        for name in condition_names
    }

    n_steps = flat_trajs[condition_names[0]].shape[0]
    summary = {}

    # PCA variance explained
    if explained_variance is not None:
        summary["pca_variance_explained"] = {
            f"PC{i+1}": float(v) for i, v in enumerate(explained_variance)
        }
        summary["pca_total_variance_top2"] = float(explained_variance[:2].sum())

    # Pairwise endpoint distances (original space)
    endpoint_dists = {}
    for i, ni in enumerate(condition_names):
        for j, nj in enumerate(condition_names):
            if i >= j:
                continue
            d = torch.norm(flat_trajs[ni][-1] - flat_trajs[nj][-1]).item()
            key = f"{labels.get(ni, ni)} vs {labels.get(nj, nj)}"
            endpoint_dists[key] = d
    summary["endpoint_distances_l2"] = endpoint_dists

    # Pairwise endpoint distances (projected space)
    proj_dists = {}
    for i, ni in enumerate(condition_names):
        for j, nj in enumerate(condition_names):
            if i >= j:
                continue
            d = np.linalg.norm(projected[ni][-1] - projected[nj][-1])
            key = f"{labels.get(ni, ni)} vs {labels.get(nj, nj)}"
            proj_dists[key] = float(d)
    summary["endpoint_distances_projected"] = proj_dists

    # Divergence onset: first step where any pairwise distance > 1% of max
    divergence_onsets = {}
    for i, ni in enumerate(condition_names):
        for j, nj in enumerate(condition_names):
            if i >= j:
                continue
            dists = torch.norm(flat_trajs[ni] - flat_trajs[nj], dim=1)
            max_d = dists.max().item()
            threshold = 0.01 * max_d
            onset = int((dists > threshold).nonzero(as_tuple=True)[0][0].item()) if (dists > threshold).any() else n_steps
            key = f"{labels.get(ni, ni)} vs {labels.get(nj, nj)}"
            divergence_onsets[key] = onset
    summary["divergence_onset_step"] = divergence_onsets

    # Total path length per condition
    path_lengths = {}
    for name in condition_names:
        deltas = flat_trajs[name][1:] - flat_trajs[name][:-1]
        total = torch.norm(deltas, dim=1).sum().item()
        path_lengths[labels.get(name, name)] = total
    summary["total_path_length"] = path_lengths

    # Cosine similarity of final tangent vectors (last step direction)
    tangent_cosines = {}
    tangents = {}
    for name in condition_names:
        t = flat_trajs[name][-1] - flat_trajs[name][-2]
        tangents[name] = t / (t.norm() + 1e-8)
    for i, ni in enumerate(condition_names):
        for j, nj in enumerate(condition_names):
            if i >= j:
                continue
            cos = torch.dot(tangents[ni], tangents[nj]).item()
            key = f"{labels.get(ni, ni)} vs {labels.get(nj, nj)}"
            tangent_cosines[key] = cos
    summary["final_tangent_cosine_similarity"] = tangent_cosines

    return summary


# ---------------------------------------------------------------------------
# CLIP Classifier Probe
# ---------------------------------------------------------------------------
@torch.no_grad()
def clip_classifier_probe(
    final_latents: Dict[str, torch.Tensor],
    labels: Dict[str, str],
    vae_or_map,
    output_path: str,
    class_prompts: Optional[Dict[str, str]] = None,
    clip_model_id: str = "openai/clip-vit-large-patch14",
) -> dict:
    """
    Zero-shot CLIP classification of decoded outputs.

    Encodes each decoded image with CLIP ViT-L/14, computes cosine similarity
    against text prototypes derived from the active prompt set, and reports
    which class each condition's output is closest to.

    Also projects all image embeddings + text prototypes into a shared 2D
    space to visualize semantic positioning.
    """
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image as PILImage

    condition_names = list(final_latents.keys())

    if class_prompts is None:
        # Fallback: use condition labels as text prototypes.
        class_prompts = {
            labels.get(name, name): labels.get(name, name).strip('"')
            for name in condition_names
        }

    device = next(iter(final_latents.values())).device

    # Load CLIP
    print("  Loading CLIP ViT-L/14 for classifier probe ...")
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    # --- Text prototypes ---
    class_names = list(class_prompts.keys())
    class_texts = list(class_prompts.values())
    text_inputs = clip_processor(text=class_texts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_embeds = clip_model.get_text_features(**text_inputs)  # (num_classes, 768)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # --- Decode latents → images → CLIP image embeddings ---
    image_embeds_dict = {}
    decoded_pil = {}

    for name in condition_names:
        vae = vae_or_map[name] if isinstance(vae_or_map, dict) else vae_or_map
        img_pil = get_image(vae, final_latents[name], 1, 1)
        decoded_pil[name] = img_pil
        img_inputs = clip_processor(images=img_pil, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        img_emb = clip_model.get_image_features(**img_inputs)  # (1, 768)
        img_emb = F.normalize(img_emb, dim=-1)
        image_embeds_dict[name] = img_emb.cpu()

    # --- Cosine similarity matrix ---
    # Rows = conditions, Columns = class prototypes
    sim_matrix = {}
    predictions = {}
    for name in condition_names:
        sims = (image_embeds_dict[name].to(device) @ text_embeds.T).squeeze(0)  # (num_classes,)
        sim_dict = {cn: float(sims[i].item()) for i, cn in enumerate(class_names)}
        sim_matrix[labels.get(name, name)] = sim_dict
        predicted_class = class_names[sims.argmax().item()]
        predictions[labels.get(name, name)] = predicted_class

    # --- Visualization: similarity heatmap + 2D projection ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Cosine similarity heatmap
    cond_labels = [labels.get(n, n) for n in condition_names]
    sim_arr = np.array([
        [sim_matrix[labels.get(n, n)][cn] for cn in class_names]
        for n in condition_names
    ])

    import seaborn as sns
    sns.heatmap(
        sim_arr, ax=axes[0],
        xticklabels=class_names, yticklabels=cond_labels,
        annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=sim_arr.min() - 0.02, vmax=sim_arr.max() + 0.02,
    )
    axes[0].set_title("CLIP Cosine Similarity\n(Image vs Text Prototype)", fontsize=11)
    axes[0].set_xlabel("Text Prototype")
    axes[0].set_ylabel("Condition")

    # Panel 2: Joint PCA of image + text embeddings in CLIP space
    all_embeds = []
    embed_labels = []
    embed_types = []  # "image" or "text"

    for name in condition_names:
        all_embeds.append(image_embeds_dict[name].squeeze(0).numpy())
        embed_labels.append(labels.get(name, name))
        embed_types.append("image")

    for cn, ct in zip(class_names, class_texts):
        all_embeds.append(text_embeds[class_names.index(cn)].cpu().numpy())
        embed_labels.append(f"[TEXT] {cn}")
        embed_types.append("text")

    all_embeds_arr = np.stack(all_embeds)
    pca_clip = PCA(n_components=2)
    proj_clip = pca_clip.fit_transform(all_embeds_arr)

    # Plot image embeddings
    img_mask = np.array([t == "image" for t in embed_types])
    txt_mask = np.array([t == "text" for t in embed_types])

    cond_colors = [CONDITION_COLORS.get(n, "#333333") for n in condition_names]
    axes[1].scatter(
        proj_clip[img_mask, 0], proj_clip[img_mask, 1],
        c=cond_colors, s=120, marker="o", edgecolors="black", linewidths=0.8,
        zorder=5,
    )
    # Plot text prototypes
    axes[1].scatter(
        proj_clip[txt_mask, 0], proj_clip[txt_mask, 1],
        c="#888888", s=80, marker="^", edgecolors="black", linewidths=0.8,
        zorder=5,
    )

    # Annotate
    for i, (lbl, typ) in enumerate(zip(embed_labels, embed_types)):
        short_lbl = lbl.replace("[TEXT] ", "T:") if typ == "text" else lbl
        # Truncate long labels
        if len(short_lbl) > 30:
            short_lbl = short_lbl[:27] + "..."
        axes[1].annotate(
            short_lbl, (proj_clip[i, 0], proj_clip[i, 1]),
            fontsize=7, textcoords="offset points", xytext=(5, 5),
        )

    axes[1].set_xlabel(f"CLIP PC1 ({pca_clip.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"CLIP PC2 ({pca_clip.explained_variance_ratio_[1]:.1%})")
    axes[1].set_title("CLIP Embedding Space\n(Image + Text Prototypes)", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("CLIP Classifier Probe", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Cleanup CLIP model
    del clip_model, clip_processor
    torch.cuda.empty_cache()

    return {
        "cosine_similarities": sim_matrix,
        "predictions": predictions,
        "clip_pca_variance_explained": {
            f"PC{i+1}": float(v) for i, v in enumerate(pca_clip.explained_variance_ratio_)
        },
    }


# ---------------------------------------------------------------------------
# Trajectory data serialiser
# ---------------------------------------------------------------------------
def save_trajectory_data(
    trackers: Dict,
    kappa_data: Dict,
    projected: Dict,
    explained,
    summary: dict,
    labels: Dict,
    conditions: Dict,
    cfg,
    output_dir: str,
    clip_results: dict = None,
):
    """
    Serialise all data needed by plot_trajectory_analysis.py into
    trajectory_data.json alongside the other experiment outputs.

    Per-condition scalars (latent_norms, velocity_magnitudes) are computed
    here from the LatentTrajectoryCollector tensors so that the plotting
    script is fully offline (no models needed).
    """
    condition_names = list(trackers.keys())

    # ---- Flatten trackers to (T+1, D) per condition ----
    flat_np = {}
    for name, tracker in trackers.items():
        traj = tracker.trajectories        # (T+1, B, C, H, W)
        T1 = traj.shape[0]
        flat_np[name] = traj[:, 0].reshape(T1, -1).float().numpy()

    # ---- Per-condition stats ----
    per_condition = {}
    for name, tracker in trackers.items():
        T1 = tracker.trajectories.shape[0]

        # Latent norms: ||x_t||_2
        lat = flat_np[name]                                     # (T+1, D)
        lat_norms = np.linalg.norm(lat, axis=1).tolist()

        # Velocity magnitudes: ||v_t||_2  (last entry is zeros — store_final has no vel)
        vel = tracker.velocities[:, 0].reshape(T1, -1).float().numpy()
        vel_mags = np.linalg.norm(vel, axis=1).tolist()

        # Projected coordinates (2D)
        proj = projected.get(name)
        proj_x = proj[:, 0].tolist() if proj is not None else []
        proj_y = proj[:, 1].tolist() if proj is not None else []

        entry = {
            "latent_norms":        lat_norms,
            "velocity_magnitudes": vel_mags,
            "projected_x":         proj_x,
            "projected_y":         proj_y,
            "kappa":               kappa_data.get(name, {}).get("kappa"),
            "log_likelihoods":     kappa_data.get(name, {}).get("log_likelihoods"),
            "concept_labels":      kappa_data.get(name, {}).get("concept_labels"),
        }
        per_condition[name] = entry

    # ---- Pairwise L2 distances over time ----
    pairs = [(i, j) for i in range(len(condition_names))
             for j in range(i + 1, len(condition_names))]
    pairwise_l2 = {}
    for i, j in pairs:
        ni, nj = condition_names[i], condition_names[j]
        li = flat_np.get(ni)
        lj = flat_np.get(nj)
        if li is None or lj is None:
            continue
        min_T = min(li.shape[0], lj.shape[0])
        dist = np.linalg.norm(li[:min_T] - lj[:min_T], axis=1).tolist()
        pairwise_l2[f"{ni}|{nj}"] = dist

    # ---- CLIP cosine similarities (terminal only) ----
    clip_terminal = {}
    if clip_results and "cosine_similarities" in clip_results:
        for cond_name, sims in clip_results["cosine_similarities"].items():
            clip_terminal[cond_name] = sims

    # ---- PCA variance explained ----
    pca_var = []
    mds_stress = None
    if explained is not None:
        pca_var = [round(float(v), 6) for v in explained]
    elif hasattr(cfg, "_mds_stress"):
        mds_stress = cfg._mds_stress

    data = {
        "conditions":            condition_names,
        "labels":                labels,
        "condition_types":       {n: conditions[n].get("type", "unknown")
                                   for n in condition_names if n in conditions},
        "n_steps":               int(flat_np[condition_names[0]].shape[0]) if condition_names else 0,
        "projection_method":     cfg.projection,
        "pca_variance_explained": pca_var,
        "mds_stress":            mds_stress,
        "config": {
            "prompt_a":            cfg.prompt_a,
            "prompt_b":            cfg.prompt_b,
            "monolithic_prompt":   cfg.monolithic_prompt,
            "model_id":            cfg.model_id,
            "num_inference_steps": cfg.num_inference_steps,
            "guidance_scale":      cfg.guidance_scale,
            "seed":                cfg.seed,
            "lift":                cfg.lift,
        },
        "decoded_images_path": "decoded_images.png",
        "per_condition":       per_condition,
        "pairwise_l2":         pairwise_l2,
        "summary":             {
            k: v for k, v in summary.items()
            if k in ("endpoint_distances_l2", "path_lengths",
                     "divergence_onset_step", "final_tangent_cosine_similarity",
                     "pca_variance_explained", "pca_total_variance_top2")
        },
        "clip": clip_terminal,
    }

    out_path = os.path.join(output_dir, "trajectory_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(cfg: TrajectoryExperimentConfig):
    """Run the full trajectory dynamics experiment."""

    # Output directory
    if not cfg.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = str(
            PROJECT_ROOT / "experiments" / "trajectory_dynamics" / timestamp
        )
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Output directory: {cfg.output_dir}")

    # Resolve active prompt set:
    # - If --prompts is provided, it overrides prompt_a/prompt_b.
    # - Otherwise use prompt_a + prompt_b.
    cli_prompts = [p.strip() for p in cfg.multi_prompts if p and p.strip()]

    device = torch.device(cfg.device)
    labels: Dict[str, str] = {}
    conditions: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Solo mode: single prompt through standard SD3.5 (no composition)
    # ------------------------------------------------------------------
    if cfg.solo:
        # Accept prompt from --monolithic, --prompts (single), or --prompt-a
        if cli_prompts and len(cli_prompts) == 1:
            solo_prompt = cli_prompts[0]
        elif cli_prompts and len(cli_prompts) > 1:
            solo_prompt = " and ".join(cli_prompts)
            print(f"  Solo mode: joining {len(cli_prompts)} prompts into one monolithic prompt.")
        elif cfg.monolithic_prompt:
            solo_prompt = cfg.monolithic_prompt
        else:
            solo_prompt = cfg.prompt_a.strip()
        if not solo_prompt:
            raise ValueError("Solo mode requires a non-empty prompt via --monolithic, --prompts, or --prompt-a.")

        cfg.monolithic_prompt = solo_prompt
        cfg.prompt_a = solo_prompt
        cfg.prompt_b = ""
        active_prompts = [solo_prompt]
        print(f"  Solo mode: \"{solo_prompt}\"")

        model_tag = "GLIGEN" if cfg.gligen else "SD3.5"
        labels["monolithic"] = f'{model_tag}: "{solo_prompt}"'
        conditions["monolithic"] = {"type": "standard", "prompt": solo_prompt}

        clip_class_prompts = {f"[1] {solo_prompt}": solo_prompt}

    else:
        # Normal multi-prompt mode
        if cfg.multi_prompts and len(cli_prompts) < 2:
            raise ValueError("--prompts requires at least 2 non-empty prompt strings.")

        active_prompts = cli_prompts if cli_prompts else [cfg.prompt_a.strip(), cfg.prompt_b.strip()]
        if any(not p for p in active_prompts):
            raise ValueError("All prompts must be non-empty strings.")
        if len(active_prompts) < 2:
            raise ValueError(f"Need at least 2 prompts, got {len(active_prompts)}")

        # Keep prompt_a/prompt_b aligned for legacy summary fields.
        cfg.prompt_a = active_prompts[0]
        cfg.prompt_b = active_prompts[1]

        # Auto-derive monolithic prompt if not explicitly set
        if not cfg.monolithic_prompt:
            cfg.monolithic_prompt = _join_prompts_with_and(active_prompts)
        print(f"  Monolithic prompt: \"{cfg.monolithic_prompt}\"")

        # Build CLIP class prototypes from the active prompt set.
        clip_class_prompts = _build_clip_class_prompts(active_prompts, cfg.monolithic_prompt)

    n_prompts = len(active_prompts)

    # Notebook stochastic backend is defined for pairwise SD (v1.4) only.
    if (not cfg.solo) and cfg.superdiff_variant == "author_stoch" and n_prompts != 2:
        raise ValueError(
            "--superdiff-variant author_stoch is only supported in 2-prompt mode "
            "(the notebook formulation is pairwise)."
        )

    # Prompt mode A: exactly 2 prompts -> pairwise comparisons (PoE optional)
    if not cfg.solo and n_prompts == 2:
        prompt_a, prompt_b = active_prompts

        labels.update({
            "prompt_a": f'"{prompt_a}"',
            "prompt_b": f'"{prompt_b}"',
            "monolithic": f'CLIP AND: "{cfg.monolithic_prompt}"',
        })
        conditions.update({
            "prompt_a": {"type": "standard", "prompt": prompt_a},
            "prompt_b": {"type": "standard", "prompt": prompt_b},
            "monolithic": {"type": "standard", "prompt": cfg.monolithic_prompt},
        })

        if not cfg.no_poe:
            labels["poe"] = f'PoE: "{prompt_a}" \u00d7 "{prompt_b}"'
            conditions["poe"] = {
                "type": "poe",
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
            }

        # Add SuperDIFF variant(s) — not available in GLIGEN mode
        if cfg.gligen:
            print("  GLIGEN mode: SuperDIFF variants skipped (incompatible paradigm).")
        else:
            sd_prompts = {"obj_prompt": prompt_a, "bg_prompt": prompt_b}
            variant = cfg.superdiff_variant
            if variant == "all":
                print("  NOTE: 'all' includes SD3 variants {ours, author_det, fm_ode}. "
                      "author_stoch is excluded because it uses a different (SD v1.4) backend.")

            if variant in ("ours", "all"):
                conditions["superdiff"] = {"type": "superdiff", **sd_prompts}
                labels["superdiff"] = f'SuperDIFF (ours): "{prompt_a}" \u2227 "{prompt_b}"'

            if variant in ("author_det", "all"):
                # Warn: JVP through SD3.5 transformer roughly doubles VRAM per forward pass
                free_mem = torch.cuda.mem_get_info(device)[0] / (1024 ** 3)
                print(f"  WARNING: author_det uses JVP (torch.func.jvp) through the SD3 transformer,")
                print(f"           requiring ~2x VRAM per forward pass. Free VRAM: {free_mem:.1f} GiB.")
                print(f"           If OOM occurs, use --superdiff-variant fm_ode instead.")
                conditions["superdiff_det"] = {"type": "superdiff_author_det", **sd_prompts}
                labels["superdiff_det"] = f'SuperDIFF (author, det): "{prompt_a}" \u2227 "{prompt_b}"'

            if variant == "author_stoch":
                print("  NOTE: author_stoch injects stochastic noise each step "
                      f"(matches superposition_AND notebook on {AUTHOR_STOCH_NOTEBOOK_MODEL_ID}).")
                conditions["superdiff_stoch"] = {"type": "superdiff_author_stoch", **sd_prompts}
                labels["superdiff_stoch"] = f'SuperDIFF (author, stoch): "{prompt_a}" \u2227 "{prompt_b}"'

            if variant in ("fm_ode", "all"):
                conditions["superdiff_fm_ode"] = {"type": "superdiff_fm_ode", **sd_prompts}
                labels["superdiff_fm_ode"] = f'SuperDIFF (FM-ODE): "{prompt_a}" \u2227 "{prompt_b}"'

            # SuperDIFF-Guided hybrid (2-prompt) — one condition per alpha
            if cfg.guided:
                if variant == "author_stoch":
                    print("  NOTE: --guided is SD3-only; skipping guided conditions for author_stoch.")
                else:
                    _register_guided_conditions(
                        conditions, labels, cfg.alphas,
                        [prompt_a, prompt_b], cfg.monolithic_prompt,
                    )

    # Prompt mode B: 3+ prompts -> each prompt + monolithic + multi-AND
    elif not cfg.solo:
        prompt_cmaps = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]
        prompt_colors = ["#e63946", "#457b9d", "#2a9d8f", "#9b5de5", "#f4a261", "#6c757d"]

        for idx, prompt in enumerate(active_prompts, start=1):
            key = f"prompt_{idx}"
            labels[key] = f'"{prompt}"'
            conditions[key] = {"type": "standard", "prompt": prompt}
            CONDITION_CMAPS[key] = prompt_cmaps[(idx - 1) % len(prompt_cmaps)]
            CONDITION_COLORS[key] = prompt_colors[(idx - 1) % len(prompt_colors)]

        labels["monolithic"] = f'CLIP AND: "{cfg.monolithic_prompt}"'
        conditions["monolithic"] = {"type": "standard", "prompt": cfg.monolithic_prompt}

        prompt_str = " \u2227 ".join(f'"{p}"' for p in active_prompts)

        if cfg.gligen:
            print(f"  GLIGEN mode: SuperDIFF multi-AND skipped (incompatible paradigm).")
        else:
            labels["superdiff_multi"] = f"SuperDIFF Multi-AND: {prompt_str}"
            conditions["superdiff_multi"] = {
                "type": "superdiff_multi",
                "prompts": active_prompts,
            }
            print(f"  Multi-prompt AND ({n_prompts} prompts): {prompt_str}")

            # SuperDIFF-Guided hybrid (multi-prompt) — one condition per alpha
            if cfg.guided:
                _register_guided_conditions(
                    conditions, labels, cfg.alphas,
                    active_prompts, cfg.monolithic_prompt,
                )

        if not cfg.no_poe:
            print("  NOTE: PoE is only defined for 2-prompt mode; skipping PoE for multi-prompt run.")
        if not cfg.gligen and cfg.superdiff_variant != "fm_ode":
            print("  NOTE: --superdiff-variant applies only to 2-prompt mode; using multi-prompt FM-ODE.")

    # ------------------------------------------------------------------
    # Concept Negation (NOT) — appended to whichever mode is active
    # ------------------------------------------------------------------
    if cfg.neg_prompt and cfg.gligen:
        print("  GLIGEN mode: concept negation (NOT) skipped (incompatible paradigm).")
    elif cfg.neg_prompt and cfg.superdiff_variant == "author_stoch":
        print("  NOTE: NOT mode is SD3-only; skipping NOT conditions for author_stoch.")
    elif cfg.neg_prompt:
        neg_prompt = cfg.neg_prompt.strip()
        pos_prompt = cfg.prompt_a  # the primary positive concept
        print(f'  NOT mode: "{pos_prompt}" NOT "{neg_prompt}"')

        # In solo mode, the "monolithic" condition is already the positive prompt.
        # In non-solo mode (2+ prompts), add the positive prompt explicitly
        # so we can compare: pos-only vs without-phrasing vs composable NOT vs SuperDIFF NOT.
        if not cfg.solo and "monolithic" not in conditions:
            labels["pos_only"] = f'SD3.5: "{pos_prompt}"'
            conditions["pos_only"] = {"type": "standard", "prompt": pos_prompt}

        # The "without" monolithic prompt — the baseline that typically fails
        without_prompt = f"{pos_prompt} without {neg_prompt}"
        labels["monolithic_not"] = f'SD3.5: "{without_prompt}"'
        conditions["monolithic_not"] = {"type": "standard", "prompt": without_prompt}

        # Composable NOT (Liu et al., Eq 13): fixed negative guidance
        labels["composable_not"] = (
            f'Composable NOT: "{pos_prompt}" \u00ac"{neg_prompt}"'
        )
        conditions["composable_not"] = {
            "type": "composable_not",
            "pos_prompt": pos_prompt,
            "neg_prompt": neg_prompt,
            "neg_scale": cfg.neg_scale,
        }

        # SuperDIFF NOT: dynamic kappa negation
        labels["superdiff_not"] = (
            f'SuperDIFF NOT (\u03bb={cfg.neg_lambda}): '
            f'"{pos_prompt}" \u00ac"{neg_prompt}"'
        )
        conditions["superdiff_not"] = {
            "type": "superdiff_not",
            "pos_prompt": pos_prompt,
            "neg_prompt": neg_prompt,
            "neg_lambda": cfg.neg_lambda,
        }

        # Update CLIP class prompts to include the negated concept
        clip_class_prompts[f"[+] {pos_prompt}"] = pos_prompt
        clip_class_prompts[f"[-] {neg_prompt}"] = neg_prompt
        clip_class_prompts[f"[without] {without_prompt}"] = without_prompt

    # Plot-facing labels: compact symbolic form + prompt key legend
    viz_labels, prompt_key_map = _build_compact_visual_labels(
        conditions=conditions,
        base_prompts=active_prompts,
        monolithic_prompt=cfg.monolithic_prompt,
        model_tag=("GLIGEN" if cfg.gligen else "SD3.5"),
    )

    # Load model(s)
    use_author_stoch_backend = (not cfg.gligen) and (cfg.superdiff_variant == "author_stoch")
    legacy_models = None

    if cfg.gligen:
        print(f"Loading GLIGEN ({cfg.gligen_model_id}) ...")
        models = get_gligen_models(
            model_id=cfg.gligen_model_id,
            dtype=cfg.dtype,
            device=device,
        )
        # GLIGEN uses 4-channel latents, 64x64 for 512x512 output
        cfg.z_channels = 4
        cfg.latent_height = 64
        cfg.latent_width = 64
    else:
        print("Loading SD3.5 Medium ...")
        models = get_sd3_models(
            model_id=cfg.model_id,
            dtype=cfg.dtype,
            device=device,
        )
        if use_author_stoch_backend:
            print(f"Loading notebook-compatible author_stoch backend ({AUTHOR_STOCH_NOTEBOOK_MODEL_ID}) ...")
            legacy_models = get_sd_models(
                model_id=AUTHOR_STOCH_NOTEBOOK_MODEL_ID,
                dtype=cfg.dtype,
                device=device,
            )
    print("  Model loaded.")

    # Collect trajectories
    print("Collecting trajectories ...")
    if use_author_stoch_backend:
        sd3_conditions = {
            name: cond for name, cond in conditions.items()
            if cond["type"] != "superdiff_author_stoch"
        }
        legacy_conditions = {
            name: cond for name, cond in conditions.items()
            if cond["type"] == "superdiff_author_stoch"
        }

        trackers_sd3, final_sd3, kappa_sd3 = {}, {}, {}
        if sd3_conditions:
            trackers_sd3, final_sd3, kappa_sd3 = collect_trajectories(cfg, models, sd3_conditions)

        trackers_legacy, final_legacy, kappa_legacy = {}, {}, {}
        if legacy_conditions:
            cfg_legacy = replace(
                cfg,
                model_id=AUTHOR_STOCH_NOTEBOOK_MODEL_ID,
                z_channels=4,
                latent_height=64,
                latent_width=64,
            )
            trackers_legacy, final_legacy, kappa_legacy = collect_trajectories(
                cfg_legacy, legacy_models, legacy_conditions,
            )

        trackers = {}
        final_latents = {}
        kappa_data = {**kappa_sd3, **kappa_legacy}
        for name in conditions.keys():
            if name in trackers_sd3:
                trackers[name] = trackers_sd3[name]
                final_latents[name] = final_sd3[name]
            elif name in trackers_legacy:
                trackers[name] = trackers_legacy[name]
                final_latents[name] = final_legacy[name]

        vae_for_decoding = {
            **{name: models["vae"] for name in trackers_sd3.keys()},
            **{name: legacy_models["vae"] for name in trackers_legacy.keys()},
        }
    else:
        trackers, final_latents, kappa_data = collect_trajectories(cfg, models, conditions)
        vae_for_decoding = models["vae"]

    # Project
    print(f"Projecting trajectories ({cfg.projection}) ...")
    projected, explained, n_steps = project_trajectories(trackers, cfg.projection)

    # Visualize
    print("Generating visualizations ...")
    plot_trajectory_manifold(
        projected, viz_labels, n_steps,
        os.path.join(cfg.output_dir, "trajectory_manifold.png"),
        projection_method=cfg.projection,
        uniform_color=cfg.uniform_color,
        prompt_key_map=prompt_key_map,
    )
    plot_trajectory_subplots(
        projected, viz_labels, n_steps,
        os.path.join(cfg.output_dir, "trajectory_subplots.png"),
        projection_method=cfg.projection,
        uniform_color=cfg.uniform_color,
        prompt_key_map=prompt_key_map,
    )
    plot_pairwise_distances(
        trackers, labels,
        os.path.join(cfg.output_dir, "pairwise_distances.png"),
    )
    plot_decoded_images(
        final_latents, viz_labels, vae_for_decoding,
        os.path.join(cfg.output_dir, "decoded_images.png"),
        prompt_key_map=prompt_key_map,
    )

    # Quantitative summary
    print("Computing summary ...")
    summary = compute_summary(trackers, projected, explained, labels)
    summary["config"] = {
        "prompts": active_prompts,
        "num_prompts": n_prompts,
        "prompt_a": cfg.prompt_a,
        "prompt_b": cfg.prompt_b,
        "monolithic_prompt": cfg.monolithic_prompt,
        "model_id": cfg.model_id,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "seed": cfg.seed,
        "projection": cfg.projection,
        "lift": cfg.lift,
        "no_poe": cfg.no_poe,
    }
    if use_author_stoch_backend:
        summary["config"]["author_stoch_model_id"] = AUTHOR_STOCH_NOTEBOOK_MODEL_ID

    # CLIP classifier probe
    if not cfg.no_clip_probe:
        print("Running CLIP classifier probe ...")
        clip_results = clip_classifier_probe(
            final_latents, labels, vae_for_decoding,
            os.path.join(cfg.output_dir, "clip_classifier_probe.png"),
            class_prompts=clip_class_prompts,
        )
        summary["clip_classifier"] = clip_results

        print("\n  CLIP Classification Results:")
        for cond, pred in clip_results["predictions"].items():
            sims = clip_results["cosine_similarities"][cond]
            sim_str = ", ".join(f"{k}={v:.3f}" for k, v in sims.items())
            print(f"    {cond}  ->  predicted: {pred}  ({sim_str})")

    summary_path = os.path.join(cfg.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Save unified data file for plot_trajectory_analysis.py
    print("Saving trajectory_data.json ...")
    save_trajectory_data(
        trackers=trackers,
        kappa_data=kappa_data,
        projected=projected,
        explained=explained,
        summary=summary,
        labels=labels,
        conditions=conditions,
        cfg=cfg,
        output_dir=cfg.output_dir,
        clip_results=summary.get("clip_classifier"),
    )

    # --- Spatial extension ---
    if cfg.spatial and cfg.superdiff_variant == "author_stoch":
        print("\n--- Spatial Extension ---")
        print("  NOTE: spatial extension is SD3-only; skipping for author_stoch backend.")
    elif cfg.spatial:
        print("\n--- Spatial Extension ---")
        spatial_labels = {
            "spatial_a": f'"{cfg.spatial_prompt_a}"',
            "spatial_b": f'"{cfg.spatial_prompt_b}"',
            "spatial_mono": f'CLIP AND: "{cfg.spatial_monolithic}"',
            "spatial_superdiff": (
                f'SuperDIFF FM-ODE: "{cfg.spatial_prompt_a}" \u2227 "{cfg.spatial_prompt_b}"'
            ),
            "spatial_multi": (
                f'SuperDIFF Multi-AND: "{cfg.spatial_monolithic}" '
                f'\u2227 "{cfg.spatial_prompt_a}" \u2227 "{cfg.spatial_prompt_b}"'
            ),
        }
        spatial_conditions = {
            "spatial_a": {"type": "standard", "prompt": cfg.spatial_prompt_a},
            "spatial_b": {"type": "standard", "prompt": cfg.spatial_prompt_b},
            "spatial_mono": {"type": "standard", "prompt": cfg.spatial_monolithic},
            "spatial_superdiff": {
                "type": "superdiff_fm_ode",
                "obj_prompt": cfg.spatial_prompt_a,
                "bg_prompt": cfg.spatial_prompt_b,
            },
            # The key experiment: monolithic + spatial constraints via multi-AND
            "spatial_multi": {
                "type": "superdiff_multi",
                "prompts": [
                    cfg.spatial_monolithic,
                    cfg.spatial_prompt_a,
                    cfg.spatial_prompt_b,
                ],
            },
        }
        spatial_cmaps = {
            "spatial_a": "Reds",
            "spatial_b": "Blues",
            "spatial_mono": "Greens",
            "spatial_superdiff": "Oranges",
            "spatial_multi": "RdPu",
        }
        if not cfg.no_poe:
            spatial_labels["spatial_poe"] = (
                f'PoE: "{cfg.spatial_prompt_a}" \u00d7 "{cfg.spatial_prompt_b}"'
            )
            spatial_conditions["spatial_poe"] = {
                "type": "poe",
                "prompt_a": cfg.spatial_prompt_a,
                "prompt_b": cfg.spatial_prompt_b,
            }
            spatial_cmaps["spatial_poe"] = "Purples"
        # Temporarily update global cmaps
        CONDITION_CMAPS.update(spatial_cmaps)

        spatial_viz_labels, spatial_prompt_key_map = _build_compact_visual_labels(
            conditions=spatial_conditions,
            base_prompts=[cfg.spatial_prompt_a, cfg.spatial_prompt_b],
            monolithic_prompt=cfg.spatial_monolithic,
            model_tag="SD3.5",
        )

        spatial_trackers, spatial_final = collect_trajectories(
            cfg, models, spatial_conditions,
        )
        spatial_proj, spatial_expl, spatial_n = project_trajectories(
            spatial_trackers, cfg.projection,
        )
        plot_trajectory_manifold(
            spatial_proj, spatial_viz_labels, spatial_n,
            os.path.join(cfg.output_dir, "spatial_trajectory_manifold.png"),
            title="Spatial Conditioning — Latent Trajectory Dynamics",
            projection_method=cfg.projection,
            uniform_color=cfg.uniform_color,
            prompt_key_map=spatial_prompt_key_map,
        )
        plot_trajectory_subplots(
            spatial_proj, spatial_viz_labels, spatial_n,
            os.path.join(cfg.output_dir, "spatial_trajectory_subplots.png"),
            projection_method=cfg.projection,
            uniform_color=cfg.uniform_color,
            prompt_key_map=spatial_prompt_key_map,
        )
        plot_pairwise_distances(
            spatial_trackers, spatial_labels,
            os.path.join(cfg.output_dir, "spatial_pairwise_distances.png"),
        )
        plot_decoded_images(
            spatial_final, spatial_viz_labels, models["vae"],
            os.path.join(cfg.output_dir, "spatial_decoded_images.png"),
            prompt_key_map=spatial_prompt_key_map,
        )

    print(f"\nExperiment complete. Results in: {cfg.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Trajectory Dynamics Experiment",
    )
    parser.add_argument("--prompt-a", type=str, default="a dog")
    parser.add_argument("--prompt-b", type=str, default="a cat")
    parser.add_argument("--monolithic", type=str, default="",
                        help="Optional explicit monolithic prompt. If omitted, auto-joins active prompts with 'and'.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--projection", type=str, default="pca",
                        choices=["pca", "mds"])
    parser.add_argument("--lift", type=float, default=0.0)
    parser.add_argument("--spatial", action="store_true",
                        help="Run spatial extension experiment")
    parser.add_argument("--spatial-a", type=str, default="a dog on the left")
    parser.add_argument("--spatial-b", type=str, default="a cat on the right")
    parser.add_argument("--spatial-mono", type=str,
                        default="a dog on the left and a cat on the right")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--model-id", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--superdiff-variant", type=str, default="fm_ode",
                        choices=["ours", "author_det", "author_stoch", "fm_ode", "all"],
                        help="Which SuperDIFF AND algorithm to use. "
                             "'author_stoch' runs notebook-compatible SD v1.4 backend.")
    parser.add_argument("--no-poe", action="store_true",
                        help="Skip PoE (Product of Experts) condition")
    parser.add_argument("--guided", action="store_true",
                        help="Add SuperDIFF-Guided hybrid: (1-α)·v_mono + α·v_superdiff")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.3],
                        help="Blending strength(s) for --guided. Multiple values create "
                             "one condition per alpha on the same plot. "
                             "E.g.: --alpha 0.1 0.2 0.3 0.5 0.7")
    parser.add_argument("--uniform-color", action="store_true",
                        help="All trajectories use the same time colormap (viridis). "
                             "Conditions are differentiated by line style and markers.")
    parser.add_argument("--neg-prompt", type=str, default="",
                        help="Concept to negate/suppress. Activates NOT mode with Composable NOT "
                             "(Eq 13) and SuperDIFF NOT (dynamic kappa). E.g.: --neg-prompt 'glasses'")
    parser.add_argument("--neg-scale", type=float, default=1.0,
                        help="Fixed negative guidance weight for Composable NOT (default 1.0)")
    parser.add_argument("--neg-lambda", type=float, default=1.0,
                        help="Suppression strength for SuperDIFF NOT. "
                             "0=ignore neg, 1=symmetric, >1=aggressive (default 1.0)")
    parser.add_argument("--solo", action="store_true",
                        help="Solo mode: run a single prompt through standard SD3.5 only "
                             "(no CLIP AND, no SuperDIFF, no PoE). Use with --monolithic or --prompt-a.")
    parser.add_argument("--no-clip-probe", action="store_true",
                        help="Skip CLIP classifier probe")
    parser.add_argument("--num-seeds", type=int, default=1,
                        help="Run experiment with multiple seeds for robustness")
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help='Active prompt list (>=2). With 2 prompts: pairwise run. '
                             'With 3+ prompts: decode each prompt, monolithic-all, and multi-AND.')
    parser.add_argument("--gligen", action="store_true",
                        help="Use GLIGEN pipeline (DDPM noise prediction + bounding-box grounding) "
                             "instead of SD3.5. SuperDIFF variants are not available in this mode.")
    parser.add_argument("--gligen-model-id", type=str,
                        default="gligen/gligen-generation-text-box")
    parser.add_argument("--gligen-phrases", type=str, default=None,
                        help='JSON list of grounding phrases, e.g. \'["a dog", "a cat"]\'')
    parser.add_argument("--gligen-boxes", type=str, default=None,
                        help='JSON list of bounding boxes [x0,y0,x1,y1], e.g. '
                             '\'[[0.0,0.25,0.5,0.75],[0.5,0.25,1.0,0.75]]\'')
    parser.add_argument("--gligen-beta", type=float, default=0.3,
                        help="Fraction of denoising steps with grounding enabled (default 0.3)")
    args = parser.parse_args()

    seeds = [args.seed + i for i in range(args.num_seeds)]

    for seed_idx, seed in enumerate(seeds):
        if args.num_seeds > 1:
            print(f"\n{'='*60}")
            print(f"  Seed {seed_idx+1}/{args.num_seeds}  (seed={seed})")
            print(f"{'='*60}")

        out_dir = args.output_dir
        if args.num_seeds > 1 and not out_dir:
            out_dir = ""  # let auto-timestamp handle it per seed

        cfg = TrajectoryExperimentConfig(
            prompt_a=args.prompt_a,
            prompt_b=args.prompt_b,
            monolithic_prompt=args.monolithic,
            model_id=args.model_id,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=seed,
            batch_size=args.batch_size,
            projection=args.projection,
            lift=args.lift,
            superdiff_variant=args.superdiff_variant,
            uniform_color=args.uniform_color,
            neg_prompt=args.neg_prompt,
            neg_scale=args.neg_scale,
            neg_lambda=args.neg_lambda,
            solo=args.solo,
            no_poe=args.no_poe,
            guided=args.guided,
            alphas=args.alpha,
            no_clip_probe=args.no_clip_probe,
            num_seeds=args.num_seeds,
            spatial=args.spatial,
            spatial_prompt_a=args.spatial_a,
            spatial_prompt_b=args.spatial_b,
            spatial_monolithic=args.spatial_mono,
            multi_prompts=args.prompts or [],
            output_dir=out_dir,
            gligen=args.gligen,
            gligen_model_id=args.gligen_model_id,
            gligen_phrases=json.loads(args.gligen_phrases) if args.gligen_phrases else [],
            gligen_boxes=json.loads(args.gligen_boxes) if args.gligen_boxes else [],
            gligen_scheduled_sampling_beta=args.gligen_beta,
        )
        run_experiment(cfg)


if __name__ == "__main__":
    main()
