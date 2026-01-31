#!/usr/bin/env python3
"""
Manifold Diagnostics and Visualization for Text-to-Image Models

Focused toolkit for understanding compositional prompts in latent space:
1. UMAP/t-SNE projections of text embeddings
2. Latent space traversal via true embedding interpolation
3. Whitening transform (W-CLIP) for anisotropic embeddings
4. Cross-attention map extraction from Stable Diffusion UNet
5. Dense Cosine Similarity Maps (DCSM)
6. Singular Value Decomposition (SVD) of text embeddings
7. Centroid distance plot for spatial bias detection

Usage:
    python notebooks/manifold_diagnostics.py
    python notebooks/manifold_diagnostics.py --prompts "cat" "dog" "cat on left and dog on right"
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from PIL import Image

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import Attention


# ---------------------------------------------------------------------------
# Cross-attention storage processor
# ---------------------------------------------------------------------------
class AttentionStore:
    """Collects cross-attention maps during UNet forward passes."""

    def __init__(self):
        self.maps: Dict[str, List[torch.Tensor]] = {}

    def clear(self):
        self.maps.clear()


class StoringAttnProcessor:
    """Custom attention processor that stores cross-attention weights."""

    def __init__(self, store: AttentionStore, name: str):
        self.store = store
        self.name = name

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states if is_cross else hidden_states)
        value = attn.to_v(encoder_hidden_states if is_cross else hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        attn_weights = torch.baddbmm(
            torch.empty(batch_size * attn.heads, query.shape[2], key.shape[2],
                        dtype=query.dtype, device=query.device),
            query.reshape(batch_size * attn.heads, -1, head_dim),
            key.reshape(batch_size * attn.heads, -1, head_dim).transpose(-1, -2),
            beta=0, alpha=head_dim ** -0.5,
        )
        attn_weights = attn_weights.softmax(dim=-1)

        # Store cross-attention maps only (not self-attention)
        if is_cross:
            # Average over heads: (batch, spatial, tokens)
            avg_attn = attn_weights.view(batch_size, attn.heads,
                                         query.shape[2], key.shape[2]).mean(dim=1)
            self.store.maps.setdefault(self.name, []).append(avg_attn.detach().cpu())

        hidden_states = torch.bmm(
            attn_weights,
            value.reshape(batch_size * attn.heads, -1, head_dim)
        )
        hidden_states = hidden_states.view(batch_size, attn.heads, -1, head_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Main diagnostics class
# ---------------------------------------------------------------------------
class ManifoldDiagnostics:
    """Comprehensive manifold analysis toolkit for T2I models."""

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "cuda", dtype=torch.float16):
        """Initialize models and diagnostics toolkit."""
        print("Loading Stable Diffusion pipeline...")
        self.device = torch.device(device)
        self.dtype = dtype

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # Convenience aliases
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet

        # Cross-attention storage
        self.attn_store = AttentionStore()

        # Save the original attention processors so we can restore them
        self._original_attn_procs = dict(self.unet.attn_processors)

        print("Pipeline loaded.\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Return text embeddings of shape (1, 77, hidden_dim)."""
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            return self.text_encoder(text_input.input_ids.to(self.device))[0]

    def _encode_prompt_pooled(self, prompt: str) -> torch.Tensor:
        """Return the EOS/CLS pooled embedding of shape (1, hidden_dim)."""
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self.text_encoder(text_input.input_ids.to(self.device))[0]
        # EOS token position
        eos_idx = text_input.input_ids.argmax(dim=-1)
        return hidden[torch.arange(hidden.shape[0]), eos_idx]

    def extract_pooled_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """Batch extract pooled embeddings -> (N, hidden_dim)."""
        return torch.cat([self._encode_prompt_pooled(p) for p in prompts], dim=0)

    def _generate(self, prompt: str, num_inference_steps: int = 50,
                  guidance_scale: float = 7.5,
                  seed: Optional[int] = None) -> Image.Image:
        """Generate a single image using the pipeline."""
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        return self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]

    def _generate_from_embeddings(self, prompt_embeds: torch.Tensor,
                                  negative_prompt_embeds: torch.Tensor,
                                  num_inference_steps: int = 50,
                                  guidance_scale: float = 7.5,
                                  seed: Optional[int] = None) -> Image.Image:
        """Generate an image directly from pre-computed text embeddings."""
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        return self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]

    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode a latent tensor to a PIL Image."""
        if latents.dim() == 3:
            latents = latents.unsqueeze(0)
        latents = latents.to(dtype=self.vae.dtype)
        with torch.no_grad():
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return Image.fromarray((image[0] * 255).round().astype("uint8"))

    def _install_attn_hooks(self):
        """Replace cross-attention processors with storing versions."""
        self.attn_store.clear()
        store_procs = {}
        for name, proc in self.unet.attn_processors.items():
            if "attn2" in name:
                store_procs[name] = StoringAttnProcessor(self.attn_store, name)
            else:
                store_procs[name] = proc
        self.unet.set_attn_processor(store_procs)

    def _remove_attn_hooks(self):
        """Restore original attention processors."""
        self.unet.set_attn_processor(self._original_attn_procs)

    # ------------------------------------------------------------------
    # 1. UMAP / t-SNE Projections
    # ------------------------------------------------------------------
    def visualize_embedding_space(self, prompts: List[str],
                                  output_dir: str = "outputs/manifold",
                                  method: str = "both") -> Dict[str, np.ndarray]:
        """Project text embeddings into 2D using UMAP and/or t-SNE.

        Args:
            prompts: Prompts to visualize.
            output_dir: Save directory.
            method: "umap", "tsne", or "both".

        Returns:
            Dict with projection arrays.
        """
        print(f"\n{'='*80}")
        print("1. EMBEDDING SPACE VISUALIZATION")
        print(f"{'='*80}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Extracting pooled embeddings...")
        embeddings = self.extract_pooled_embeddings(prompts).cpu().numpy()
        print(f"Embedding shape: {embeddings.shape}")

        results = {}

        ncols = 2 if method == "both" else 1
        fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7))
        if ncols == 1:
            axes = [axes]

        ax_idx = 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(prompts)))

        # --- UMAP ---
        if method in ("umap", "both"):
            print("\nComputing UMAP projection...")
            reducer = umap.UMAP(
                n_neighbors=min(15, len(prompts) - 1),
                min_dist=0.1, n_components=2, random_state=42,
            )
            proj = reducer.fit_transform(embeddings)
            results["umap"] = proj

            ax = axes[ax_idx]; ax_idx += 1
            for i, (x, y) in enumerate(proj):
                ax.scatter(x, y, c=[colors[i]], s=200, alpha=0.7,
                           edgecolors="black", linewidths=2)
                ax.annotate(prompts[i], (x, y), fontsize=9, ha="center", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors[i], alpha=0.3))

            if len(prompts) >= 3:
                self._annotate_composition_geometry(ax, proj, prompts)

            ax.set_xlabel("UMAP 1", fontsize=12)
            ax.set_ylabel("UMAP 2", fontsize=12)
            ax.set_title("UMAP Projection of Text Embeddings", fontsize=14, fontweight="bold")
            ax.grid(alpha=0.3)

        # --- t-SNE ---
        if method in ("tsne", "both"):
            print("\nComputing t-SNE projection...")
            perplexity = min(30, len(prompts) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            proj = tsne.fit_transform(embeddings)
            results["tsne"] = proj

            ax = axes[ax_idx]
            for i, (x, y) in enumerate(proj):
                ax.scatter(x, y, c=[colors[i]], s=200, alpha=0.7,
                           edgecolors="black", linewidths=2)
                ax.annotate(prompts[i], (x, y), fontsize=9, ha="center", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors[i], alpha=0.3))

            if len(prompts) >= 3:
                self._annotate_composition_geometry(ax, proj, prompts)

            ax.set_xlabel("t-SNE 1", fontsize=12)
            ax.set_ylabel("t-SNE 2", fontsize=12)
            ax.set_title("t-SNE Projection of Text Embeddings", fontsize=14, fontweight="bold")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = output_path / "embedding_projections.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved: {save_path}")
        plt.close()

        return results

    def _annotate_composition_geometry(self, ax, projection, prompts):
        """Draw dashed lines and midpoint marker between basis/compositional prompts."""
        comp_indices = [i for i, p in enumerate(prompts) if "and" in p.lower()]
        basis_indices = [i for i in range(len(prompts)) if i not in comp_indices]

        if len(basis_indices) >= 2 and comp_indices:
            for comp_idx in comp_indices:
                for basis_idx in basis_indices[:2]:
                    ax.plot(
                        [projection[basis_idx][0], projection[comp_idx][0]],
                        [projection[basis_idx][1], projection[comp_idx][1]],
                        "k--", alpha=0.3, linewidth=1,
                    )
            mid = (projection[basis_indices[0]] + projection[basis_indices[1]]) / 2
            ax.scatter(*mid, c="red", marker="x", s=200, linewidths=3,
                       label="Linear Midpoint")
            ax.legend()

    # ------------------------------------------------------------------
    # 2. Latent Space Traversal (true embedding interpolation)
    # ------------------------------------------------------------------
    def latent_space_traversal(self, prompt_a: str, prompt_b: str,
                               n_steps: int = 10,
                               output_dir: str = "outputs/manifold") -> Dict:
        """Interpolate between two prompts in CLIP embedding space and
        generate an image at each step via the diffusers pipeline.

        Args:
            prompt_a: Start prompt.
            prompt_b: End prompt.
            n_steps: Number of interpolation steps.
            output_dir: Save directory.

        Returns:
            Dict with alphas, interpolated embeddings, and generated images.
        """
        print(f"\n{'='*80}")
        print("2. LATENT SPACE TRAVERSAL")
        print(f"{'='*80}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Interpolating between:")
        print(f"  A: '{prompt_a}'")
        print(f"  B: '{prompt_b}'")
        print(f"  Steps: {n_steps}\n")

        # Encode both prompts to full-sequence embeddings
        emb_a = self._encode_prompt(prompt_a)  # (1, 77, dim)
        emb_b = self._encode_prompt(prompt_b)

        # Unconditional (negative) embedding for CFG
        uncond = self._encode_prompt("")

        alphas = np.linspace(0, 1, n_steps)
        images = []
        interpolated_embeddings = []

        print("Generating images along traversal path...")
        seed = 42

        for i, alpha in enumerate(alphas):
            print(f"  Step {i+1}/{n_steps} (alpha={alpha:.2f})...", end="\r")
            interp_emb = (1 - alpha) * emb_a + alpha * emb_b
            interpolated_embeddings.append(interp_emb.cpu())
            img = self._generate_from_embeddings(
                interp_emb, uncond, num_inference_steps=50, seed=seed
            )
            images.append(img)

        print(f"\nGenerated {len(images)} images")

        # Plot grid
        nrows = 2
        ncols = n_steps // nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = axes.flatten()

        for i, (alpha, img) in enumerate(zip(alphas, images)):
            axes[i].imshow(img)
            axes[i].set_title(f"\u03b1={alpha:.2f}", fontsize=10)
            axes[i].axis("off")

        plt.suptitle(
            f'Latent Space Traversal: "{prompt_a}" \u2192 "{prompt_b}"',
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()

        save_path = output_path / "latent_traversal.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}\n")
        plt.close()

        return {"alphas": alphas, "embeddings": interpolated_embeddings, "images": images}

    # ------------------------------------------------------------------
    # 3. Whitening Transform (W-CLIP)
    # ------------------------------------------------------------------
    def apply_whitening_transform(self, prompts: List[str],
                                  output_dir: str = "outputs/manifold") -> Tuple[np.ndarray, np.ndarray]:
        """Apply ZCA whitening to reshape the anisotropic CLIP cone into an
        isotropic Gaussian, making geometric relationships more interpretable.

        Returns:
            (original_embeddings, whitened_embeddings) as numpy arrays.
        """
        print(f"\n{'='*80}")
        print("3. WHITENING TRANSFORM (W-CLIP)")
        print(f"{'='*80}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        embeddings = self.extract_pooled_embeddings(prompts).cpu().numpy()
        print(f"Original embedding shape: {embeddings.shape}")

        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        cov = np.cov(centered.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # ZCA whitening: W = V @ diag(1/sqrt(lambda)) @ V^T
        sqrt_inv = np.diag(1.0 / np.sqrt(eigenvalues + 1e-5))
        W = eigenvectors @ sqrt_inv @ eigenvectors.T
        whitened = centered @ W

        print(f"Whitened embedding shape: {whitened.shape}")
        print(f"\nEigenvalue statistics:")
        print(f"  Min: {eigenvalues.min():.4f}")
        print(f"  Max: {eigenvalues.max():.4f}")
        print(f"  Ratio (max/min): {eigenvalues.max() / (eigenvalues.min() + 1e-10):.2f}")

        # --- Visualize ---
        fig = plt.figure(figsize=(16, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(prompts)))

        # Original (PCA)
        ax1 = plt.subplot(131)
        orig_2d = PCA(n_components=2).fit_transform(embeddings)
        for i, (x, y) in enumerate(orig_2d):
            ax1.scatter(x, y, c=[colors[i]], s=150, alpha=0.7)
            ax1.annotate(prompts[i][:20], (x, y), fontsize=8, ha="center", va="bottom")
        ax1.set_title("Original CLIP Embeddings (PCA)", fontweight="bold")
        ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.grid(alpha=0.3)

        # Whitened (PCA)
        ax2 = plt.subplot(132)
        white_2d = PCA(n_components=2).fit_transform(whitened)
        for i, (x, y) in enumerate(white_2d):
            ax2.scatter(x, y, c=[colors[i]], s=150, alpha=0.7)
            ax2.annotate(prompts[i][:20], (x, y), fontsize=8, ha="center", va="bottom")
        ax2.set_title("Whitened Embeddings (W-CLIP)", fontweight="bold")
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.grid(alpha=0.3)

        # Eigenvalue spectrum
        ax3 = plt.subplot(133)
        ax3.plot(sorted(eigenvalues, reverse=True), "o-", linewidth=2, markersize=8)
        ax3.set_xlabel("Component Index"); ax3.set_ylabel("Eigenvalue")
        ax3.set_title("Eigenvalue Spectrum", fontweight="bold")
        ax3.set_yscale("log"); ax3.grid(alpha=0.3)

        plt.suptitle("Whitening Transform Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = output_path / "whitening_transform.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved: {save_path}\n")
        plt.close()

        return embeddings, whitened

    # ------------------------------------------------------------------
    # 4. Cross-Attention Maps
    # ------------------------------------------------------------------
    def extract_cross_attention_maps(self, prompt: str,
                                     output_dir: str = "outputs/manifold",
                                     tokens_to_visualize: Optional[List[str]] = None) -> Dict:
        """Generate an image while recording cross-attention maps and
        produce per-token spatial heatmaps.

        Args:
            prompt: Text prompt to generate from.
            output_dir: Save directory.
            tokens_to_visualize: Specific tokens to plot (e.g. ["cat","dog","left","right"]).

        Returns:
            Dict with tokens, aggregated attention maps, and generated image.
        """
        print(f"\n{'='*80}")
        print("4. CROSS-ATTENTION MAP EXTRACTION")
        print(f"{'='*80}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Prompt: '{prompt}'")

        # Tokenize
        text_input = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        token_ids = text_input.input_ids[0].tolist()
        tokens = [self.tokenizer.decode([tid]).strip() for tid in token_ids]

        # Find indices of tokens we want to visualize
        token_indices = {}
        if tokens_to_visualize:
            for target in tokens_to_visualize:
                for idx, tok in enumerate(tokens):
                    if tok.lower() == target.lower():
                        token_indices[target] = idx
                        break
        print(f"\nTokens (first 12): {tokens[:12]}")
        print(f"Target token indices: {token_indices}")

        # Install storing processors and generate
        self._install_attn_hooks()

        print("\nGenerating image with attention capture...")
        img = self._generate(prompt, num_inference_steps=50, seed=42)

        # Aggregate attention maps: average across layers and timesteps
        # Each stored map has shape (batch, spatial_tokens, text_tokens)
        aggregated = {}
        for layer_name, maps_list in self.attn_store.maps.items():
            # maps_list is a list (over timesteps) of (B, S, T) tensors
            stacked = torch.stack(maps_list)          # (timesteps, B, S, T)
            aggregated[layer_name] = stacked.mean(0)  # (B, S, T)

        # Pick the highest-resolution layer for visualization
        best_layer = max(aggregated, key=lambda k: aggregated[k].shape[1])
        attn_map = aggregated[best_layer][0]  # (spatial, text_tokens)
        spatial_size = int(attn_map.shape[0] ** 0.5)

        self._remove_attn_hooks()

        print(f"Captured {len(self.attn_store.maps)} cross-attention layers")
        print(f"Best layer: {best_layer} (spatial {spatial_size}x{spatial_size})")

        # Save generated image
        img.save(output_path / "cross_attention_image.png")

        # --- Visualization ---
        viz_tokens = list(token_indices.keys()) if token_indices else tokens_to_visualize or []
        n_plots = 1 + len(viz_tokens)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        axes[0].imshow(img)
        axes[0].set_title("Generated Image", fontweight="bold")
        axes[0].axis("off")

        for ax_i, token in enumerate(viz_tokens, start=1):
            if token in token_indices:
                tidx = token_indices[token]
                heatmap = attn_map[:, tidx].reshape(spatial_size, spatial_size).numpy()
                # Upsample heatmap to image resolution
                heatmap_resized = np.array(
                    Image.fromarray(heatmap).resize(img.size, Image.BILINEAR)
                )
                axes[ax_i].imshow(img)
                axes[ax_i].imshow(heatmap_resized, alpha=0.5, cmap="jet")
                axes[ax_i].set_title(f'Attention: "{token}"', fontweight="bold")
            else:
                axes[ax_i].imshow(img)
                axes[ax_i].set_title(f'"{token}" (not found)', fontweight="bold")
            axes[ax_i].axis("off")

        plt.suptitle(f'Cross-Attention Maps: "{prompt}"', fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = output_path / "cross_attention_maps.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}\n")
        plt.close()

        return {
            "tokens": tokens,
            "token_indices": token_indices,
            "attention_maps": aggregated,
            "generated_image": img,
        }

    # ------------------------------------------------------------------
    # 5. SVD Analysis
    # ------------------------------------------------------------------
    def compute_svd_analysis(self, prompts: List[str],
                             output_dir: str = "outputs/manifold") -> Dict:
        """Decompose text embeddings via SVD to reveal semantic structure."""
        print(f"\n{'='*80}")
        print("5. SINGULAR VALUE DECOMPOSITION (SVD) ANALYSIS")
        print(f"{'='*80}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        embeddings = self.extract_pooled_embeddings(prompts).cpu().numpy()
        print(f"Embedding matrix shape: {embeddings.shape}")

        U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)

        print(f"\nSVD components:")
        print(f"  U  (left singular vectors):  {U.shape}")
        print(f"  S  (singular values):         {S.shape}")
        print(f"  Vt (right singular vectors):  {Vt.shape}")
        print(f"\n  Condition number: {S[0] / S[-1]:.2f}")

        explained_var = (S ** 2) / (S ** 2).sum()
        cumulative_var = np.cumsum(explained_var)

        # --- 6-panel figure ---
        fig = plt.figure(figsize=(16, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(prompts)))

        ax = plt.subplot(231)
        ax.plot(range(1, len(S)+1), S, "o-", linewidth=2, markersize=8)
        ax.set_xlabel("Index"); ax.set_ylabel("Singular Value")
        ax.set_title("Singular Value Spectrum", fontweight="bold")
        ax.set_yscale("log"); ax.grid(alpha=0.3)

        ax = plt.subplot(232)
        ax.bar(range(1, len(explained_var)+1), explained_var * 100, alpha=0.7)
        ax.set_xlabel("Component"); ax.set_ylabel("Explained Variance (%)")
        ax.set_title("Explained Variance Ratio", fontweight="bold"); ax.grid(alpha=0.3)

        ax = plt.subplot(233)
        ax.plot(range(1, len(cumulative_var)+1), cumulative_var * 100,
                "o-", linewidth=2, markersize=6)
        ax.axhline(y=90, color="r", linestyle="--", label="90% variance")
        ax.set_xlabel("# Components"); ax.set_ylabel("Cumulative Variance (%)")
        ax.set_title("Cumulative Explained Variance", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)

        ax = plt.subplot(234)
        proj_2d = U[:, :2]
        for i, (x, y) in enumerate(proj_2d):
            ax.scatter(x, y, c=[colors[i]], s=200, alpha=0.7,
                       edgecolors="black", linewidths=2)
            ax.annotate(prompts[i][:30], (x, y), fontsize=8, ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.3))
        ax.set_xlabel("1st Singular Vector"); ax.set_ylabel("2nd Singular Vector")
        ax.set_title("Projection onto Top 2 Singular Vectors", fontweight="bold")
        ax.grid(alpha=0.3)

        ax = plt.subplot(235)
        n_comp = min(10, Vt.shape[0])
        im = ax.imshow(Vt[:n_comp, :50], aspect="auto", cmap="RdBu_r")
        ax.set_xlabel("Embedding Dimension"); ax.set_ylabel("Singular Vector")
        ax.set_title("Right Singular Vectors (Semantic Basis)", fontweight="bold")
        plt.colorbar(im, ax=ax)

        ax = plt.subplot(236)
        errors = []
        for k in range(1, min(len(S), 20) + 1):
            recon = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            errors.append(np.linalg.norm(embeddings - recon, "fro"))
        ax.plot(range(1, len(errors)+1), errors, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("# Components"); ax.set_ylabel("Reconstruction Error (Frobenius)")
        ax.set_title("Reconstruction Error", fontweight="bold")
        ax.set_yscale("log"); ax.grid(alpha=0.3)

        plt.suptitle("Singular Value Decomposition Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = output_path / "svd_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved: {save_path}\n")
        plt.close()

        return {"U": U, "S": S, "Vt": Vt,
                "explained_variance": explained_var,
                "cumulative_variance": cumulative_var}

    # ------------------------------------------------------------------
    # 6. Centroid Distance Analysis (Spatial Bias Detection)
    # ------------------------------------------------------------------
    def centroid_distance_analysis(self, prompt: str,
                                   n_samples: int = 10,
                                   output_dir: str = "outputs/manifold") -> Dict:
        """Generate multiple images and measure the centre-of-mass offset
        to expose spatial preference bias.

        Args:
            prompt: Text prompt.
            n_samples: Number of images to generate.
            output_dir: Save directory.

        Returns:
            Dict with positions and statistics.
        """
        print(f"\n{'='*80}")
        print("6. CENTROID DISTANCE ANALYSIS (SPATIAL BIAS)")
        print(f"{'='*80}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Prompt: '{prompt}'")
        print(f"Generating {n_samples} samples...\n")

        images = []
        for i in range(n_samples):
            print(f"  Sample {i+1}/{n_samples}...", end="\r")
            img = self._generate(prompt, num_inference_steps=50, seed=42 + i)
            images.append(img)
        print(f"\nGenerated {n_samples} samples")

        # Compute centre-of-mass from pixel intensity
        positions = []
        for img in images:
            arr = np.array(img).astype(np.float32) / 255.0
            # Grayscale intensity
            intensity = arr.mean(axis=2)
            h, w = intensity.shape

            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            total = intensity.sum()
            if total > 0:
                cy = (intensity * y_coords).sum() / total
                cx = (intensity * x_coords).sum() / total
            else:
                cy, cx = h / 2, w / 2

            dist = np.sqrt((cy - h / 2) ** 2 + (cx - w / 2) ** 2)
            positions.append({"center_x": cx, "center_y": cy, "distance": dist,
                              "img_h": h, "img_w": w})

        distances = [p["distance"] for p in positions]
        img_h, img_w = positions[0]["img_h"], positions[0]["img_w"]

        # --- Visualization ---
        fig = plt.figure(figsize=(16, 10))

        # Sample images with centre-of-mass marker
        for i in range(min(6, n_samples)):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(images[i])
            pos = positions[i]
            ax.plot(pos["center_x"], pos["center_y"],
                    "r*", markersize=20, markeredgewidth=2, markeredgecolor="white")
            ax.set_title(f'Sample {i+1}\nDist: {pos["distance"]:.1f}px', fontsize=9)
            ax.axis("off")

        # Distance histogram
        ax = plt.subplot(3, 3, 7)
        ax.hist(distances, bins=max(5, n_samples // 3), alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(distances), color="r", linestyle="--", linewidth=2,
                   label=f"Mean: {np.mean(distances):.1f}")
        ax.set_xlabel("Distance from Centre (px)"); ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Distances", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)

        # 2D scatter of centres
        ax = plt.subplot(3, 3, 8)
        xs = [p["center_x"] for p in positions]
        ys = [p["center_y"] for p in positions]
        ax.scatter(xs, ys, s=100, alpha=0.6, edgecolors="black")
        ax.scatter([img_w / 2], [img_h / 2], c="red", s=200, marker="x",
                   linewidths=3, label="Image Centre")
        circle = plt.Circle((img_w / 2, img_h / 2), np.mean(distances),
                             fill=False, color="red", linestyle="--", linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(0, img_w); ax.set_ylim(0, img_h)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
        ax.set_title("Spatial Distribution of Content", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)

        # Statistics panel
        ax = plt.subplot(3, 3, 9); ax.axis("off")
        expected_dist = np.sqrt(2) * min(img_h, img_w) / 4
        stats = (
            f"SPATIAL BIAS STATISTICS\n\n"
            f"Samples:    {n_samples}\n\n"
            f"Distance from centre:\n"
            f"  Mean: {np.mean(distances):.1f} px\n"
            f"  Std:  {np.std(distances):.1f} px\n"
            f"  Min:  {np.min(distances):.1f} px\n"
            f"  Max:  {np.max(distances):.1f} px\n\n"
            f"Expected (uniform): ~{expected_dist:.1f} px\n\n"
            f"Lower mean => centre bias."
        )
        ax.text(0.1, 0.5, stats, fontsize=10, family="monospace",
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

        plt.suptitle(f'Centroid Distance Analysis: "{prompt}"',
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = output_path / "centroid_distance_analysis.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved: {save_path}\n")
        plt.close()

        return {
            "positions": positions,
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
        }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Manifold diagnostics for text-to-image models"
    )

    parser.add_argument("--prompts", type=str, nargs="+",
                        default=["a cat", "a dog",
                                 "a cat on the left", "a dog on the right",
                                 "a cat on the left and a dog on the right"])
    parser.add_argument("--prompt-a", type=str, default="a cat")
    parser.add_argument("--prompt-b", type=str, default="a dog")
    parser.add_argument("--output-dir", type=str, default="outputs/manifold_diagnostics")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--traversal-steps", type=int, default=8)
    parser.add_argument("--skip-traversal", action="store_true")
    parser.add_argument("--skip-centroid", action="store_true")

    args = parser.parse_args()

    diag = ManifoldDiagnostics()

    print("\n" + "=" * 80)
    print("MANIFOLD DIAGNOSTICS FOR TEXT-TO-IMAGE MODELS")
    print("=" * 80)
    print(f"\n  Prompts: {args.prompts}")
    print(f"  Output:  {args.output_dir}")
    print("=" * 80)

    # 1. Embedding space
    diag.visualize_embedding_space(args.prompts, output_dir=args.output_dir, method="both")

    # 2. Latent traversal
    if not args.skip_traversal:
        diag.latent_space_traversal(
            args.prompt_a, args.prompt_b,
            n_steps=args.traversal_steps, output_dir=args.output_dir,
        )

    # 3. Whitening
    diag.apply_whitening_transform(args.prompts, output_dir=args.output_dir)

    # 4. Cross-attention
    comp_prompt = args.prompts[-1] if len(args.prompts) > 2 else args.prompts[0]
    tokens_to_viz = ["cat", "dog", "left", "right"] if "and" in comp_prompt.lower() else None
    diag.extract_cross_attention_maps(
        comp_prompt, output_dir=args.output_dir, tokens_to_visualize=tokens_to_viz,
    )

    # 5. SVD
    diag.compute_svd_analysis(args.prompts, output_dir=args.output_dir)

    # 6. Centroid distance
    if not args.skip_centroid:
        diag.centroid_distance_analysis(
            comp_prompt, n_samples=args.n_samples, output_dir=args.output_dir,
        )

    # Summary
    out = Path(args.output_dir)
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {out.absolute()}")
    print(f"\n  1. {out / 'embedding_projections.png'}       - UMAP / t-SNE projections")
    if not args.skip_traversal:
        print(f"  2. {out / 'latent_traversal.png'}            - Embedding interpolation")
    print(f"  3. {out / 'whitening_transform.png'}          - W-CLIP analysis")
    print(f"  4. {out / 'cross_attention_maps.png'}         - Spatial attention heatmaps")
    print(f"  5. {out / 'svd_analysis.png'}                 - SVD semantic structure")
    if not args.skip_centroid:
        print(f"  6. {out / 'centroid_distance_analysis.png'}  - Spatial bias detection")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS TO LOOK FOR:")
    print("=" * 80)
    print("""
1. Embedding Projections:
   - Does the compositional prompt lie on a linear path between basis concepts?
   - Or does it occupy a "rare concept" region far from the mean?

2. Latent Traversal:
   - Is the transition smooth (hybridization) or discrete (composition)?
   - Look for semantic jumps vs. gradual morphing.

3. Whitening Transform:
   - High eigenvalue ratio indicates anisotropic "narrow cone".
   - Whitening should make relationships more interpretable.

4. Cross-Attention:
   - Do spatial tokens ("left", "right") attend to correct regions?
   - Or does attention diffuse across the entire frame?

5. SVD Analysis:
   - Low intrinsic dimensionality suggests manifold structure.
   - Top singular vectors reveal dominant semantic axes.

6. Centroid Distance:
   - Low mean distance reveals centre bias.
   - Affects spatial composition capability.
""")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
