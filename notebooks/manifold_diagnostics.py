#!/usr/bin/env python3
"""
Manifold Diagnostics and Visualization for Text-to-Image Models

Focused toolkit for understanding compositional prompts in latent space:
1. UMAP/t-SNE projections of text embeddings
2. Latent space traversal and interpolation
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
import sys
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from notebooks.utils import get_sd_models, get_text_embedding, get_image
from diffusion.sampling import sample_uncond
from diffusion.equations import ODE_Solver


class ManifoldDiagnostics:
    """Comprehensive manifold analysis toolkit for T2I models."""

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "cuda", dtype=torch.float16):
        """Initialize models and diagnostics toolkit."""
        print("Loading Stable Diffusion models...")
        self.device = torch.device(device)
        self.dtype = dtype

        models = get_sd_models(model_id, dtype=dtype, device=self.device)
        self.vae = models["vae"]
        self.tokenizer = models["tokenizer"]
        self.text_encoder = models["text_encoder"]
        self.unet = models["unet"]

        # Storage for cross-attention maps
        self.attention_maps = {}
        self._register_attention_hooks()

        print("Models loaded successfully!\n")

    def _register_attention_hooks(self):
        """Register hooks to capture cross-attention maps from UNet."""
        def hook_fn(name):
            def forward_hook(module, input, output):
                # Store attention weights if they exist
                if hasattr(module, 'attn_weights'):
                    self.attention_maps[name] = module.attn_weights.detach().cpu()
            return forward_hook

        # Register hooks on cross-attention layers
        for name, module in self.unet.named_modules():
            if 'attn2' in name or 'cross_attn' in name.lower():
                module.register_forward_hook(hook_fn(name))

    def get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """Get CLIP text embeddings for a list of prompts.

        Args:
            prompts: List of text prompts

        Returns:
            Tensor of shape (len(prompts), 77, 768) - full sequence embeddings
        """
        embeddings = []
        for prompt in prompts:
            emb = get_text_embedding(prompt, self.tokenizer, self.text_encoder, self.device)
            embeddings.append(emb)
        return torch.stack(embeddings)

    def extract_pooled_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """Extract pooled (sentence-level) CLIP embeddings.

        Args:
            prompts: List of text prompts

        Returns:
            Tensor of shape (len(prompts), 768) - pooled embeddings
        """
        embeddings = []
        for prompt in prompts:
            text_input = self.tokenizer(
                prompt, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.text_encoder(text_input.input_ids.to(self.device))
                # Use the [CLS] token embedding (first token)
                pooled = outputs.last_hidden_state[:, 0, :]
                embeddings.append(pooled)
        return torch.cat(embeddings, dim=0)

    def visualize_embedding_space(self, prompts: List[str],
                                  output_dir: str = "outputs/manifold",
                                  method: str = "both") -> Dict[str, np.ndarray]:
        """Create 2D projections of text embeddings using UMAP and/or t-SNE.

        Args:
            prompts: List of text prompts to visualize
            output_dir: Directory to save plots
            method: "umap", "tsne", or "both"

        Returns:
            Dictionary with projection results
        """
        print(f"\n{'='*80}")
        print("1. EMBEDDING SPACE VISUALIZATION")
        print(f"{'='*80}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get pooled embeddings for sentence-level comparison
        print("Extracting pooled embeddings...")
        embeddings = self.extract_pooled_embeddings(prompts).cpu().numpy()
        print(f"Embedding shape: {embeddings.shape}")

        results = {}

        # Create figure
        if method == "both":
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 7))
            axes = [axes]

        ax_idx = 0

        # UMAP projection
        if method in ["umap", "both"]:
            print("\nComputing UMAP projection...")
            reducer = umap.UMAP(n_neighbors=min(15, len(prompts)-1),
                              min_dist=0.1, n_components=2, random_state=42)
            umap_proj = reducer.fit_transform(embeddings)
            results['umap'] = umap_proj

            ax = axes[ax_idx]
            ax_idx += 1

            # Plot points
            colors = plt.cm.tab10(np.linspace(0, 1, len(prompts)))
            for i, (x, y) in enumerate(umap_proj):
                ax.scatter(x, y, c=[colors[i]], s=200, alpha=0.7,
                          edgecolors='black', linewidths=2)
                ax.annotate(prompts[i], (x, y), fontsize=9,
                           ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor=colors[i], alpha=0.3))

            # Check for compositional prompts
            if len(prompts) >= 3:
                self._annotate_composition_geometry(ax, umap_proj, prompts)

            ax.set_xlabel('UMAP 1', fontsize=12)
            ax.set_ylabel('UMAP 2', fontsize=12)
            ax.set_title('UMAP Projection of Text Embeddings', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)

        # t-SNE projection
        if method in ["tsne", "both"]:
            print("\nComputing t-SNE projection...")
            perplexity = min(30, len(prompts) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            tsne_proj = tsne.fit_transform(embeddings)
            results['tsne'] = tsne_proj

            ax = axes[ax_idx]

            # Plot points
            colors = plt.cm.tab10(np.linspace(0, 1, len(prompts)))
            for i, (x, y) in enumerate(tsne_proj):
                ax.scatter(x, y, c=[colors[i]], s=200, alpha=0.7,
                          edgecolors='black', linewidths=2)
                ax.annotate(prompts[i], (x, y), fontsize=9,
                           ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor=colors[i], alpha=0.3))

            # Check for compositional prompts
            if len(prompts) >= 3:
                self._annotate_composition_geometry(ax, tsne_proj, prompts)

            ax.set_xlabel('t-SNE 1', fontsize=12)
            ax.set_ylabel('t-SNE 2', fontsize=12)
            ax.set_title('t-SNE Projection of Text Embeddings', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = output_path / "embedding_projections.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}")
        plt.close()

        return results

    def _annotate_composition_geometry(self, ax, projection, prompts):
        """Annotate the geometric relationships between compositional prompts."""
        # Try to identify basis and compositional prompts
        # Heuristic: compositional prompts usually contain "and"
        comp_indices = [i for i, p in enumerate(prompts) if "and" in p.lower()]
        basis_indices = [i for i in range(len(prompts)) if i not in comp_indices]

        if len(basis_indices) >= 2 and len(comp_indices) >= 1:
            # Draw lines from basis to compositional prompts
            for comp_idx in comp_indices:
                comp_point = projection[comp_idx]

                # Draw lines to basis concepts
                for basis_idx in basis_indices[:2]:  # Limit to first 2 basis
                    basis_point = projection[basis_idx]
                    ax.plot([basis_point[0], comp_point[0]],
                           [basis_point[1], comp_point[1]],
                           'k--', alpha=0.3, linewidth=1)

                # Compute and show linear interpolation midpoint
                if len(basis_indices) >= 2:
                    mid_x = (projection[basis_indices[0]][0] + projection[basis_indices[1]][0]) / 2
                    mid_y = (projection[basis_indices[0]][1] + projection[basis_indices[1]][1]) / 2
                    ax.scatter(mid_x, mid_y, c='red', marker='x', s=200,
                             linewidths=3, label='Linear Midpoint')
                    ax.legend()

    def latent_space_traversal(self, prompt_a: str, prompt_b: str,
                               n_steps: int = 10,
                               output_dir: str = "outputs/manifold") -> Dict:
        """Traverse latent space between two prompts via linear interpolation.

        Args:
            prompt_a: First prompt
            prompt_b: Second prompt
            n_steps: Number of interpolation steps
            output_dir: Directory to save results

        Returns:
            Dictionary with interpolation results and generated images
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

        # Get embeddings
        emb_a = get_text_embedding(prompt_a, self.tokenizer, self.text_encoder, self.device)
        emb_b = get_text_embedding(prompt_b, self.tokenizer, self.text_encoder, self.device)

        # Linear interpolation
        alphas = np.linspace(0, 1, n_steps)
        interpolated_embeddings = []

        for alpha in alphas:
            interp_emb = (1 - alpha) * emb_a + alpha * emb_b
            interpolated_embeddings.append(interp_emb)

        # Generate images for each interpolation step
        print("Generating images along traversal path...")
        solver = ODE_Solver(self.unet, num_inference_steps=50, device=self.device, dtype=self.dtype)

        images = []
        for i, emb in enumerate(interpolated_embeddings):
            print(f"  Step {i+1}/{n_steps}...", end='\r')
            latent = sample_uncond(
                emb,
                torch.randn(1, 4, 64, 64, device=self.device, dtype=self.dtype),
                solver=solver,
                guidance_scale=7.5
            )
            images.append(latent)

        print(f"\n✓ Generated {len(images)} images")

        # Visualize traversal
        fig, axes = plt.subplots(2, n_steps//2, figsize=(20, 8))
        axes = axes.flatten()

        for i, (alpha, latent) in enumerate(zip(alphas, images)):
            img = get_image(self.vae, latent, 1, 1)
            axes[i].imshow(img)
            axes[i].set_title(f'α={alpha:.2f}', fontsize=10)
            axes[i].axis('off')

        plt.suptitle(f'Latent Space Traversal: "{prompt_a}" → "{prompt_b}"',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = output_path / "latent_traversal.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved: {save_path}\n")
        plt.close()

        return {
            'alphas': alphas,
            'embeddings': interpolated_embeddings,
            'images': images
        }

    def apply_whitening_transform(self, prompts: List[str],
                                  output_dir: str = "outputs/manifold") -> Tuple[np.ndarray, np.ndarray]:
        """Apply whitening transform (W-CLIP) to reshape anisotropic embeddings.

        This transforms the narrow cone of CLIP embeddings into a more isotropic
        Gaussian distribution, making the manifold easier to reason about.

        Args:
            prompts: List of text prompts
            output_dir: Directory to save visualizations

        Returns:
            Tuple of (original_embeddings, whitened_embeddings)
        """
        print(f"\n{'='*80}")
        print("3. WHITENING TRANSFORM (W-CLIP)")
        print(f"{'='*80}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get embeddings
        embeddings = self.extract_pooled_embeddings(prompts).cpu().numpy()
        print(f"Original embedding shape: {embeddings.shape}")

        # Compute mean and covariance
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        cov = np.cov(centered.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Whitening transformation: X_white = (X - μ) @ V @ Λ^(-1/2)
        # where V are eigenvectors and Λ are eigenvalues
        sqrt_inv_eigenvalues = np.diag(1.0 / np.sqrt(eigenvalues + 1e-5))
        whitening_matrix = eigenvectors @ sqrt_inv_eigenvalues @ eigenvectors.T

        whitened = centered @ whitening_matrix

        print(f"Whitened embedding shape: {whitened.shape}")
        print(f"\nEigenvalue statistics:")
        print(f"  Min: {eigenvalues.min():.4f}")
        print(f"  Max: {eigenvalues.max():.4f}")
        print(f"  Ratio (max/min): {eigenvalues.max() / eigenvalues.min():.2f}")

        # Visualize before/after
        fig = plt.figure(figsize=(16, 6))

        # Original embeddings (PCA projection)
        ax1 = plt.subplot(131)
        pca = PCA(n_components=2)
        orig_2d = pca.fit_transform(embeddings)

        colors = plt.cm.tab10(np.linspace(0, 1, len(prompts)))
        for i, (x, y) in enumerate(orig_2d):
            ax1.scatter(x, y, c=[colors[i]], s=150, alpha=0.7)
            ax1.annotate(prompts[i][:20], (x, y), fontsize=8, ha='center', va='bottom')

        ax1.set_title('Original CLIP Embeddings (PCA)', fontweight='bold')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.grid(alpha=0.3)

        # Whitened embeddings (PCA projection)
        ax2 = plt.subplot(132)
        pca_white = PCA(n_components=2)
        white_2d = pca_white.fit_transform(whitened)

        for i, (x, y) in enumerate(white_2d):
            ax2.scatter(x, y, c=[colors[i]], s=150, alpha=0.7)
            ax2.annotate(prompts[i][:20], (x, y), fontsize=8, ha='center', va='bottom')

        ax2.set_title('Whitened Embeddings (W-CLIP)', fontweight='bold')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.grid(alpha=0.3)

        # Eigenvalue spectrum
        ax3 = plt.subplot(133)
        ax3.plot(sorted(eigenvalues, reverse=True), 'o-', linewidth=2, markersize=8)
        ax3.set_xlabel('Component Index', fontsize=11)
        ax3.set_ylabel('Eigenvalue', fontsize=11)
        ax3.set_title('Eigenvalue Spectrum', fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(alpha=0.3)

        plt.suptitle('Whitening Transform Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = output_path / "whitening_transform.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}\n")
        plt.close()

        return embeddings, whitened

    def extract_cross_attention_maps(self, prompt: str,
                                     output_dir: str = "outputs/manifold",
                                     tokens_to_visualize: Optional[List[str]] = None) -> Dict:
        """Extract and visualize cross-attention maps from UNet during generation.

        Args:
            prompt: Text prompt to generate from
            output_dir: Directory to save attention maps
            tokens_to_visualize: Specific tokens to visualize (e.g., ["cat", "dog", "left", "right"])

        Returns:
            Dictionary with attention maps
        """
        print(f"\n{'='*80}")
        print("4. CROSS-ATTENTION MAP EXTRACTION")
        print(f"{'='*80}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Prompt: '{prompt}'")

        # Clear previous attention maps
        self.attention_maps = {}

        # Tokenize to get token IDs
        text_input = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        token_ids = text_input.input_ids[0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        print(f"\nTokens: {tokens[:10]}...")  # Show first 10 tokens

        # Get text embedding
        text_emb = get_text_embedding(prompt, self.tokenizer, self.text_encoder, self.device)

        # Generate image and capture attention
        print("\nGenerating image and capturing attention maps...")
        solver = ODE_Solver(self.unet, num_inference_steps=50, device=self.device, dtype=self.dtype)

        latent = sample_uncond(
            text_emb,
            torch.randn(1, 4, 64, 64, device=self.device, dtype=self.dtype),
            solver=solver,
            guidance_scale=7.5
        )

        # Decode image
        img = get_image(self.vae, latent, 1, 1)

        print(f"✓ Captured {len(self.attention_maps)} attention layers")

        # Note: Extracting actual attention weights from diffusers UNet requires
        # modifying the forward pass. This is a placeholder for the methodology.
        print("\n⚠ Note: Full attention extraction requires custom UNet forward hooks.")
        print("   This demonstrates the methodology. For production use, implement")
        print("   attention weight capture in the attention processor.")

        # Save generated image
        save_path = output_path / f"cross_attention_image.png"
        img.save(save_path)
        print(f"\n✓ Saved generated image: {save_path}")

        # Create visualization template
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Show original image in first subplot
        axes[0].imshow(img)
        axes[0].set_title('Generated Image', fontweight='bold')
        axes[0].axis('off')

        # Placeholder heatmaps for specific tokens
        if tokens_to_visualize:
            for idx, token in enumerate(tokens_to_visualize[:5], start=1):
                # In actual implementation, extract attention for this token
                # Here we show the methodology with a placeholder
                axes[idx].imshow(img)
                axes[idx].set_title(f'Attention: "{token}"', fontweight='bold')
                axes[idx].axis('off')
                axes[idx].text(0.5, 0.5, 'Placeholder\n(requires custom hooks)',
                             ha='center', va='center',
                             transform=axes[idx].transAxes,
                             fontsize=12, color='red',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle(f'Cross-Attention Maps: "{prompt}"', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = output_path / "cross_attention_maps.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved attention visualization: {save_path}\n")
        plt.close()

        return {
            'tokens': tokens,
            'attention_maps': self.attention_maps,
            'generated_image': img
        }

    def compute_svd_analysis(self, prompts: List[str],
                            output_dir: str = "outputs/manifold") -> Dict:
        """Analyze singular vectors of text embeddings to reveal semantic structure.

        Args:
            prompts: List of text prompts
            output_dir: Directory to save visualizations

        Returns:
            Dictionary with SVD results
        """
        print(f"\n{'='*80}")
        print("5. SINGULAR VALUE DECOMPOSITION (SVD) ANALYSIS")
        print(f"{'='*80}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get embeddings
        embeddings = self.extract_pooled_embeddings(prompts).cpu().numpy()
        print(f"Embedding matrix shape: {embeddings.shape}")

        # Perform SVD: X = U @ S @ V^T
        U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)

        print(f"\nSVD components:")
        print(f"  U (left singular vectors): {U.shape}")
        print(f"  S (singular values): {S.shape}")
        print(f"  V^T (right singular vectors): {Vt.shape}")

        print(f"\nSingular value statistics:")
        print(f"  Max: {S.max():.2f}")
        print(f"  Min: {S.min():.4f}")
        print(f"  Condition number: {S.max() / S.min():.2f}")

        # Compute explained variance ratio
        explained_var = (S ** 2) / (S ** 2).sum()
        cumulative_var = np.cumsum(explained_var)

        # Visualizations
        fig = plt.figure(figsize=(16, 10))

        # 1. Singular value spectrum
        ax1 = plt.subplot(231)
        ax1.plot(range(1, len(S)+1), S, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Singular Value Index', fontsize=11)
        ax1.set_ylabel('Singular Value', fontsize=11)
        ax1.set_title('Singular Value Spectrum', fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(alpha=0.3)

        # 2. Explained variance
        ax2 = plt.subplot(232)
        ax2.bar(range(1, len(explained_var)+1), explained_var * 100, alpha=0.7)
        ax2.set_xlabel('Component', fontsize=11)
        ax2.set_ylabel('Explained Variance (%)', fontsize=11)
        ax2.set_title('Explained Variance Ratio', fontweight='bold')
        ax2.grid(alpha=0.3)

        # 3. Cumulative explained variance
        ax3 = plt.subplot(233)
        ax3.plot(range(1, len(cumulative_var)+1), cumulative_var * 100,
                'o-', linewidth=2, markersize=6)
        ax3.axhline(y=90, color='r', linestyle='--', label='90% variance')
        ax3.set_xlabel('Number of Components', fontsize=11)
        ax3.set_ylabel('Cumulative Variance (%)', fontsize=11)
        ax3.set_title('Cumulative Explained Variance', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Projection onto first 2 singular vectors
        ax4 = plt.subplot(234)
        proj_2d = U[:, :2]  # Project onto first 2 left singular vectors

        colors = plt.cm.tab10(np.linspace(0, 1, len(prompts)))
        for i, (x, y) in enumerate(proj_2d):
            ax4.scatter(x, y, c=[colors[i]], s=200, alpha=0.7,
                       edgecolors='black', linewidths=2)
            ax4.annotate(prompts[i][:30], (x, y), fontsize=8,
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor=colors[i], alpha=0.3))

        ax4.set_xlabel('1st Singular Vector', fontsize=11)
        ax4.set_ylabel('2nd Singular Vector', fontsize=11)
        ax4.set_title('Projection onto Top 2 Singular Vectors', fontweight='bold')
        ax4.grid(alpha=0.3)

        # 5. Heatmap of first few right singular vectors
        ax5 = plt.subplot(235)
        n_components = min(10, Vt.shape[0])
        im = ax5.imshow(Vt[:n_components, :50], aspect='auto', cmap='RdBu_r')
        ax5.set_xlabel('Embedding Dimension', fontsize=11)
        ax5.set_ylabel('Singular Vector', fontsize=11)
        ax5.set_title('Right Singular Vectors (Semantic Basis)', fontweight='bold')
        plt.colorbar(im, ax=ax5)

        # 6. Reconstruction error vs number of components
        ax6 = plt.subplot(236)
        reconstruction_errors = []
        for k in range(1, min(len(S), 20) + 1):
            # Reconstruct with k components
            X_recon = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            error = np.linalg.norm(embeddings - X_recon, 'fro')
            reconstruction_errors.append(error)

        ax6.plot(range(1, len(reconstruction_errors)+1), reconstruction_errors,
                'o-', linewidth=2, markersize=6)
        ax6.set_xlabel('Number of Components', fontsize=11)
        ax6.set_ylabel('Reconstruction Error (Frobenius)', fontsize=11)
        ax6.set_title('Reconstruction Error vs Components', fontweight='bold')
        ax6.set_yscale('log')
        ax6.grid(alpha=0.3)

        plt.suptitle('Singular Value Decomposition Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = output_path / "svd_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}\n")
        plt.close()

        return {
            'U': U,
            'S': S,
            'Vt': Vt,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var
        }

    def centroid_distance_analysis(self, prompt: str,
                                   n_samples: int = 10,
                                   output_dir: str = "outputs/manifold") -> Dict:
        """Analyze spatial bias by measuring distance from image center.

        This diagnostic reveals whether the model has a spatial preference bias,
        where objects are better recognized in the center than at boundaries.

        Args:
            prompt: Text prompt to generate from
            n_samples: Number of samples to generate
            output_dir: Directory to save visualizations

        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*80}")
        print("6. CENTROID DISTANCE ANALYSIS (SPATIAL BIAS)")
        print(f"{'='*80}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Prompt: '{prompt}'")
        print(f"Generating {n_samples} samples...\n")

        # Get text embedding
        text_emb = get_text_embedding(prompt, self.tokenizer, self.text_encoder, self.device)

        # Generate multiple samples
        solver = ODE_Solver(self.unet, num_inference_steps=50, device=self.device, dtype=self.dtype)

        samples = []
        for i in range(n_samples):
            print(f"  Sample {i+1}/{n_samples}...", end='\r')
            latent = sample_uncond(
                text_emb,
                torch.randn(1, 4, 64, 64, device=self.device, dtype=self.dtype),
                solver=solver,
                guidance_scale=7.5
            )
            samples.append(latent)

        print(f"\n✓ Generated {len(samples)} samples")

        # For demonstration, we'll analyze the latent space distance from center
        # In practice, this would involve object detection and localization

        latent_positions = []
        for latent in samples:
            # Compute center of mass in latent space
            l = latent[0].cpu().numpy()  # (C, H, W)

            # Create coordinate grids
            h, w = l.shape[1], l.shape[2]
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

            # Compute intensity (magnitude across channels)
            intensity = np.linalg.norm(l, axis=0)

            # Compute center of mass
            total_intensity = intensity.sum()
            if total_intensity > 0:
                center_y = (intensity * y_coords).sum() / total_intensity
                center_x = (intensity * x_coords).sum() / total_intensity
            else:
                center_y, center_x = h // 2, w // 2

            # Distance from image center
            dist_from_center = np.sqrt((center_y - h/2)**2 + (center_x - w/2)**2)
            latent_positions.append({
                'center_x': center_x,
                'center_y': center_y,
                'distance': dist_from_center
            })

        # Visualize
        fig = plt.figure(figsize=(16, 10))

        # 1. Sample images
        for i in range(min(6, n_samples)):
            ax = plt.subplot(3, 3, i+1)
            img = get_image(self.vae, samples[i], 1, 1)
            ax.imshow(img)

            # Mark center of mass
            pos = latent_positions[i]
            # Scale to image coordinates (latent is 64x64, image is 512x512)
            scale = 512 / 64
            ax.plot(pos['center_x'] * scale, pos['center_y'] * scale,
                   'r*', markersize=20, markeredgewidth=2, markeredgecolor='white')

            ax.set_title(f'Sample {i+1}\nDist from center: {pos["distance"]:.2f}px',
                        fontsize=9)
            ax.axis('off')

        # 2. Distance distribution
        ax = plt.subplot(3, 3, 7)
        distances = [p['distance'] for p in latent_positions]
        ax.hist(distances, bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(distances), color='r', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(distances):.2f}')
        ax.set_xlabel('Distance from Center (latent pixels)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Distances', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. 2D scatter of centers
        ax = plt.subplot(3, 3, 8)
        x_positions = [p['center_x'] for p in latent_positions]
        y_positions = [p['center_y'] for p in latent_positions]

        ax.scatter(x_positions, y_positions, s=100, alpha=0.6, edgecolors='black')
        ax.scatter([32], [32], c='red', s=200, marker='x', linewidths=3,
                  label='Image Center')

        # Draw circle at mean distance
        circle = plt.Circle((32, 32), np.mean(distances),
                          fill=False, color='red', linestyle='--', linewidth=2)
        ax.add_patch(circle)

        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('X Position (latent space)', fontsize=11)
        ax.set_ylabel('Y Position (latent space)', fontsize=11)
        ax.set_title('Spatial Distribution of Content', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. Statistics
        ax = plt.subplot(3, 3, 9)
        ax.axis('off')

        stats_text = f"""
        SPATIAL BIAS STATISTICS

        Number of samples: {n_samples}

        Distance from center:
          Mean: {np.mean(distances):.3f} px
          Std:  {np.std(distances):.3f} px
          Min:  {np.min(distances):.3f} px
          Max:  {np.max(distances):.3f} px

        Interpretation:
          Lower mean distance indicates
          center bias in generation.

          For truly uniform spatial
          distribution, expect mean
          distance ≈ {np.sqrt(2) * 64 / 4:.2f} px
        """

        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle(f'Centroid Distance Analysis: "{prompt}"',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = output_path / "centroid_distance_analysis.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}\n")
        plt.close()

        return {
            'positions': latent_positions,
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Manifold diagnostics for text-to-image models'
    )

    # Prompts
    parser.add_argument('--prompts', type=str, nargs='+',
                       default=["a cat", "a dog",
                               "a cat on the left", "a dog on the right",
                               "a cat on the left and a dog on the right"],
                       help='List of prompts to analyze')

    parser.add_argument('--prompt-a', type=str, default="a cat",
                       help='First prompt for traversal')
    parser.add_argument('--prompt-b', type=str, default="a dog",
                       help='Second prompt for traversal')

    # Options
    parser.add_argument('--output-dir', type=str, default="outputs/manifold_diagnostics",
                       help='Output directory for results')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Number of samples for centroid analysis')
    parser.add_argument('--traversal-steps', type=int, default=8,
                       help='Number of steps for latent traversal')

    parser.add_argument('--skip-traversal', action='store_true',
                       help='Skip latent space traversal (slow)')
    parser.add_argument('--skip-centroid', action='store_true',
                       help='Skip centroid analysis (slow)')

    args = parser.parse_args()

    # Initialize diagnostics
    diagnostics = ManifoldDiagnostics()

    print("\n" + "="*80)
    print("MANIFOLD DIAGNOSTICS FOR TEXT-TO-IMAGE MODELS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Prompts: {args.prompts}")
    print(f"  Output: {args.output_dir}")
    print("="*80)

    # 1. Embedding space visualization
    diagnostics.visualize_embedding_space(
        args.prompts,
        output_dir=args.output_dir,
        method="both"
    )

    # 2. Latent space traversal
    if not args.skip_traversal:
        diagnostics.latent_space_traversal(
            args.prompt_a,
            args.prompt_b,
            n_steps=args.traversal_steps,
            output_dir=args.output_dir
        )

    # 3. Whitening transform
    diagnostics.apply_whitening_transform(
        args.prompts,
        output_dir=args.output_dir
    )

    # 4. Cross-attention maps
    comp_prompt = args.prompts[-1] if len(args.prompts) > 2 else args.prompts[0]
    tokens_to_viz = ["cat", "dog", "left", "right"] if "and" in comp_prompt.lower() else None

    diagnostics.extract_cross_attention_maps(
        comp_prompt,
        output_dir=args.output_dir,
        tokens_to_visualize=tokens_to_viz
    )

    # 5. SVD analysis
    diagnostics.compute_svd_analysis(
        args.prompts,
        output_dir=args.output_dir
    )

    # 6. Centroid distance analysis
    if not args.skip_centroid:
        diagnostics.centroid_distance_analysis(
            comp_prompt,
            n_samples=args.n_samples,
            output_dir=args.output_dir
        )

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    output_path = Path(args.output_dir)
    print(f"\nResults saved to: {output_path.absolute()}")
    print("\nGenerated visualizations:")
    print(f"  1. {output_path / 'embedding_projections.png'}")
    print(f"     → UMAP and t-SNE projections of text embeddings")

    if not args.skip_traversal:
        print(f"  2. {output_path / 'latent_traversal.png'}")
        print(f"     → Linear interpolation between prompts")

    print(f"  3. {output_path / 'whitening_transform.png'}")
    print(f"     → W-CLIP analysis for anisotropic embeddings")
    print(f"  4. {output_path / 'cross_attention_maps.png'}")
    print(f"     → Spatial attention patterns (methodology)")
    print(f"  5. {output_path / 'svd_analysis.png'}")
    print(f"     → Singular value decomposition and semantic structure")

    if not args.skip_centroid:
        print(f"  6. {output_path / 'centroid_distance_analysis.png'}")
        print(f"     → Spatial bias detection")

    print("\n" + "="*80)
    print("KEY INSIGHTS TO LOOK FOR:")
    print("="*80)
    print("""
1. Embedding Projections:
   • Do compositional prompts lie on a linear path between basis concepts?
   • Or do they occupy a "rare concept" region far from the mean?

2. Latent Traversal:
   • Is the transition smooth (hybridization) or discrete (composition)?
   • Look for semantic jumps vs. gradual morphing

3. Whitening Transform:
   • High eigenvalue ratio indicates anisotropic "narrow cone"
   • Whitening should make relationships more interpretable

4. Cross-Attention:
   • Do spatial tokens ("left", "right") attend to correct regions?
   • Or does attention diffuse across the entire frame?

5. SVD Analysis:
   • Low intrinsic dimensionality suggests manifold structure
   • Top singular vectors reveal dominant semantic axes

6. Centroid Distance:
   • Low mean distance reveals center bias
   • Affects spatial composition capability
    """)

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
