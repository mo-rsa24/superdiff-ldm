#!/usr/bin/env python3
"""
Visualization script for Riemannian Score-Based SDE samples.
Compares generated embeddings against real CLIP text embeddings to ground the manifold.

Usage:
    python notebooks/visualize_riemannian_samples.py \
        --generated-samples path/to/samples.npy \
        --category "car"
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import umap
from sklearn.decomposition import PCA
import torch

# Ensure we can import from the same directory
sys.path.append(str(Path(__file__).parent))
try:
    from manifold_diagnostics import ManifoldDiagnostics
except ImportError:
    # Fallback if running from root
    sys.path.append(str(Path(__file__).parent.parent))
    from notebooks.manifold_diagnostics import ManifoldDiagnostics

def generate_reference_prompts(category: str, n_samples: int = 50) -> list[str]:
    """Generate diverse prompts to define the 'real' manifold for a category."""
    templates = [
        f"a photo of a {category}",
        f"a {category}",
        f"a close up of a {category}",
        f"a red {category}",
        f"a blue {category}",
        f"a vintage {category}",
        f"a modern {category}",
        f"a {category} on the street",
        f"a {category} in a studio",
        f"a sketch of a {category}",
    ]
    # Repeat or sample to get n_samples
    return [templates[i % len(templates)] for i in range(n_samples)]

def plot_3d_pca(real_embeds, gen_embeds, category, output_path):
    """Generate a 3D PCA plot of the Riemannian manifold."""
    pca = PCA(n_components=3)
    
    # Fit on both to find a common basis
    all_data = np.concatenate([real_embeds, gen_embeds], axis=0)
    pca.fit(all_data)
    
    real_pca = pca.transform(real_embeds)
    gen_pca = pca.transform(gen_embeds)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Real (Reference)
    ax.scatter(real_pca[:, 0], real_pca[:, 1], real_pca[:, 2], 
               c='blue', alpha=0.6, label=f'Real CLIP ({category})', s=30)
    
    # Plot Generated (Riemannian Model)
    ax.scatter(gen_pca[:, 0], gen_pca[:, 1], gen_pca[:, 2], 
               c='red', alpha=0.6, label='Generated Samples', s=30, marker='^')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'Riemannian Manifold Structure (3D PCA)\nCategory: {category}')
    ax.legend()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D plot to {output_path}")

def plot_umap_manifold(real_embeds, gen_embeds, category, output_path):
    """Generate a 2D UMAP projection of the latent manifold."""
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    
    all_data = np.concatenate([real_embeds, gen_embeds], axis=0)
    embedding = reducer.fit_transform(all_data)
    
    n_real = len(real_embeds)
    real_umap = embedding[:n_real]
    gen_umap = embedding[n_real:]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(real_umap[:, 0], real_umap[:, 1], 
                c='blue', alpha=0.5, label=f'Real CLIP ({category})', s=50)
    plt.scatter(gen_umap[:, 0], gen_umap[:, 1], 
                c='red', alpha=0.5, label='Generated Samples', s=50, marker='^')
    
    plt.title(f'Latent Manifold Visualization (UMAP)\nGrounding Generated vs Real Embeddings')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved UMAP plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Riemannian SDE samples vs CLIP manifold")
    parser.add_argument("--generated-samples", type=str, required=True,
                        help="Path to .npy file containing generated embeddings")
    parser.add_argument("--category", type=str, default="car",
                        help="Category name to ground against (e.g., 'car', 'person')")
    parser.add_argument("--output-dir", type=str, default="outputs/riemannian_viz")
    parser.add_argument("--n-ref-samples", type=int, default=100,
                        help="Number of reference prompts to generate")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Generated Samples
    print(f"Loading generated samples from {args.generated_samples}...")
    try:
        gen_embeds = np.load(args.generated_samples)
        print(f"Loaded {len(gen_embeds)} samples with shape {gen_embeds.shape}")
    except Exception as e:
        print(f"Error loading samples: {e}")
        return

    # 2. Generate Real Reference Embeddings
    print(f"Initializing diagnostics to generate reference CLIP embeddings for '{args.category}'...")
    diag = ManifoldDiagnostics() # Uses default SD 1.5 CLIP
    
    prompts = generate_reference_prompts(args.category, args.n_ref_samples)
    print(f"Encoding {len(prompts)} reference prompts...")
    
    # Extract pooled embeddings (vectors) to match Riemannian model output
    real_embeds = diag.extract_pooled_embeddings(prompts).cpu().numpy()
    
    # 3. Visualize
    print("Generating plots...")
    plot_3d_pca(real_embeds, gen_embeds, args.category, out_dir / "riemannian_3d_pca.png")
    plot_umap_manifold(real_embeds, gen_embeds, args.category, out_dir / "latent_manifold_umap.png")
    
    print("\nDone! Check the output directory for visualizations.")

if __name__ == "__main__":
    main()
