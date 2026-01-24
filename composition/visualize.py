import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

from sklearn.manifold import TSNE


def plot_latent_tsne(latents_a, latents_b, latents_superdiff, output_path="latent_tsne.png", perplexity=30):
    """
    Performs t-SNE on flattened latents. Better for visualizing non-linear cluster separation.
    """

    # 1. Flatten
    def flatten(l):
        return np.array(l).reshape(l.shape[0], -1)

    flat_a = flatten(latents_a)
    flat_b = flatten(latents_b)
    flat_sd = flatten(latents_superdiff)

    # 2. Combine
    all_data = np.concatenate([flat_a, flat_b, flat_sd], axis=0)

    # 3. Fit t-SNE
    # perplexity: related to number of nearest neighbors. 5-50 is typical.
    # n_iter: 1000 is standard, increase if convergence is bad.
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    tsne_result = tsne.fit_transform(all_data)

    # 4. Split
    n_a = len(flat_a)
    n_b = len(flat_b)

    tsne_a = tsne_result[:n_a]
    tsne_b = tsne_result[n_a:n_a + n_b]
    tsne_sd = tsne_result[n_a + n_b:]

    # 5. Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_a[:, 0], y=tsne_a[:, 1], color='blue', label='Normal', alpha=0.6, s=60)
    sns.scatterplot(x=tsne_b[:, 0], y=tsne_b[:, 1], color='orange', label='TB', alpha=0.6, s=60)
    sns.scatterplot(x=tsne_sd[:, 0], y=tsne_sd[:, 1], color='green', label='Composition', marker='X', s=100,
                    edgecolor='white')

    plt.title(f"Latent Space t-SNE (Perplexity={perplexity})")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"t-SNE plot saved to {output_path}")

def plot_log_trajectories(log_q_a_hist, log_q_b_hist, steps=None, output_path="log_trajectories.png"):
    """
    Plots the cumulative Log-Density trajectories for both models.

    Args:
        log_q_a_hist: Array of shape (T, Batch) or (T,) - Log density of Model A (Normal)
        log_q_b_hist: Array of shape (T, Batch) or (T,) - Log density of Model B (TB)
        steps: Total number of inference steps (optional, for x-axis scaling)
    """
    # Convert to numpy if JAX array
    log_q_a = np.array(log_q_a_hist).squeeze()
    log_q_b = np.array(log_q_b_hist).squeeze()

    # If batch dimension exists, average over batch for the line, or plot first seed
    if log_q_a.ndim > 1:
        log_q_a = np.mean(log_q_a, axis=1)
        log_q_b = np.mean(log_q_b, axis=1)

    t = np.arange(len(log_q_a))
    if steps:
        # Map indices to diffusion time (1.0 -> 0.0)
        time_axis = np.linspace(1.0, 0.0, len(log_q_a))
        xlabel = "Diffusion Time (t)"
        x_vals = time_axis
    else:
        xlabel = "Inference Steps"
        x_vals = t

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, log_q_a, label="Log q(Normal)", linewidth=2, color='blue')
    plt.plot(x_vals, log_q_b, label="Log q(TB)", linewidth=2, color='orange', linestyle="--")

    plt.title("SuperDiff AND: Log-Density Trajectories")
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative Log Likelihood (Relative)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Invert X axis if using Diffusion Time (1.0 -> 0.0)
    if steps:
        plt.gca().invert_xaxis()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Log plot saved to {output_path}")


def plot_kappa_trajectory(kappa_hist, steps=None, output_path="kappa_trajectory.png"):
    """
    Plots the mixing weight Kappa over time.
    """
    kappa = np.array(kappa_hist)

    # Handle batch dimension: plot mean with standard deviation shading
    if kappa.ndim > 1:
        reduce_axes = tuple(range(1, kappa.ndim))
        kappa_mean = np.mean(kappa, axis=reduce_axes)
        kappa_std = np.std(kappa, axis=reduce_axes)
    else:
        kappa_mean = kappa
        kappa_std = None

    t = np.arange(len(kappa_mean))
    if steps:
        x_vals = np.linspace(1.0, 0.0, len(kappa_mean))
        xlabel = "Diffusion Time (t)"
    else:
        x_vals = t
        xlabel = "Inference Steps"

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, kappa_mean, color='purple', linewidth=2, label=r"Average $\kappa$")

    if kappa_std is not None:
        plt.fill_between(x_vals, kappa_mean - kappa_std, kappa_mean + kappa_std, color='purple', alpha=0.2,
                         label="Batch Std Dev")

    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label="Neutral (0.5)")
    plt.axhline(0.0, color='red', linestyle='--', alpha=0.3)
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.3)

    plt.title(r"Mixing Trajectory ($\kappa$)")
    plt.xlabel(xlabel)
    plt.ylabel(r"$\kappa$ Value")
    plt.ylim(-0.5, 2.5)  # Based on your clip value
    plt.legend()
    plt.grid(True, alpha=0.3)

    if steps:
        plt.gca().invert_xaxis()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Kappa plot saved to {output_path}")


def plot_latent_pca(latents_a, latents_b, latents_superdiff, output_path="latent_pca.png"):
    """
    Performs PCA on the flattened latents to visualize distribution intersection.

    Args:
        latents_*: Numpy arrays of shape (N, H, W, C)
    """

    # 1. Flatten Latents: (N, H, W, C) -> (N, Features)
    def flatten(l):
        return np.array(l).reshape(l.shape[0], -1)

    flat_a = flatten(latents_a)
    flat_b = flatten(latents_b)
    flat_sd = flatten(latents_superdiff)

    # 2. Combine for PCA fitting
    # We fit on all data to find the common subspace
    all_data = np.concatenate([flat_a, flat_b, flat_sd], axis=0)

    # 3. Fit PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_data)

    # 4. Split back
    n_a = len(flat_a)
    n_b = len(flat_b)

    pca_a = pca_result[:n_a]
    pca_b = pca_result[n_a:n_a + n_b]
    pca_sd = pca_result[n_a + n_b:]

    # 5. Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_a[:, 0], y=pca_a[:, 1], color='blue', label='Model A (Normal)', alpha=0.6, s=60)
    sns.scatterplot(x=pca_b[:, 0], y=pca_b[:, 1], color='orange', label='Model B (TB)', alpha=0.6, s=60)
    sns.scatterplot(x=pca_sd[:, 0], y=pca_sd[:, 1], color='green', label='SuperDiff (Composition)', marker='X', s=100,
                    edgecolor='white')

    plt.title("Latent Space PCA Analysis")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"PCA plot saved to {output_path}")