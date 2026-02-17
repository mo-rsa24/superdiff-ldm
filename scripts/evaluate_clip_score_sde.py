"""
Evaluate a trained Riemannian Score SDE on CLIP embeddings.

Produces:
  1. Log-likelihood table: text embeddings vs real image embeddings
  2. PCA/UMAP projection plots with density contours
  3. Semantic dilution metric per pair

Usage (after training clip_s767):
    python scripts/evaluate_clip_score_sde.py \
        --ckpt results/clip_s767/.../ckpt \
        --pair "person+car"

    # Or evaluate all pairs at once:
    python scripts/evaluate_clip_score_sde.py \
        --ckpt results/clip_s767/.../ckpt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# JAX / Haiku
import jax
import jax.numpy as jnp
import haiku as hk
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIB_ROOT = PROJECT_ROOT / "libraries" / "riemannian-score-sde"
CLIP_DIR = PROJECT_ROOT / "clip_embeddings" / "coco_common_pairs"

sys.path.insert(0, str(LIB_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["GEOMSTATS_BACKEND"] = "jax"

from score_sde.utils import restore
from score_sde.models.flow import SDEPushForward


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to checkpoint directory from training")
    p.add_argument("--config_dir", type=Path, default=None,
                   help="Path to .hydra config dir (if not adjacent to ckpt)")
    p.add_argument("--pair", type=str, default=None,
                   help="Specific pair to evaluate (e.g. 'person+car'). "
                        "If None, evaluates all pairs.")
    p.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "results" / "clip_eval")
    p.add_argument("--n_samples", type=int, default=512,
                   help="Number of samples to draw from the model for plots")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Load CLIP embeddings ─────────────────────────────────────────────

def load_clip_data(clip_dir):
    """Load all pre-extracted CLIP embeddings and metadata."""
    image_embeds = np.load(clip_dir / "image_embeddings.npy")
    with open(clip_dir / "image_manifest.jsonl") as f:
        manifest = [json.loads(line) for line in f if line.strip()]

    text_dir = clip_dir / "text_embeddings"
    comp = np.load(text_dir / "compositional.npy")
    ind_a = np.load(text_dir / "individual_a.npy")
    ind_b = np.load(text_dir / "individual_b.npy")
    with open(text_dir / "prompts.json") as f:
        prompts = json.load(f)

    return {
        "image_embeds": image_embeds,
        "manifest": manifest,
        "compositional": comp,
        "individual_a": ind_a,
        "individual_b": ind_b,
        "prompts": prompts,
        "pair_keys": list(prompts.keys()),
    }


# ── Rebuild model from checkpoint ────────────────────────────────────

def load_model_and_pushforward(ckpt_path, cfg):
    """Reconstruct the model stack and load trained weights."""
    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    beta_schedule = instantiate(cfg.beta_schedule)
    flow = instantiate(cfg.flow, manifold=model_manifold, beta_schedule=beta_schedule)
    base = instantiate(cfg.base, model_manifold, flow)
    pushforward = instantiate(cfg.pushf, flow, base, transform=transform)

    def model_fn(y, t, context=None):
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator, cfg.architecture, cfg.embedding,
            output_shape, manifold=model_manifold,
        )
        if context is not None:
            t_expanded = jnp.expand_dims(t.reshape(-1), -1)
            if context.shape[0] != y.shape[0]:
                context = jnp.repeat(jnp.expand_dims(context, 0), y.shape[0], 0)
            context = jnp.concatenate([t_expanded, context], axis=-1)
        else:
            context = t
        return score(y, context)

    model = hk.transform_with_state(model_fn)
    train_state = restore(str(ckpt_path))

    return model, train_state, pushforward, data_manifold, transform


# ── Log-likelihood evaluation ────────────────────────────────────────

def compute_log_likelihoods(model, train_state, pushforward, points, batch_size=256):
    """Compute log p(x) for a set of points using the trained model."""
    model_w_dicts = (model, train_state.params_ema, train_state.model_state)
    likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False)
    likelihood_fn = jax.jit(likelihood_fn)

    all_logp = []
    n = points.shape[0]
    for start in range(0, n, batch_size):
        batch = jnp.array(points[start:start + batch_size])
        logp, nfe = likelihood_fn(batch)
        all_logp.append(np.array(logp))

    return np.concatenate(all_logp)


# ── Sampling from the model ──────────────────────────────────────────

def sample_from_model(model, train_state, pushforward, n_samples, rng, cfg):
    """Draw samples from the learned distribution."""
    model_w_dicts = (model, train_state.params_ema, train_state.model_state)
    sampler_kwargs = dict(N=100, eps=cfg.eps, predictor="GRW")
    sampler = pushforward.get_sampler(model_w_dicts, train=False, **sampler_kwargs)
    samples = sampler(rng, (n_samples,), None)
    return np.array(samples)


# ── PCA projection plot ──────────────────────────────────────────────

def plot_pca_projection(image_embeds, text_comp, text_a, text_b,
                        samples, pair_key, prompts, output_path):
    """2D PCA projection of image embeddings, text embeddings, and model samples."""
    # Fit PCA on all available points so it works even with very few images
    all_pts = [image_embeds, text_comp.reshape(1, -1), text_a.reshape(1, -1),
               text_b.reshape(1, -1)]
    if samples is not None and len(samples) > 0:
        all_pts.append(samples)
    fit_data = np.concatenate(all_pts, axis=0)

    n_components = min(2, fit_data.shape[0], fit_data.shape[1])
    if n_components < 2:
        print(f"  Skipping PCA plot for {pair_key}: not enough data for 2D projection")
        return

    pca = PCA(n_components=2)
    pca.fit(fit_data)
    img_2d = pca.transform(image_embeds)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Real data + text embeddings
    ax = axes[0]
    ax.scatter(img_2d[:, 0], img_2d[:, 1], c="steelblue", alpha=0.3, s=10,
               label=f"Real images (N={len(image_embeds)})")

    # Project text embeddings
    comp_2d = pca.transform(text_comp.reshape(1, -1))
    a_2d = pca.transform(text_a.reshape(1, -1))
    b_2d = pca.transform(text_b.reshape(1, -1))

    p = prompts[pair_key]
    ax.scatter(*comp_2d[0], c="red", s=200, marker="*", zorder=5,
               label=f'"{p["compositional"]}"')
    ax.scatter(*a_2d[0], c="orange", s=150, marker="^", zorder=5,
               label=f'"{p["individual_a"]}"')
    ax.scatter(*b_2d[0], c="green", s=150, marker="v", zorder=5,
               label=f'"{p["individual_b"]}"')

    # Draw lines: individual concepts → compositional
    ax.plot([a_2d[0, 0], comp_2d[0, 0]], [a_2d[0, 1], comp_2d[0, 1]],
            "r--", alpha=0.5)
    ax.plot([b_2d[0, 0], comp_2d[0, 0]], [b_2d[0, 1], comp_2d[0, 1]],
            "r--", alpha=0.5)

    ax.set_title(f"Real Image Embeddings + CLIP Text\n{pair_key}")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    # Right: Model samples + text embeddings
    ax = axes[1]
    if samples is not None and len(samples) > 0:
        samp_2d = pca.transform(samples)
        ax.scatter(samp_2d[:, 0], samp_2d[:, 1], c="purple", alpha=0.2, s=10,
                   label=f"Model samples (N={len(samples)})")
    ax.scatter(img_2d[:, 0], img_2d[:, 1], c="steelblue", alpha=0.15, s=10,
               label="Real images")
    ax.scatter(*comp_2d[0], c="red", s=200, marker="*", zorder=5,
               label="Compositional text")
    ax.scatter(*a_2d[0], c="orange", s=150, marker="^", zorder=5,
               label="Individual A text")
    ax.scatter(*b_2d[0], c="green", s=150, marker="v", zorder=5,
               label="Individual B text")

    ax.set_title(f"Model Samples vs Real\n{pair_key}")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved PCA plot: {output_path}")


# ── Semantic dilution report ─────────────────────────────────────────

def print_dilution_report(pair_key, prompts, logp_images, logp_comp, logp_a, logp_b):
    """Print the semantic dilution metric for one pair."""
    mean_img_logp = logp_images.mean()
    std_img_logp = logp_images.std()

    print(f"\n{'=' * 70}")
    print(f"  SEMANTIC DILUTION REPORT: {pair_key}")
    print(f"{'=' * 70}")

    p = prompts[pair_key]
    print(f"\n  Real image embeddings (N={len(logp_images)}):")
    print(f"    mean log p = {mean_img_logp:.2f}  (std={std_img_logp:.2f})")
    print(f"    range: [{logp_images.min():.2f}, {logp_images.max():.2f}]")

    print(f"\n  Text embedding log-likelihoods:")
    print(f"    '{p['compositional']}':  log p = {logp_comp:.2f}")
    print(f"    '{p['individual_a']}':       log p = {logp_a:.2f}")
    print(f"    '{p['individual_b']}':         log p = {logp_b:.2f}")

    dilution = mean_img_logp - logp_comp
    print(f"\n  Semantic Dilution = mean_img_logp - comp_text_logp")
    print(f"                    = {mean_img_logp:.2f} - ({logp_comp:.2f})")
    print(f"                    = {dilution:.2f}")

    z_score = (logp_comp - mean_img_logp) / max(std_img_logp, 1e-8)
    print(f"  Z-score of compositional text: {z_score:.2f}")

    if dilution > 0:
        print(f"\n  >> CLIP's AND embedding is LESS likely than real images")
        print(f"     under the learned data density (dilution = {dilution:.2f})")
    else:
        print(f"\n  >> CLIP's AND embedding is within the real data density")

    print(f"{'=' * 70}")

    return {
        "pair": pair_key,
        "mean_img_logp": float(mean_img_logp),
        "std_img_logp": float(std_img_logp),
        "comp_text_logp": float(logp_comp),
        "individual_a_logp": float(logp_a),
        "individual_b_logp": float(logp_b),
        "dilution": float(dilution),
        "z_score": float(z_score),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load CLIP data
    print("Loading CLIP embeddings...")
    clip_data = load_clip_data(CLIP_DIR)

    # Load Hydra config from the training run
    config_path = None
    if args.config_dir is not None:
        config_path = args.config_dir
    else:
        # Search common locations relative to ckpt
        candidates = [
            args.ckpt.parent / ".hydra",       # ckpt is inside run dir
            args.ckpt.parent.parent / ".hydra", # ckpt/name/ inside run dir
            args.ckpt / ".hydra",               # ckpt IS the run dir
        ]
        # Also check results/{ckpt_name}/  (Hydra output dir)
        ckpt_name = args.ckpt.name
        for results_sub in sorted((PROJECT_ROOT / "results").glob(f"{ckpt_name}/*/.hydra")):
            candidates.append(results_sub)
        for c in candidates:
            if c.exists():
                config_path = c
                break

    if config_path is None or not config_path.exists():
        print(f"ERROR: Could not find .hydra config. Searched near {args.ckpt}")
        print("  Use --config_dir to specify the .hydra directory explicitly.")
        return

    config_path = config_path.resolve()
    print(f"Loading config from {config_path}")
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="config")

    # Load model
    print("Loading trained model...")
    model, train_state, pushforward, data_manifold, transform = \
        load_model_and_pushforward(args.ckpt, cfg)

    rng = jax.random.PRNGKey(args.seed)

    # Determine which pairs to evaluate
    pair_keys = [args.pair] if args.pair else clip_data["pair_keys"]

    all_results = []

    for pi, pair_key in enumerate(pair_keys):
        if pair_key not in clip_data["prompts"]:
            print(f"Skipping unknown pair: {pair_key}")
            continue

        idx = clip_data["pair_keys"].index(pair_key)
        print(f"\n[{pi+1}/{len(pair_keys)}] Evaluating: {pair_key}")

        # Get image embeddings for this pair
        pair_mask = np.array([m["pair"] == pair_key for m in clip_data["manifest"]])
        pair_images = clip_data["image_embeds"][pair_mask]

        # Get text embeddings
        text_comp = clip_data["compositional"][idx]
        text_a = clip_data["individual_a"][idx]
        text_b = clip_data["individual_b"][idx]

        # Stack all points for likelihood evaluation
        all_points = np.concatenate([
            pair_images,
            text_comp.reshape(1, -1),
            text_a.reshape(1, -1),
            text_b.reshape(1, -1),
        ], axis=0)

        # Compute log-likelihoods
        print("  Computing log-likelihoods...")
        logp = compute_log_likelihoods(
            model, train_state, pushforward, all_points, args.batch_size
        )

        logp_images = logp[:len(pair_images)]
        logp_comp = logp[len(pair_images)]
        logp_a = logp[len(pair_images) + 1]
        logp_b = logp[len(pair_images) + 2]

        # Print report
        result = print_dilution_report(
            pair_key, clip_data["prompts"],
            logp_images, logp_comp, logp_a, logp_b
        )
        all_results.append(result)

        # Generate samples and plot
        print("  Sampling from model...")
        rng, sample_rng = jax.random.split(rng)
        try:
            samples = sample_from_model(
                model, train_state, pushforward,
                args.n_samples, sample_rng, cfg
            )
        except Exception as e:
            print(f"  Sampling failed: {e}")
            samples = None

        plot_path = args.output_dir / f"pca_{pair_key.replace('+', '_')}.png"
        plot_pca_projection(
            pair_images, text_comp, text_a, text_b,
            samples, pair_key, clip_data["prompts"], plot_path
        )

    # Save summary
    summary_path = args.output_dir / "dilution_report.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Print summary table
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("  SUMMARY: SEMANTIC DILUTION ACROSS ALL PAIRS")
        print(f"{'=' * 70}")
        print(f"  {'Pair':<28s}  {'img logp':>10s}  {'AND logp':>10s}  {'dilution':>10s}  {'z-score':>8s}")
        print(f"  {'-' * 70}")
        for r in sorted(all_results, key=lambda x: -x["dilution"]):
            print(f"  {r['pair']:<28s}  {r['mean_img_logp']:>+10.2f}  "
                  f"{r['comp_text_logp']:>+10.2f}  {r['dilution']:>+10.2f}  "
                  f"{r['z_score']:>+8.2f}")
        mean_dilution = np.mean([r["dilution"] for r in all_results])
        print(f"  {'-' * 70}")
        print(f"  {'MEAN':<28s}  {'':>10s}  {'':>10s}  {mean_dilution:>+10.2f}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
