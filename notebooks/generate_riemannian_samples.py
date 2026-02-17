"""
Generate samples from a trained Riemannian Score SDE model and save as .npy.

Usage:
    python scripts/generate_riemannian_samples.py \
        --ckpt outputs/2023-10-27/10-00-00/ckpt/checkpoint \
        --n_samples 1000 \
        --output_path samples.npy
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import jax
import haiku as hk
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate, get_class

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIB_ROOT = PROJECT_ROOT / "libraries" / "riemannian-score-sde"
sys.path.insert(0, str(LIB_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Set backend
os.environ["GEOMSTATS_BACKEND"] = "jax"

from score_sde.utils import restore

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint directory")
    p.add_argument("--config_dir", type=Path, default=None,
                   help="Path to .hydra config dir (if not adjacent to ckpt)")
    p.add_argument("--output_path", type=Path, default="samples.npy", help="Where to save the .npy file")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def load_model(ckpt_path, config_dir=None):
    # Locate the .hydra config directory
    if config_dir is not None:
        config_path = config_dir
    else:
        run_dir = ckpt_path.parent.parent
        config_path = run_dir / ".hydra"
        if not config_path.exists():
            if (ckpt_path.parent / ".hydra").exists():
                config_path = ckpt_path.parent / ".hydra"
            else:
                # Search in results/ directory matching ckpt name
                project_root = Path(__file__).resolve().parent.parent
                ckpt_name = ckpt_path.name
                matches = sorted((project_root / "results").glob(f"{ckpt_name}/*/.hydra"))
                if matches:
                    config_path = matches[0]
                else:
                    raise FileNotFoundError(
                        f"Could not find .hydra config. Searched near {ckpt_path}\n"
                        "  Use --config_dir to specify the .hydra directory explicitly."
                    )

    print(f"Loading config from {config_path}")
    with initialize_config_dir(config_dir=str(config_path.absolute()), version_base=None):
        cfg = compose(config_name="config")

    # Instantiate components
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
            t_expanded = jax.numpy.expand_dims(t.reshape(-1), -1)
            if context.shape[0] != y.shape[0]:
                 context = jax.numpy.repeat(jax.numpy.expand_dims(context, 0), y.shape[0], 0)
            context = jax.numpy.concatenate([t_expanded, context], axis=-1)
        else:
            context = t
        return score(y, context)

    model = hk.transform_with_state(model_fn)
    
    print(f"Restoring from {ckpt_path}")
    train_state = restore(str(ckpt_path))
    
    return model, train_state, pushforward, cfg

def main():
    args = parse_args()
    
    print("Loading model...")
    model, train_state, pushforward, cfg = load_model(args.ckpt, args.config_dir)
    
    rng = jax.random.PRNGKey(args.seed)
    
    all_samples = []
    num_batches = int(np.ceil(args.n_samples / args.batch_size))
    
    print(f"Generating {args.n_samples} samples in {num_batches} batches...")
    
    # Prepare sampler
    # Use EMA params if available
    params = train_state.params_ema if hasattr(train_state, 'params_ema') else train_state.params
    model_w_dicts = (model, params, train_state.model_state)
    
    # Use default sampler settings (N=100, GRW predictor)
    eps = cfg.eps if hasattr(cfg, 'eps') else 1e-3
    sampler = pushforward.get_sampler(model_w_dicts, train=False, N=100, eps=eps, predictor="GRW")
    
    for i in range(num_batches):
        print(f"Batch {i+1}/{num_batches}", end="\r")
        current_batch_size = min(args.batch_size, args.n_samples - len(all_samples))
        rng, step_rng = jax.random.split(rng)
        
        # Pass None for context (unconditional sampling or default behavior)
        batch_samples = sampler(step_rng, (current_batch_size,), None)
        all_samples.append(np.array(batch_samples))
        
    all_samples = np.concatenate(all_samples, axis=0)
    print(f"\nGenerated {all_samples.shape} samples.")
    
    # Ensure output directory exists
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, all_samples)
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()