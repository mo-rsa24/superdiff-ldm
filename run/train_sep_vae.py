"""
Training script for Multi-head SepVAE with frozen CheSS backbone.

This script implements the full training loop for SepVAE with:
- Dual optimizer setup (VAE + MI discriminator)
- CheSS weight loading and injection
- VinBigData triplet dataset
- W&B logging
- Checkpoint saving/loading

Usage:
    python -m run.train_sep_vae --wandb --epochs 100 --batch_size 8
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes
import torch
from torch.utils.data import DataLoader

# Local imports
from datasets.VinBigData import VinBigDataTripletDataset, jax_collate_fn
from models.sep_vae_jax import SepVAE
from losses.sep_vae_losses import MIDiscriminator, SepVAELossConfig, sepvae_loss
from utils.weight_converter import convert_chess_resnet50, load_converted_weights
from utils.sepvae_analysis import visualize_backbone_features

# Optional W&B
try:
    import wandb
    _WANDB = True
except ImportError:
    wandb = None
    _WANDB = False


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser("Multi-head SepVAE with CheSS backbone trainer")

    # Data
    p.add_argument("--dicom_dir", type=str, default="/datasets/mmolefe/vinbigdata/train",
                   help="Path to DICOM files")
    p.add_argument("--csv_path", type=str, default="/datasets/mmolefe/vinbigdata/train.csv",
                   help="Path to train.csv")
    p.add_argument("--img_size", type=int, default=512,
                   help="Image size (default: 512)")

    # Model architecture (spatial latents: 64×64×channels)
    p.add_argument("--z_channels_common", type=int, default=4,
                   help="Common latent channels (default: 4 for 64×64×4)")
    p.add_argument("--z_channels_disease", type=int, default=2,
                   help="Disease-specific latent channels (default: 2 for 64×64×2 each)")
    p.add_argument("--frozen_backbone", action="store_true", default=True,
                   help="Freeze CheSS backbone")

    # CheSS weights
    p.add_argument("--chess_checkpoint", type=str,
                   default="/datasets/mmolefe/chess/pretrained_weights.pth.tar",
                   help="Path to CheSS PyTorch checkpoint")
    p.add_argument("--chess_converted", type=str, default=None,
                   help="Path to converted JAX weights (.npy). If None, will convert on-the-fly")

    # Loss weights
    p.add_argument("--weight_rec", type=float, default=1.0,
                   help="Reconstruction loss weight")
    p.add_argument("--weight_kl_common", type=float, default=1e-4,
                   help="Common KL weight")
    p.add_argument("--weight_kl_disease", type=float, default=1e-4,
                   help="Disease KL weight")
    p.add_argument("--weight_null", type=float, default=1e-3,
                   help="Nulling loss weight")
    p.add_argument("--weight_mi", type=float, default=1e-3,
                   help="MI penalty weight")
    p.add_argument("--sigma_inactive", type=float, default=0.1,
                   help="Tight prior std dev for inactive heads")

    # Optimizer
    p.add_argument("--lr_vae", type=float, default=1e-4,
                   help="VAE learning rate")
    p.add_argument("--lr_disc", type=float, default=1e-4,
                   help="MI discriminator learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="Weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping norm")

    # Training
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size (triplets, results in 3*batch_size images)")
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of epochs")
    p.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader workers")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed")

    # Logging & checkpoints
    p.add_argument("--output_root", type=str, default="runs_sepvae",
                   help="Output directory root")
    p.add_argument("--exp_name", type=str, default="sepvae_chess",
                   help="Experiment name")
    p.add_argument("--log_every", type=int, default=100,
                   help="Log every N steps")
    p.add_argument("--save_every", type=int, default=1,
                   help="Save checkpoint every N epochs")

    # Verbose diagnostics
    p.add_argument("--verbose_backbone", action="store_true",
                   help="Visualize frozen backbone features before training (PCA, class diffs, top-K channels)")
    p.add_argument("--verbose_n_samples", type=int, default=12,
                   help="Number of samples for backbone visualization (default: 12)")

    # W&B
    p.add_argument("--wandb", action="store_true",
                   help="Enable W&B logging")
    p.add_argument("--wandb_project", type=str, default="sepvae-chess",
                   help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="W&B entity (username/team)")

    return p.parse_args()


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def main():
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    print("="*60)
    print("MULTI-HEAD SEPVAE WITH CHESS BACKBONE")
    print("="*60)

    # ===== Setup output directories =====
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_slug = f"{args.exp_name}-{timestamp}"
    output_dir = Path(args.output_root) / exp_slug
    ckpt_dir = ensure_dir(output_dir / "checkpoints")

    print(f"\nOutput directory: {output_dir}")

    # ===== Initialize W&B =====
    if args.wandb and _WANDB:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=exp_slug,
            config=vars(args)
        )
        print("✓ W&B initialized")
    elif args.wandb and not _WANDB:
        print("⚠ W&B requested but not installed, skipping")

    # ===== Load dataset =====
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)

    dataset = VinBigDataTripletDataset(
        dicom_dir=args.dicom_dir,
        csv_path=args.csv_path,
        img_size=args.img_size
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=jax_collate_fn,
        drop_last=True
    )

    print(f"✓ Dataset loaded: {len(dataset)} triplets")
    print(f"✓ Batch size: {args.batch_size} triplets → {args.batch_size * 3} images")
    print(f"✓ Steps per epoch: {len(loader)}")

    # ===== Load CheSS weights =====
    print("\n" + "="*60)
    print("LOADING CHESS WEIGHTS")
    print("="*60)

    if args.chess_converted and os.path.exists(args.chess_converted):
        print(f"Loading pre-converted weights from: {args.chess_converted}")
        chess_params, chess_batch_stats = load_converted_weights(args.chess_converted)
    else:
        print(f"Converting CheSS weights from: {args.chess_checkpoint}")
        chess_params, chess_batch_stats = convert_chess_resnet50(args.chess_checkpoint, verbose=True)

        # Save converted weights for future use
        from utils.weight_converter import save_converted_weights
        converted_path = output_dir / "chess_jax_params.npy"
        save_converted_weights((chess_params, chess_batch_stats), str(converted_path))
        print(f"✓ Saved converted weights to: {converted_path}")

    # ===== Initialize models =====
    print("\n" + "="*60)
    print("INITIALIZING MODELS")
    print("="*60)

    # SepVAE model
    sepvae = SepVAE(
        z_channels_common=args.z_channels_common,
        z_channels_disease=args.z_channels_disease,
        frozen_backbone=args.frozen_backbone
    )

    # MI discriminator
    mi_disc = MIDiscriminator(hidden_dim=512)

    # Dummy inputs for initialization
    dummy_x = jnp.ones((1, args.img_size, args.img_size, 1))
    dummy_labels = jnp.array([0])
    dummy_z_c = jnp.ones((1, 64, 64, args.z_channels_common))  # Spatial
    dummy_z_d = jnp.ones((1, 64, 64, args.z_channels_disease))  # Spatial

    # Initialize SepVAE
    rng, init_rng = jax.random.split(rng)
    vae_vars = sepvae.init(
        {'params': init_rng, 'dropout': init_rng},
        dummy_x, dummy_labels, key=init_rng, train=True
    )
    vae_params = vae_vars['params']
    vae_batch_stats = vae_vars.get('batch_stats', {})

    # Inject CheSS weights into backbone (both params and batch_stats)
    print("\nInjecting CheSS weights into backbone...")
    vae_params = vae_params.copy()
    vae_params['encoder']['backbone'] = chess_params

    # Inject BatchNorm running statistics from CheSS
    if vae_batch_stats:
        vae_batch_stats = vae_batch_stats.copy()
        vae_batch_stats['encoder']['backbone'] = chess_batch_stats
    else:
        vae_batch_stats = {'encoder': {'backbone': chess_batch_stats}}

    # Ensure all are JAX arrays
    vae_params = jax.tree_util.tree_map(jnp.array, vae_params)
    vae_batch_stats = jax.tree_util.tree_map(jnp.array, vae_batch_stats)

    vae_param_count = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
    print(f"✓ SepVAE parameters: {vae_param_count:,}")
    bs_count = sum(p.size for p in jax.tree_util.tree_leaves(vae_batch_stats))
    print(f"✓ SepVAE batch_stats: {bs_count:,} (frozen BN running stats)")

    # Initialize MI discriminator
    rng, init_rng = jax.random.split(rng)
    mi_vars = mi_disc.init(init_rng, dummy_z_c, dummy_z_d, train=True)
    mi_params = mi_vars['params']

    mi_param_count = sum(p.size for p in jax.tree_util.tree_leaves(mi_params))
    print(f"✓ MI discriminator parameters: {mi_param_count:,}")

    # ===== Create optimizers =====
    print("\n" + "="*60)
    print("CREATING OPTIMIZERS")
    print("="*60)

    # VAE optimizer
    tx_vae = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=args.lr_vae, weight_decay=args.weight_decay)
    )

    # MI discriminator optimizer
    tx_disc = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=args.lr_disc, weight_decay=args.weight_decay)
    )

    vae_state = TrainState.create(apply_fn=None, params=vae_params, tx=tx_vae)
    disc_state = TrainState.create(apply_fn=None, params=mi_params, tx=tx_disc)

    print(f"✓ VAE optimizer: AdamW (lr={args.lr_vae}, wd={args.weight_decay})")
    print(f"✓ Discriminator optimizer: AdamW (lr={args.lr_disc}, wd={args.weight_decay})")

    # ===== Loss configuration =====
    loss_cfg = SepVAELossConfig(
        weight_rec=args.weight_rec,
        weight_kl_common=args.weight_kl_common,
        weight_kl_disease=args.weight_kl_disease,
        weight_null=args.weight_null,
        weight_mi=args.weight_mi,
        sigma_inactive=args.sigma_inactive
    )

    print(f"\nLoss weights:")
    print(f"  Reconstruction: {loss_cfg.weight_rec}")
    print(f"  KL (common): {loss_cfg.weight_kl_common}")
    print(f"  KL (disease): {loss_cfg.weight_kl_disease}")
    print(f"  Nulling: {loss_cfg.weight_null}")
    print(f"  MI penalty: {loss_cfg.weight_mi}")

    # ===== Define training steps =====
    # vae_batch_stats is frozen (never updated), captured in closure by jit
    @jax.jit
    def vae_step(vae_state, disc_state, batch, key):
        """Update VAE parameters."""
        def loss_fn(params):
            total_loss, (logs, _) = sepvae_loss(
                sepvae, mi_disc, params, disc_state.params,
                batch, key, loss_cfg, batch_stats=vae_batch_stats
            )
            return total_loss, logs

        (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(vae_state.params)
        vae_state = vae_state.apply_gradients(grads=grads)

        return vae_state, logs

    @jax.jit
    def disc_step(vae_state, disc_state, batch, key):
        """Update MI discriminator parameters."""
        def loss_fn(mi_params):
            _, (_, disc_loss) = sepvae_loss(
                sepvae, mi_disc, vae_state.params, mi_params,
                batch, key, loss_cfg, batch_stats=vae_batch_stats
            )
            return disc_loss

        disc_loss, grads = jax.value_and_grad(loss_fn)(disc_state.params)
        disc_state = disc_state.apply_gradients(grads=grads)

        return disc_state, disc_loss

    # ===== Verbose backbone diagnostics (optional) =====
    if args.verbose_backbone:
        print("\n" + "="*60)
        print("BACKBONE FEATURE DIAGNOSTICS")
        print("="*60)

        # Collect images from all three classes (normal, effusion, cardiomegaly)
        # Each triplet batch has x_norm (label=0), x_disease1 (label=1), x_disease2 (label=2)
        vis_images = []
        vis_labels = []
        n_collected = 0

        for batch_torch in loader:
            x_norm = batch_torch['x_norm'].permute(0, 2, 3, 1).numpy()
            x_dis1 = batch_torch['x_disease1'].permute(0, 2, 3, 1).numpy()
            x_dis2 = batch_torch['x_disease2'].permute(0, 2, 3, 1).numpy()
            B = x_norm.shape[0]

            # Stack all three: (3*B, H, W, C) with matching labels
            vis_images.append(jnp.array(np.concatenate([x_norm, x_dis1, x_dis2], axis=0)))
            vis_labels.append(np.array(batch_torch['disease_labels'].numpy()))
            n_collected += 3 * B

            if n_collected >= args.verbose_n_samples:
                break

        vis_images = jnp.concatenate(vis_images, axis=0)[:args.verbose_n_samples]
        vis_labels = jnp.array(np.concatenate(vis_labels, axis=0)[:args.verbose_n_samples])

        diag_dir = ensure_dir(output_dir / "backbone_diagnostics")
        visualize_backbone_features(
            sepvae, vae_state.params, vis_images, vis_labels,
            save_dir=str(diag_dir), batch_stats=vae_batch_stats
        )

        if args.wandb and _WANDB:
            import glob
            for img_path in sorted(glob.glob(f"{diag_dir}/*.png")):
                name = os.path.basename(img_path).replace('.png', '')
                wandb.log({f"backbone/{name}": wandb.Image(img_path)})

        print(f"Backbone diagnostics saved to: {diag_dir}")

    # ===== Training loop =====
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        epoch_logs = []

        for batch_idx, batch_torch in enumerate(loader):
            # Convert PyTorch tensors to JAX arrays (NCHW → NHWC)
            batch = {
                'x_norm': jnp.array(batch_torch['x_norm'].permute(0, 2, 3, 1).numpy()),
                'x_disease1': jnp.array(batch_torch['x_disease1'].permute(0, 2, 3, 1).numpy()),
                'x_disease2': jnp.array(batch_torch['x_disease2'].permute(0, 2, 3, 1).numpy()),
                'disease_labels': jnp.array(batch_torch['disease_labels'].numpy()),
            }

            # Get random key for this step
            rng, step_key = jax.random.split(rng)

            # Update VAE
            vae_state, vae_logs = vae_step(vae_state, disc_state, batch, step_key)

            # Update MI discriminator
            disc_state, disc_loss = disc_step(vae_state, disc_state, batch, step_key)

            # Add discriminator loss to logs
            vae_logs['loss/mi_disc_update'] = disc_loss

            epoch_logs.append(vae_logs)

            # Logging
            if global_step % args.log_every == 0:
                log_str = (
                    f"Step {global_step}: "
                    f"Loss={vae_logs['loss/total']:.4f}, "
                    f"Rec={vae_logs['loss/reconstruction']:.4f}, "
                    f"KL={vae_logs['loss/kl_total']:.4f}, "
                    f"Null={vae_logs['loss/nulling']:.4f}, "
                    f"MI={vae_logs['loss/mi_penalty']:.4f}"
                )
                print(log_str)

                if args.wandb and _WANDB:
                    # Convert JAX arrays to Python floats for W&B
                    wandb_logs = {k: float(v) for k, v in vae_logs.items()}
                    wandb_logs['epoch'] = epoch
                    wandb.log(wandb_logs, step=global_step)

            global_step += 1

        # Epoch summary
        avg_logs = {
            key: float(jnp.mean(jnp.array([log[key] for log in epoch_logs])))
            for key in epoch_logs[0].keys()
        }

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Total Loss: {avg_logs['loss/total']:.4f}")
        print(f"  Reconstruction: {avg_logs['loss/reconstruction']:.4f}")
        print(f"  KL (total): {avg_logs['loss/kl_total']:.4f}")
        print(f"  Nulling: {avg_logs['loss/nulling']:.4f}")
        print(f"  MI Penalty: {avg_logs['loss/mi_penalty']:.4f}")

        # Save checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:04d}.pkl"

            ckpt_data = {
                'epoch': epoch,
                'global_step': global_step,
                'vae_params': vae_state.params,
                'vae_batch_stats': vae_batch_stats,
                'disc_params': disc_state.params,
                'vae_opt_state': vae_state.opt_state,
                'disc_opt_state': disc_state.opt_state,
                'rng': rng,
                'args': vars(args),
            }

            with open(ckpt_path, 'wb') as f:
                f.write(to_bytes(ckpt_data))

            print(f"✓ Saved checkpoint: {ckpt_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

    # Save final checkpoint
    final_ckpt_path = ckpt_dir / "checkpoint_final.pkl"
    ckpt_data = {
        'epoch': args.epochs,
        'global_step': global_step,
        'vae_params': vae_state.params,
        'vae_batch_stats': vae_batch_stats,
        'disc_params': disc_state.params,
        'vae_opt_state': vae_state.opt_state,
        'disc_opt_state': disc_state.opt_state,
        'rng': rng,
        'args': vars(args),
    }

    with open(final_ckpt_path, 'wb') as f:
        f.write(to_bytes(ckpt_data))

    print(f"✓ Saved final checkpoint: {final_ckpt_path}")

    if args.wandb and _WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
