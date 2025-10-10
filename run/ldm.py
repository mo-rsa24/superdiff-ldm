# run/ldm.py
import argparse
import os
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax.training.train_state import TrainState
from flax.serialization import from_bytes, to_bytes
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from torchvision.utils import save_image
import torch
from typing import Any
from datasets.ChestXRay import ChestXrayDataset
from diffusion.vp_equation import alpha_fn, marginal_prob_std_fn, diffusion_coeff_fn
from models.ae_kl import AutoencoderKL
from models.cxr_unet import ScoreNet
from diffusion.sampling import Euler_Maruyama_sampler  # make sure this has the corrected drift

# W&B is optional
try:
    import wandb
    _WANDB = True
except ImportError:
    wandb = None
    _WANDB = False


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# ── Pretty + parseable step blocks ─────────────────────────────────────────────
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule
    _RICH = True
    _console = Console(log_time=False, log_path=False)
except Exception:
    _RICH = False
    _console = None

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def open_block(kind: str, step: int = None, epoch: int = None, note: str = ""):
    tag = f"{kind.upper()} | step={step} epoch={epoch} time={_now()}"
    # machine-parsable sentinels (easy to grep):
    print(f"\n<<<{kind.upper()}_BEGIN step={step} epoch={epoch}>>>", flush=True)
    if _RICH:
        _console.rule(f"[bold cyan]{tag}")
        if note:
            _console.print(note, style="dim")
    else:
        print("=" * 100)
        print(tag)
        if note: print(note)
        print("-" * 100)
    # keep stdout unbuffered on clusters: export PYTHONUNBUFFERED=1

def close_block(kind: str, step: int = None):
    tag = f"END {kind.upper()} | step={step} time={_now()}"
    if _RICH:
        _console.rule(f"[bold cyan]{tag}")
    else:
        print("-" * 100)
    print(f"<<<{kind.upper()}_END step={step}>>>", flush=True)

def pretty_table(title: str, metrics: dict):
    if _RICH:
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")
        for k, v in metrics.items():
            table.add_row(k, f"{float(v):.6f}" if isinstance(v, (float, int)) else str(v))
        _console.print(table)
    else:
        keys = list(metrics.keys())
        w = max(len(k) for k in keys) if keys else 10
        print(f"{title}")
        for k in keys:
            v = metrics[k]
            v = f"{float(v):.6f}" if isinstance(v, (float, int)) else str(v)
            print(f"{k:<{w}} : {v}")

def log_sample_diversity(samples_np: np.ndarray, step: int, epoch: int):
    """Calculates and logs pairwise MSE to check for sample collapse."""
    if samples_np.ndim != 4:
        raise ValueError(f"Expected samples_np to be 4D (B, H, W, C), but got {samples_np.shape}")
    if samples_np.shape[0] <= 1:
        print("[diversity] Batch size is 1, skipping diversity check.")
        return

    batch_size = samples_np.shape[0]
    # Flatten each image into a vector
    flattened_samples = samples_np.reshape(batch_size, -1)

    # Calculate pairwise Mean Squared Error (MSE)
    # Using broadcasting for an efficient calculation: (a - b)^2 = a^2 - 2ab + b^2
    sum_sq = np.sum(flattened_samples**2, axis=1, keepdims=True)
    dot_prod = flattened_samples @ flattened_samples.T
    # mse[i, j] = mean((img[i] - img[j])^2)
    mse_matrix = (sum_sq + sum_sq.T - 2 * dot_prod) / flattened_samples.shape[1]

    # We only need the upper triangle of the matrix (excluding the diagonal)
    # as the matrix is symmetric and mse(i, i) is 0.
    indices = np.triu_indices(batch_size, k=1)
    pairwise_mse_vals = mse_matrix[indices]

    metrics = {
        "pairwise_mse_mean": float(np.mean(pairwise_mse_vals)),
        "pairwise_mse_std": float(np.std(pairwise_mse_vals)),
        "pairwise_mse_min": float(np.min(pairwise_mse_vals)),
        "pairwise_mse_max": float(np.max(pairwise_mse_vals)),
    }

    open_block("diversity", step=step, epoch=epoch, note="Pairwise MSE between generated samples in the batch")
    pretty_table("sample_diversity/metrics", metrics)
    close_block("diversity", step=step)
    if _WANDB and wandb.run is not None:
        wandb.log({"sample_diversity/metrics": metrics, "epoch": epoch, "step": step})

class TrainStateWithEMA(TrainState):
    ema_params: Any = None

def parse_args():
    p = argparse.ArgumentParser("JAX Latent Diffusion Model (CXR) Trainer")
    # --- Data & Debugging ---
    p.add_argument("--data_root", default="../datasets/cleaned")
    p.add_argument("--task", choices=["TB", "PNEUMONIA"], default="TB")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--class_filter", type=int, default=1,
                   help="Optional: keep a class index only (e.g., 1 for disease, 0 for normal)")
    p.add_argument("--overfit_one", action="store_true", help="Repeat a single sample to overfit.")
    # p.add_argument(
    #     "--overfit_one",
    #     type=bool,
    #     default=False,
    #     help="Overfit on a single batch of data."
    # )
    p.add_argument("--overfit_k", type=int, default=0, help="If >0, train on a fixed tiny subset of size K.")
    p.add_argument("--repeat_len", type=int, default=500,
                   help="Virtual length for the repeated one-sample dataset.")

    # --- Pretrained Autoencoder ---
    p.add_argument("--ae_ckpt_path", required=True, help="Path to the last.flax of the pretrained autoencoder.")
    p.add_argument("--ae_config_path", required=True, help="Path to the run_meta.json of the AE run.")
    p.add_argument("--latent_scale_factor", type=float, default=1.0, help="From stable-diffusion v1.")

    # --- LDM UNet Architecture ---
    p.add_argument("--ldm_ch_mults", type=str, default="1,2,4", help="Channel multipliers for UNet, relative to base_ch.")
    p.add_argument("--ldm_base_ch", type=int, default=128)
    p.add_argument("--ldm_num_res_blocks", type=int, default=2)
    p.add_argument("--ldm_attn_res", type=str, default="16", help="Resolutions for attention blocks, e.g., '16,8'")

    # --- Optimizer ---
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_per_device", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # --- Logging & Checkpoints ---
    p.add_argument("--output_root", default="runs_ldm")
    p.add_argument("--exp_name", default="cxr_ldm")
    p.add_argument("--run_name", default=None)
    p.add_argument("--resume_dir", default=None)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--sample_batch_size", type=int, default=16)
    # 2. Add EMA command-line arguments
    p.add_argument("--use_ema", action="store_true", help="Enable EMA for model parameters.")
    p.add_argument("--ema_decay", type=float, default=0.999, help="Decay rate for EMA.")
    p.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases")
    p.add_argument("--wandb_project", default="cxr-ldm")
    p.add_argument("--wandb_tags", default="")
    return p.parse_args()


def load_autoencoder(config_path, ckpt_path):
    print(f"Loading AE from config: {config_path}")
    with open(config_path, 'r') as f:
        ae_args = json.load(f)

    # Parse ch_mults relative to base_ch if needed
    if isinstance(ae_args['ch_mults'], str):
        ch_mult_factors = tuple(int(c.strip()) for c in ae_args['ch_mults'].split(',') if c.strip())
        base_ch = ae_args.get('base_ch', 64)
        ae_ch_mults = tuple(base_ch * m for m in ch_mult_factors)
    else:
        ae_ch_mults = tuple(ae_args['ch_mults'])

    attn_res = tuple(int(r) for r in ae_args.get('attn_res', '16').split(',') if r)

    enc_cfg = dict(ch_mults=ae_ch_mults, num_res_blocks=ae_args['num_res_blocks'], z_ch=ae_args['z_channels'],
                   double_z=True, attn_resolutions=attn_res, in_ch=1)
    dec_cfg = dict(ch_mults=ae_ch_mults, num_res_blocks=ae_args['num_res_blocks'], out_ch=1,
                   attn_resolutions=attn_res)

    ae_model = AutoencoderKL(enc_cfg=enc_cfg, dec_cfg=dec_cfg, embed_dim=ae_args['embed_dim'])

    # Init variables
    rng = jax.random.PRNGKey(0)
    fake_img = jnp.ones((1, ae_args['img_size'], ae_args['img_size'], 1))
    ae_variables = ae_model.init({'params': rng, 'dropout': rng}, fake_img, rng=rng)

    # Optimizer scaffold to load checkpoint
    def get_ae_tx(lr, grad_clip, weight_decay):
        return optax.chain(
            optax.clip_by_global_norm(grad_clip) if grad_clip > 0 else optax.identity(),
            optax.adamw(lr, weight_decay=weight_decay)
        )

    tx = get_ae_tx(lr=ae_args.get('lr', 1e-4), grad_clip=ae_args.get('grad_clip', 1.0),
                   weight_decay=ae_args.get('weight_decay', 1e-4))

    gen_params = {'ae': ae_variables['params']}
    from losses.lpips_gan import LPIPSWithDiscriminatorJAX, LPIPSGANConfig
    loss_cfg = LPIPSGANConfig(disc_num_layers=ae_args.get('disc_layers', 3))
    loss_mod = LPIPSWithDiscriminatorJAX(loss_cfg)
    loss_params_dummy = loss_mod.init({'params': rng}, x_in=fake_img, x_rec=fake_img, posterior=None, step=jnp.array(0))['params']
    disc_params_dummy = {'loss': loss_params_dummy}

    dummy_gen_state = TrainState.create(apply_fn=None, params=gen_params, tx=tx)
    dummy_disc_state = TrainState.create(apply_fn=None, params=disc_params_dummy, tx=tx)

    print(f"Loading AE checkpoint from: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()

    restored_gen_state, _ = from_bytes((dummy_gen_state, dummy_disc_state), blob)
    print("Autoencoder loaded successfully.")
    return ae_model, restored_gen_state.params['ae']


def main():
    args = parse_args()
    print(f"[config] latent_scale_factor = {args.latent_scale_factor}")
    if args.use_ema:
        print(f"✅ EMA is enabled with a decay rate of {args.ema_decay}")
    else:
        print("❌ EMA is disabled for this run.")
    rng = jax.random.PRNGKey(args.seed)

    # --- Setup Directories ---
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.resume_dir if args.resume_dir else os.path.join(args.output_root, args.run_name or f"{args.exp_name}-{ts}")
    ckpt_dir = ensure_dir(os.path.join(run_dir, "ckpts"))
    samples_dir = ensure_dir(os.path.join(run_dir, "samples"))
    ckpt_latest = os.path.join(ckpt_dir, "last.flax")
    with open(os.path.join(run_dir, "ldm_meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Setup Dataset ---
    base_ds = ChestXrayDataset(root_dir=args.data_root, task=args.task, split=args.split, img_size=args.img_size,
                               class_filter=args.class_filter)
    batch_size = args.batch_per_device * jax.local_device_count()
    if args.overfit_one:
        ds = Subset(base_ds, [0])
        ds = torch.utils.data.ConcatDataset([ds] * args.repeat_len)
    elif args.overfit_k > 0:
        ds = Subset(base_ds, list(range(min(args.overfit_k, len(base_ds)))))
    else:
        ds = base_ds

    if args.overfit_one:
        print("INFO: Overfitting on one sample. Disabling data loader workers and shuffle.")
        loader_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
    else:
        loader_kwargs = dict(batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    loader = DataLoader(ds, **loader_kwargs)

    # --- Load Pretrained Autoencoder ---
    ae_model, ae_params = load_autoencoder(args.ae_config_path, args.ae_ckpt_path)
    ae_params = jax.device_put_replicated(ae_params, jax.local_devices())

    # --- Setup LDM UNet ---
    with open(args.ae_config_path, 'r') as f:
        ae_args = json.load(f)
    z_channels = ae_args['z_channels']

    # Diagnostics to compute latent spatial size
    if isinstance(ae_args['ch_mults'], str):
        num_downsamples = len(ae_args['ch_mults'].split(',')) - 1
    else:
        num_downsamples = len(ae_args['ch_mults']) - 1
    downsample_factor = 2 ** num_downsamples
    latent_size = args.img_size // downsample_factor

    print("--- Shape & Channel Verification ---")
    print(f"AE z_channels: {z_channels}")
    print(f"AE ch_mults: {ae_args['ch_mults']}")
    print(f"Downsample factor: {downsample_factor}")
    print(f"Expected latent spatial size: {latent_size}x{latent_size}")

    ldm_chans = tuple(args.ldm_base_ch * int(m) for m in args.ldm_ch_mults.split(','))
    attn_res = tuple(int(r) for r in args.ldm_attn_res.split(','))
    ldm_model = ScoreNet(z_channels=z_channels, channels=ldm_chans,
                         num_res_blocks=args.ldm_num_res_blocks, attn_resolutions=attn_res)
    rng, init_rng = jax.random.split(rng)
    fake_latent = jnp.ones((1, latent_size, latent_size, z_channels))
    fake_time = jnp.ones((1,))
    ldm_params = ldm_model.init(init_rng, fake_latent, fake_time)['params']

    # --- Setup TrainState ---
    tx = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adamw(args.lr, weight_decay=args.weight_decay))
    ema_params = ldm_params if args.use_ema else None
    ldm_state = TrainStateWithEMA.create(
        apply_fn=ldm_model.apply,
        params=ldm_params,
        ema_params=ema_params,  # Add this line
        tx=tx
    )
    if args.resume_dir and tf.io.gfile.exists(ckpt_latest):
        print(f"[info] Resuming LDM from {ckpt_latest}")
        with tf.io.gfile.GFile(ckpt_latest, "rb") as f:
            blob = f.read()
        ldm_state = from_bytes(ldm_state, blob)
    ldm_state = jax.device_put_replicated(ldm_state, jax.local_devices())

    # --- Setup W&B ---
    use_wandb = bool(args.wandb and _WANDB)
    if use_wandb:
        tags = []
        if args.wandb_tags:  # Check if the string is not None and not empty
            tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
        if not tags:
            tags = None
        wandb.init(project=args.wandb_project, name=args.run_name or f"{args.exp_name}-{ts}",
                   config=args, tags=tags)

    # ---------------------------------------------------------
    # Precompute a fixed latent z0 for overfit-one (deterministic)
    # ---------------------------------------------------------
    precomputed_z0 = None
    if args.overfit_one:
        one_loader = DataLoader(Subset(base_ds, [0]), batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        (x0, _), = list(one_loader)
        x0 = jnp.asarray(x0.numpy()).transpose(0, 2, 3, 1)  # NCHW -> NHWC
        # Your dataset is in [-1,1]; AE trained on [0,1] → keep this scaling
        x0 = (x0 + 1.0) / 2.0
        unrep_ae_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ae_params))
        posterior0 = ae_model.apply({'params': unrep_ae_params}, x0, method=ae_model.encode, train=False)
        z0 = posterior0.mode() * args.latent_scale_factor  # fixed latent, no encode noise
        # Repeat to global batch then shard
        global_bs = args.batch_per_device * jax.local_device_count()
        z0_tiled = jnp.tile(z0, (global_bs, 1, 1, 1))
        precomputed_z0 = z0_tiled.reshape((jax.local_device_count(), -1) + z0.shape[1:])
        print("Precomputed z0 for overfit-one:", precomputed_z0.shape)

    # --- Define Training Step ---
    def train_step(rng, ldm_state, ae_params, x_batch, precomputed_z0):
        """One pmap-ed training step."""
        # x_batch: images in [0,1]; you already encode -> z elsewhere if needed.
        rng, rng_diff = jax.random.split(rng, 2)

        def loss_fn(ldm_params):
            if precomputed_z0 is not None:
                z = precomputed_z0
            else:
                posterior = ae_model.apply({'params': ae_params}, x_batch, method=ae_model.encode, train=False)
                z = posterior.sample(rng) * args.latent_scale_factor

            # Sample t ~ U(1e-5, 1) and ε ~ N(0, I)
            rng_t, rng_noise = jax.random.split(rng_diff, 2)
            t = jax.random.uniform(rng_t, (z.shape[0],), minval=1e-5, maxval=1.0)
            noise = jax.random.normal(rng_noise, z.shape)

            # VP forward perturbation
            sigma = marginal_prob_std_fn(t)  # σ(t)  [B]
            alpha = alpha_fn(t)  # α(t)  [B]
            sigma_b = sigma[:, None, None, None]
            alpha_b = alpha[:, None, None, None]
            x_t = alpha_b * z + sigma_b * noise  # x_t = α z + σ ε

            # Predict ε and compute simple ε-MSE
            eps_hat = ldm_model.apply({'params': ldm_params}, x_t, t)  # [B,H,W,C]
            loss = jnp.mean((eps_hat - noise) ** 2)
            def _cos(a, b, eps=1e-8):
                num = jnp.sum(a * b, axis=tuple(range(1, a.ndim)))
                den = jnp.linalg.norm(a.reshape(a.shape[0], -1), axis=-1) * \
                      jnp.linalg.norm(b.reshape(b.shape[0], -1), axis=-1) + eps
                return num / den

            aux = dict(
                t_mean=float(jnp.mean(t)),
                sigma_mean=float(jnp.mean(sigma)),
                alpha_mean=float(jnp.mean(alpha)),
                cos_eps=float(jnp.mean(_cos(eps_hat, noise))),
                z_mean=float(jnp.mean(z)), z_std=float(jnp.std(z)),
                xt_mean=float(jnp.mean(x_t)), xt_std=float(jnp.std(x_t)),
                eps_hat_mean=float(jnp.mean(eps_hat)), eps_hat_std=float(jnp.std(eps_hat)),
            )
            return loss, aux
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ldm_state.params)
        grad_norm = optax.global_norm(grads)
        aux = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='device'), aux)
        aux = {**aux, "grad_norm": jax.lax.pmean(grad_norm, axis_name='device')}
        loss = jax.lax.pmean(loss, axis_name='device')
        new_ldm_state = ldm_state.apply_gradients(grads=grads)
        if args.use_ema:
            new_ema_params = optax.incremental_update(new_ldm_state.params, ldm_state.ema_params, args.ema_decay)
            new_ldm_state = new_ldm_state.replace(ema_params=new_ema_params)

        return new_ldm_state, loss, aux

    pmapped_train_step = jax.pmap(train_step, axis_name='device')


    # --- Training Loop ---
    global_step = int(ldm_state.step[0])
    for ep in range(args.epochs):
        progress_bar = tqdm(loader, desc=f"Epoch {ep + 1}/{args.epochs}", leave=False)
        for batch in progress_bar:
            x, _ = batch
            x = jnp.asarray(x.numpy()).transpose(0, 2, 3, 1)
            # Your dataset tensor is in [-1,1]; AE expects [0,1]
            x = (x + 1.0) / 2.0
            x_sharded = x.reshape((jax.local_device_count(), -1) + x.shape[1:])

            rng, step_rng = jax.random.split(rng)
            rng_sharded = jax.random.split(step_rng, jax.local_device_count())
            # ldm_state, loss = pmapped_train_step(rng_sharded, ldm_state, ae_params, x_sharded, precomputed_z0)
            ldm_state, loss, aux = pmapped_train_step(rng_sharded, ldm_state, ae_params, x_sharded, precomputed_z0)

            if global_step % args.log_every == 0:
                loss_val = float(np.asarray(loss[0]))
                aux_host = jax.tree_map(lambda x: float(np.asarray(x[0])), aux)  # take device 0
                metrics = {
                    "step": global_step,
                    "loss": loss_val,
                    "grad_norm": aux_host["grad_norm"],
                    "z.mean": aux_host["z_mean"], "z.std": aux_host["z_std"],
                    "z.min": aux_host["z_min"], "z.max": aux_host["z_max"],
                    "ε.target_mean": aux_host["target_mean"], "ε.target_std": aux_host["target_std"],
                    "ε.pred_mean": aux_host["pred_mean"], "ε.pred_std": aux_host["pred_std"],
                }
                open_block("train", step=global_step, epoch=ep + 1, note="per-device means (pmean)")
                pretty_table("train/metrics", metrics)
                close_block("train", step=global_step)
                if use_wandb:
                    wandb.log({"train/loss": loss_val, "train/step": global_step})
            global_step += 1

        # --- Sampling & Checkpointing ---
        if (ep + 1) % args.sample_every == 0:
            print(f"Sampling at epoch {ep + 1}...")
            print(f"[sampling] using z_std = {1.0 / args.latent_scale_factor}")

            rng, sample_rng = jax.random.split(rng)

            # Unreplicate params for inference
            unrep_ldm_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ldm_state.params))
            unrep_ae_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ae_params))

            if args.use_ema and ldm_state.ema_params is not None:
                print("[info] Using EMA parameters for sampling.")
                sampling_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ldm_state.ema_params))
            else:
                sampling_params = unrep_ldm_params

            # --- sanity A: decode the fixed training latent (z0) directly ---
            if precomputed_z0 is not None:
                # take device-0 slice and undo sharding
                z0_host = jax.device_get(precomputed_z0[0, 0:1, ...])  # shape (1,H,W,C) on host
                # IMPORTANT: the AE expects latents divided by the scale factor
                z0_for_decode = z0_host / args.latent_scale_factor
                x0_hat = ae_model.apply({'params': unrep_ae_params}, z0_for_decode, method=ae_model.decode, train=False)
                x0_hat = jnp.clip(x0_hat, 0., 1.)
                x0_hat = jnp.transpose(x0_hat, (0, 3, 1, 2))  # NHWC -> NCHW
                x0_hat_t = torch.from_numpy(np.asarray(x0_hat))
                save_image(x0_hat_t, os.path.join(samples_dir, f"sanityA_recon_z0_ep{ep + 1:04d}.png"))

            # --- sanity B: decode a noisy latent at mid-time (t=0.5) ---
            if precomputed_z0 is not None:
                mid_t = jnp.ones((1,)) * 0.5
                std_mid = marginal_prob_std_fn(mid_t)[0]
                rng_tmp = jax.random.PRNGKey(123)
                noisy = z0_host + std_mid * jax.random.normal(rng_tmp, z0_host.shape)
                noisy_for_decode = noisy / args.latent_scale_factor
                x_mid = ae_model.apply({'params': unrep_ae_params}, noisy_for_decode, method=ae_model.decode,
                                       train=False)
                x_mid = jnp.clip(x_mid, 0., 1.)
                x_mid = jnp.transpose(x_mid, (0, 3, 1, 2))
                x_mid_t = torch.from_numpy(np.asarray(x_mid))
                save_image(x_mid_t, os.path.join(samples_dir, f"sanityB_decode_noisy_latent_ep{ep + 1:04d}.png"))
            if args.use_ema:
                open_block("ema", step=global_step, epoch=ep + 1, note="Validate EMA Impact")
                flat_params = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(unrep_ldm_params)])
                flat_ema_params = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(sampling_params)])
                cosine_sim = jnp.dot(flat_params, flat_ema_params) / (
                            jnp.linalg.norm(flat_params) * jnp.linalg.norm(flat_ema_params))
                print(f"[info] Cosine similarity between base and EMA weights: {cosine_sim:.6f}")
                pretty_table("ema", metrics)
                close_block("ema", step=global_step)

                if use_wandb:
                    wandb.log({"train/cosine_similarity": float(cosine_sim), "epoch": ep + 1})

            open_block("sample", step=global_step, epoch=ep + 1, note="Euler-Maruyama SDE Sampler")
            sample_rng = jax.random.fold_in(rng, ep + 1)
            sample_rng = jax.random.fold_in(sample_rng, global_step)
            samples_grid, final_latent = Euler_Maruyama_sampler(
                rng=sample_rng,
                ldm_model=ldm_model,
                ldm_params=sampling_params,
                ae_model=ae_model,
                ae_params=unrep_ae_params,
                marginal_prob_std_fn=marginal_prob_std_fn,
                diffusion_coeff_fn=diffusion_coeff_fn,
                latent_size=latent_size,
                batch_size=args.sample_batch_size,
                z_channels=z_channels,
                z_std=(1.0 / args.latent_scale_factor)
            )
            final_latent_np = np.asarray(final_latent)
            log_sample_diversity(final_latent_np, step=global_step, epoch=ep + 1)
            stats = {
                "mean": np.mean(final_latent_np),
                "std": np.std(final_latent_np),
                "min": np.min(final_latent_np),
                "max": np.max(final_latent_np),
            }
            pretty_table("final_latent_stats", stats)
            out_path = os.path.join(samples_dir, f"sample_ep{ep + 1:04d}.png")
            save_image(samples_grid, out_path)
            close_block("sample", step=global_step)
            if use_wandb:
                wandb.log({"samples": wandb.Image(out_path), "epoch": ep + 1})

        # Save checkpoint
        unrep_state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], ldm_state))
        with tf.io.gfile.GFile(ckpt_latest, "wb") as f:
            f.write(to_bytes(unrep_state))

    if use_wandb:
        wandb.finish()
    print(f"Training complete. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], "GPU")
    main()
