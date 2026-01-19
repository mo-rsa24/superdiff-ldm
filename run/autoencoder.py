# run/autoencoder.py
import argparse, os, json, math, threading, queue
from datetime import datetime
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes

# --- Local datasets & helpers ---
from datasets.ChestXRay import ChestXrayDataset
from models.ae_kl import AutoencoderKL
from losses.lpips_gan import LPIPSWithDiscriminatorJAX, LPIPSGANConfig, PerceptualHook

# Prevent TF from grabbing GPU memory (JAX needs it)
tf.config.set_visible_devices([], "GPU")

try:
    import wandb

    _WANDB = True
except Exception:
    wandb = None
    _WANDB = False


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def make_grid_torch(imgs_tensor, nrow=None):
    from torchvision.utils import make_grid
    N = imgs_tensor.shape[0]
    if nrow is None:
        nrow = int(math.sqrt(max(1, N)))
    return make_grid(imgs_tensor, nrow=nrow)


def int_or_none(value):
    if str(value).lower() == 'none': return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer or 'None'")


class BackgroundGenerator:
    """Prefetches PyTorch batches to JAX device in a background thread."""

    def __init__(self, generator, max_prefetch=2):
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = threading.Thread(target=self._loop)
        self.daemon.daemon = True
        self.daemon.start()

    def _loop(self):
        for item in self.generator:
            # Move to JAX device immediately
            x_np = item[0].numpy()  # item is (x, y)
            x_jax = jax.device_put(jnp.asarray(x_np))
            self.queue.put((x_jax, item[1]))
        self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item


def parse_args():
    p = argparse.ArgumentParser("JAX AutoencoderKL (CXR) trainer")
    # Data
    p.add_argument("--data_root", default="../datasets/cleaned")
    p.add_argument("--task", choices=["TB", "PNEUMONIA", "All_CXR"], default="TB")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--class_filter", type=int_or_none, default=None)
    p.add_argument("--overfit_one", action="store_true")
    p.add_argument("--overfit_k", type=int, default=0)
    p.add_argument("--repeat_len", type=int, default=500)

    # Model
    p.add_argument("--num_res_blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--z_channels", type=int, default=4)
    p.add_argument("--attn_res", type=str, default="16,8")
    p.add_argument("--embed_dim", type=int_or_none, default=None)
    p.add_argument("--base_ch", type=int, default=128)
    p.add_argument("--ch_mults", type=str, default="1,2,4")

    # Loss
    p.add_argument("--kl_weight", type=float, default=1.0e-5)
    p.add_argument("--pixel_weight", type=float, default=1.0)
    p.add_argument("--disc_start", type=int, default=50000)
    p.add_argument("--disc_factor", type=float, default=1.0)
    p.add_argument("--disc_weight", type=float, default=0.1)
    p.add_argument("--disc_layers", type=int, default=2)
    p.add_argument("--disc_loss", choices=["hinge", "vanilla"], default="hinge")
    p.add_argument("--perceptual_weight", type=float, default=0.01)

    # Optimizer
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_per_device", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)

    # Logging
    p.add_argument("--output_root", default="runs")
    p.add_argument("--exp_name", default="cxr_ae")
    p.add_argument("--run_name", default=None)
    p.add_argument("--resume_dir", default=None)
    p.add_argument("--sample_every", type=int, default=20)
    p.add_argument("--log_every", type=int, default=5)

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="cxr-ae")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_tags", default="")
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_id", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)

    # Setup directories
    H = W = int(args.img_size)
    per_dev = max(1, args.batch_per_device)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    ch_mult_factors = tuple(int(c.strip()) for c in args.ch_mults.split(',') if c.strip())
    ch_mults = tuple(args.base_ch * m for m in ch_mult_factors)
    embed_dim = args.embed_dim if args.embed_dim is not None else args.z_channels

    # Run naming
    mode = "full"
    if args.overfit_one:
        mode = "of1"
    elif args.overfit_k > 0:
        mode = f"tiny{args.overfit_k}"

    exp_slug = (f"{args.exp_name}-{args.task.lower()}-{args.split}-cxr{H}-{mode}-"
                f"z{args.z_channels}-e{embed_dim}-lr{args.lr:g}")
    run_dir = args.resume_dir if args.resume_dir else os.path.join(args.output_root, args.run_name or exp_slug, ts)

    ckpt_dir = ensure_dir(os.path.join(run_dir, "ckpts"))
    samples_dir = ensure_dir(os.path.join(run_dir, "samples"))
    ckpt_latest = os.path.join(ckpt_dir, "last.flax")

    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump({**vars(args), "run_dir": run_dir}, f, indent=2)

    # Dataset
    base_ds = ChestXrayDataset(
        root_dir=args.data_root, task=args.task, split=args.split,
        img_size=args.img_size, class_filter=args.class_filter
    )
    label_counts = Counter(base_ds.labels)

    class RepeatOne(Dataset):
        def __init__(self, item, length): self.item, self.length = item, length

        def __len__(self): return self.length

        def __getitem__(self, idx): return self.item

    if args.overfit_one:
        ds = RepeatOne(base_ds[0], length=args.repeat_len)
        loader_kwargs = {"shuffle": False, "num_workers": 0}
    elif args.overfit_k > 0:
        ds = Subset(base_ds, list(range(min(args.overfit_k, len(base_ds)))))
        loader_kwargs = {"shuffle": True, "num_workers": 4, "persistent_workers": True}
    else:
        ds = base_ds
        loader_kwargs = {
            "shuffle": True, "num_workers": 8, "drop_last": True,
            "persistent_workers": True, "pin_memory": True, "prefetch_factor": 2
        }

    loader = DataLoader(ds, batch_size=per_dev, **loader_kwargs)

    # Model
    attn_res = tuple(int(r.strip()) for r in args.attn_res.split(',') if r.strip())
    enc_cfg = dict(ch_mults=ch_mults, in_ch=1, z_ch=args.z_channels,
                   num_res_blocks=args.num_res_blocks, dropout=args.dropout,
                   double_z=True, attn_resolutions=attn_res)
    dec_cfg = dict(ch_mults=ch_mults, out_ch=1, z_ch=args.z_channels,
                   num_res_blocks=args.num_res_blocks, dropout=args.dropout,
                   attn_resolutions=attn_res)
    ae = AutoencoderKL(enc_cfg=enc_cfg, dec_cfg=dec_cfg, embed_dim=embed_dim)

    # Loss
    loss_cfg = LPIPSGANConfig(
        disc_start=args.disc_start, kl_weight=args.kl_weight, pixel_weight=args.pixel_weight,
        disc_num_layers=args.disc_layers, disc_in_channels=1, disc_factor=args.disc_factor,
        disc_weight=args.disc_weight, disc_loss=args.disc_loss, perceptual_weight=args.perceptual_weight
    )
    loss_mod = LPIPSWithDiscriminatorJAX(loss_cfg, perc=PerceptualHook(fn=None, weight=args.perceptual_weight))

    # Init
    fake = jnp.ones((per_dev, H, W, 1), dtype=jnp.float32)
    variables = ae.init({'params': rng, 'dropout': rng}, fake, rng=rng, sample_posterior=True, train=True)
    params = variables['params']

    # Init Loss
    fake_loss = jnp.ones((1, 32, 32, 1), dtype=jnp.float32)
    loss_vars = loss_mod.init({'params': rng}, x_in=fake_loss, x_rec=fake_loss, posterior=None, step=jnp.array(0),
                              train=True)
    loss_params = loss_vars['params']

    # Optimizers
    def tx(lr):
        return optax.chain(
            optax.clip_by_global_norm(args.grad_clip) if args.grad_clip > 0 else optax.identity(),
            optax.adamw(lr, weight_decay=args.weight_decay),
        )

    gen_state = TrainState.create(apply_fn=None, params={'ae': params}, tx=tx(args.lr))
    disc_state = TrainState.create(apply_fn=None, params={'loss': loss_params}, tx=tx(args.lr))

    if args.resume_dir and tf.io.gfile.exists(ckpt_latest):
        print(f"[info] resume from {ckpt_latest}")
        with tf.io.gfile.GFile(ckpt_latest, "rb") as f: blob = f.read()
        gen_state, disc_state = from_bytes((gen_state, disc_state), blob)

    if args.wandb and _WANDB:
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.run_name or exp_slug, id=args.wandb_id,
            group=args.wandb_group, resume="allow" if args.wandb_id else None,
            dir=run_dir,
            config={**vars(args), "n_devices": jax.local_device_count(), "label_counts": dict(label_counts)}
        )

    def model_apply(ae_params, x, *, rng, train):
        return ae.apply({'params': ae_params}, x, rng=rng, sample_posterior=True, train=train, dtype=jnp.bfloat16)

    # --- Training Step with Diagnostics and Normalization Fix ---
    @jax.jit
    def gen_step(gen_state, disc_state, x, step):
        # x is [0, 1] at this point
        def loss_fn(params):
            rng1, rng2 = jax.random.split(jax.random.PRNGKey(step))
            x_bf16 = x.astype(jnp.bfloat16)
            # The DiagonalGaussian class is local in ae_kl.py, but we can reconstruct it or rely on call
            # Let's use the standard call to avoid imports issues if helper isn't exported:
            xrec, posterior = model_apply(params['ae'], x_bf16, rng=rng1, train=True)

            xrec_f32 = xrec.astype(jnp.float32)  # [0, 1]

            # NORMALIZATION FIX:
            # LPIPS expects [-1, 1], model uses [0, 1]
            x_norm = x * 2.0 - 1.0  # [0,1] -> [-1,1]
            xrec_norm = xrec_f32 * 2.0 - 1.0

            g_loss, logs_g, d_loss, logs_d = loss_mod.apply(
                {'params': disc_state.params['loss']},
                x_in=x_norm,
                x_rec=xrec_norm,
                posterior=posterior,
                step=jnp.array(step),
                train=True,
                mutable=False
            )

            # --- DIAGNOSTICS ---
            # Extract mean/std from the posterior object (which is DiagonalGaussian)
            # Assuming posterior.mean and posterior.logvar exist
            logs_g['val/z_mean'] = jnp.mean(posterior.mean)
            logs_g['val/z_std'] = jnp.mean(jnp.exp(0.5 * posterior.logvar))
            logs_g['val/z_max'] = jnp.max(jnp.abs(posterior.mean))
            logs_g['val/kl_raw'] = jnp.mean(posterior.kl())

            return g_loss, (logs_g, xrec, posterior)

        (g_loss, (logs_g, xrec, posterior)), grads = jax.value_and_grad(loss_fn, has_aux=True)(gen_state.params)
        gen_state = gen_state.apply_gradients(grads=grads)
        return gen_state, logs_g, xrec, posterior

    @jax.jit
    def disc_step(gen_params, disc_state, x, step):
        rng1 = jax.random.PRNGKey(step)
        xrec, posterior = model_apply(gen_params['ae'], x, rng=rng1, train=True)

        # Norm Fix for Disc
        x_rec_f32 = xrec.astype(jnp.float32)
        x_norm = x * 2.0 - 1.0
        xrec_norm = x_rec_f32 * 2.0 - 1.0

        def loss_fn(dparams):
            g_loss, logs_g, d_loss, logs_d = loss_mod.apply(
                {'params': dparams['loss']},
                x_in=x_norm, x_rec=xrec_norm, posterior=posterior, step=jnp.array(step), train=True, mutable=False
            )
            return d_loss, logs_d

        (d_loss, logs_d), grads = jax.value_and_grad(loss_fn, has_aux=True)(disc_state.params)
        disc_state = disc_state.apply_gradients(grads=grads)
        return disc_state, logs_d

    # --- Loop ---
    global_step = 0
    for ep in tqdm.trange(args.epochs, desc="epochs"):
        # Wrap loader in BackgroundGenerator for prefetching to Device
        prefetch_loader = BackgroundGenerator(loader, max_prefetch=4)

        inner = tqdm.tqdm(prefetch_loader, desc=f"epoch {ep + 1}", leave=False, total=len(loader))
        for batch_jax, _ in inner:
            # batch_jax is already on device, shape (B, 1, H, W) float32 [-1, 1] from Dataset

            # Prepare input: (N, H, W, C)
            x = jnp.transpose(batch_jax, (0, 2, 3, 1))

            # DATASET IS [-1, 1], MODEL IS [0, 1] (Sigmoid)
            # Transform to [0, 1] for model input
            x = (x + 1.0) * 0.5

            gen_state, logs_g, xrec, posterior = gen_step(gen_state, disc_state, x, global_step)
            disc_state, logs_d = disc_step(gen_state.params, disc_state, x, global_step)

            global_step += 1
            if args.wandb and _WANDB and (global_step % max(1, args.log_every) == 0):
                payload = {"train/step": global_step}
                payload.update({k: float(v) for k, v in logs_g.items()})
                payload.update({k: float(v) for k, v in logs_d.items()})
                wandb.log(payload)

        # Checkpointing
        payload = to_bytes((gen_state, disc_state))
        with tf.io.gfile.GFile(ckpt_latest, "wb") as f:
            f.write(payload)

        # Sampling
        if ((ep + 1) % max(1, args.sample_every)) == 0:
            with torch.no_grad():
                # xrec is [0, 1] from model
                xnp = np.asarray(xrec)
                xnp = np.transpose(xnp, (0, 3, 1, 2))
                imgs = torch.tensor(xnp).clamp(0, 1)
                grid = make_grid_torch(imgs, nrow=8)
                grid_np = grid.permute(1, 2, 0).numpy()
                out_path = os.path.join(samples_dir, f"recon_ep{ep + 1:03d}.png")
                from PIL import Image
                Image.fromarray((grid_np * 255).astype(np.uint8)).save(out_path)
                if args.wandb and _WANDB:
                    wandb.log({"samples/recon_grid": wandb.Image(out_path)})

    print(f"[done] run dir: {run_dir}")
    if args.wandb and _WANDB: wandb.finish()


if __name__ == "__main__":
    main()