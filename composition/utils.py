import os
from operator import itemgetter
import jax
import optax
import flax
import argparse
import json
import jax.numpy as jnp
from PIL import Image
import numpy as np
from flax.training.train_state import TrainState
from typing import Any
from models.cxr_unet import ScoreNet
from notebooks.superdiff_sweep import create_labelled_grid
from run.ldm import load_autoencoder_


class TrainStateWithEMA(TrainState):
    ema_params: Any = None

def parse_args():
    parser = argparse.ArgumentParser(description="SuperDiff AND with two unconditional LDMs (JAX)")
    parser.add_argument("--run_dir_normal", type=str, required=True, help="Path to Normal LDM run directory")
    parser.add_argument("--run_dir_tb", type=str, required=True, help="Path to TB LDM run directory")
    parser.add_argument("--output_path", type=str, default="superdiff_result.png", help="Output image path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_scale_factor", type=float, default=0.99937266)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lift", type=float, default=0.0, help="Lift parameter for SuperDiff stability")
    parser.add_argument("--score", type=bool, default=False, help="Parameter that determines if we should divide by -sigma")
    parser.add_argument("--sweep", type=bool, default=False, help="Parameter that determines if we should generates sampling with varying lift parameter")
    parser.add_argument("--sample_images", type=bool, default=False, help="Parameter that determines if we sample generates")
    parser.add_argument("--sampler", choices=['Euler', 'Ancestral', 'Faithful', 'PoE'], default='Ancestral', help='Sampler to use for sampling from the posterior')
    return parser.parse_args()

def load_ldm_state(run_dir, ckpt_name="last.flax"):
    """
    Helper to load an LDM model and its parameters from a run directory.
    """
    meta_path = os.path.join(run_dir, "ldm_meta.json")
    ckpt_path = os.path.join(run_dir, "ckpts", ckpt_name)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find config at {meta_path}")

    with open(meta_path, 'r') as f:
        config = json.load(f)

    # --- Initialize LDM Architecture ---
    if isinstance(config['ldm_ch_mults'], str):
        ldm_chans = tuple(config['ldm_base_ch'] * int(m) for m in config['ldm_ch_mults'].split(','))
    else:
        ldm_chans = tuple(config['ldm_base_ch'] * int(m) for m in config['ldm_ch_mults'])

    attn_res = tuple(int(r) for r in str(config['ldm_attn_res']).split(','))

    # Determine latent dimensions from AE config
    with open(config['ae_config_path'], 'r') as f:
        ae_cfg_json = json.load(f)
    if isinstance(ae_cfg_json['ch_mults'], str):
        n_down = len(ae_cfg_json['ch_mults'].split(',')) - 1
    else:
        n_down = len(ae_cfg_json['ch_mults']) - 1

    latent_size = config['img_size'] // (2 ** n_down)
    z_channels = ae_cfg_json['z_channels']

    ldm_model = ScoreNet(
        z_channels=z_channels,
        channels=ldm_chans,
        num_res_blocks=config['ldm_num_res_blocks'],
        attn_resolutions=attn_res,
        use_remat=False,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    tx = optax.adamw(1e-4)

    # --- Load Checkpoint ---
    print(f"Loading LDM Checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        blob = f.read()
    raw_state = flax.serialization.msgpack_restore(blob)

    loaded_params = raw_state.get('params')
    loaded_ema_params = raw_state.get('ema_params')

    if loaded_params is None:
        raise ValueError("Checkpoint does not contain 'params'!")

    from flax.core.frozen_dict import freeze
    ldm_state = TrainStateWithEMA.create(
        apply_fn=ldm_model.apply,
        params=freeze(loaded_params),
        ema_params=freeze(loaded_ema_params) if loaded_ema_params else None,
        tx=tx
    )

    # Select params
    use_ema = config.get('use_ema', False)
    params = ldm_state.ema_params if (use_ema and ldm_state.ema_params) else ldm_state.params
    print(f"Loaded {config.get('run_name', 'Unknown')} (EMA={use_ema})")

    return {
        "ldm_model": ldm_model,
        "params": params,
        "config": config,
        "latent_size": latent_size,
        "z_channels": z_channels
    }


def load_models(config_1, run_dir_normal: str, run_dir_tb: str):
    print("Loading Autoencoder...")
    ae_model, ae_params = load_autoencoder_(config_1['ae_config_path'], config_1['ae_ckpt_path'])
    # --- 2. Load Both LDMs ---
    print("Loading Model 1 (Normal)...")
    normal = load_ldm_state(run_dir_normal)
    print("Loading Model 2 (TB)...")
    tb = load_ldm_state(run_dir_tb)
    return ae_model, ae_params, normal, tb

def decode_image(vae, vae_params, latents, latent_scale_factor = 1.0):
    """Decodes latents to images using the VAE."""
    latents = 1 / latent_scale_factor * latents
    imgs = vae.apply({'params': vae_params}, latents, method=vae.decode, train=False)
    imgs = jnp.clip(imgs, 0.0, 1.0)

    # 4. Quantize to uint8
    imgs = (imgs * 255).astype(jnp.uint8)

    # Squeeze channel dim if grayscale: (B, H, W, 1) -> (B, H, W)
    if imgs.shape[-1] == 1:
        imgs = imgs.squeeze(-1)
    return imgs


def save_image_grid(images_np, output_path, cols: int = None, rows: int = None):
    """Saves images as a square grid using PIL (No Torch dependency)."""
    batch_size, h, w = images_np.shape[0], images_np.shape[1], images_np.shape[2]

    # Calculate grid dimensions (e.g., 4 -> 2x2)
    n_cols = int(np.ceil(np.sqrt(batch_size))) if cols is None else cols
    n_rows = int(np.ceil(batch_size / n_cols)) if rows is None else rows

    # Create canvas
    grid_img = Image.new('L', (w * n_cols, h * n_rows))

    for i in range(batch_size):
        row = i // n_cols
        col = i % n_cols
        img = Image.fromarray(images_np[i], mode='L')
        grid_img.paste(img, (col * w, row * h))

    grid_img.save(output_path)


def setup_run(args):
    """Loads models and configurations."""
    meta_path = os.path.join(args.run_dir_normal, "ldm_meta.json")
    with open(meta_path, 'r') as f:
        config_1 = json.load(f)
    ae_model, ae_params, normal, tb = load_models(config_1, args.run_dir_normal, args.run_dir_tb)
    model_n, params_n = itemgetter("ldm_model", "params")(normal)
    model_t, params_t = itemgetter("ldm_model", "params")(tb)
    lsize, zch = itemgetter("latent_size", "z_channels")(normal)

    return ae_model, ae_params, model_n, params_n, model_t, params_t, lsize, zch

def save_results(final_latents, ae_model, ae_params, args):
    """Decodes and saves the final images."""
    print("Decoding images...")
    images = decode_image(ae_model, ae_params, final_latents, latent_scale_factor=args.latent_scale_factor)
    images_np = np.array(images)

    if args.sweep:
        create_labelled_grid(images_np, args.num_rows, len(args.lift_values), args.lift_values, args.output_path)
    else:
        save_image_grid(images_np, args.output_path)

    print(f"Result saved to {args.output_path}")