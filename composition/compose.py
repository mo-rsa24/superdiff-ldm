import os
import json
from composition.dynamics import stochastic_super_diff_and_uncond, ddpm_ancestral_superdiff_and_uncond, \
    generate_ldm_samples, get_sweep_configuration
from composition.utils import decode_image, save_image_grid, load_ldm_state, parse_args, load_models
import jax
from operator import itemgetter
import numpy as np

from notebooks.superdiff_sweep import create_labelled_grid

LIFT_VALUES = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]  # Sorted for logical interpolation
NUM_ROWS = 4
NUM_COLS = len(LIFT_VALUES)

def main():
    args = parse_args()
    meta_path = os.path.join(args.run_dir_normal, "ldm_meta.json")
    with open(meta_path, 'r') as f:
        config_1 = json.load(f)
    latent_scale_factor = args.latent_scale_factor

    # 2. Load Models
    ae_model, ae_params, normal, tb = load_models(config_1, args.run_dir_normal, args. run_dir_tb)
    model_normal, params_normal, cfg1, lsize, zch = itemgetter("ldm_model", "params",  "config", "latent_size",  "z_channels")(normal)
    model_tb, params_tb, cfg2, _, _ = itemgetter("ldm_model", "params", "config", "latent_size", "z_channels")(tb)

    # 3. Initialize Latents
    rng = jax.random.PRNGKey(args.seed)
    latent_shape = (args.batch_size, lsize, lsize, zch)
    latents = jax.random.normal(rng, latent_shape)
    lift = args.lift if args.sampler == 'Euler' else None
    if args.sweep:
        latents, lift = get_sweep_configuration(lsize, z_channels = zch, seed = args.seed)

    if args.sample_images:
        generate_ldm_samples(rng, normal, tb, ae_model, ae_params,
                             batch_size = args.batch_size, sampler = args.sampler, n_steps = args.steps, latent_scale_factor = args.latent_scale_factor , output_path =  args.output_path)

    # 4. Run Stochastic SuperDiff
    if args.sampler == 'Ancestral':
        print("Running Ancestral SuperDiff Composition...")
        final_latents, kappas = ddpm_ancestral_superdiff_and_uncond(
            rng,
            latents,
            model_normal, params_normal,
            model_tb, params_tb,
            num_inference_steps=args.steps,
            lift=lift,
            kappa_clip=2.0,
        )
    elif args.sampler == 'Euler':
        print("Running Euler SuperDiff Composition...")
        final_latents, kappas = stochastic_super_diff_and_uncond(
            latents,
            model_normal, params_normal,
            model_tb, params_tb,
            num_inference_steps=args.steps,
            lift=lift,
            score=args.score
        )


    # 5. Decode and Save
    print("Decoding images...")
    images = decode_image(ae_model, ae_params, final_latents, latent_scale_factor=latent_scale_factor)
    images_np = np.array(images)  # [B, H, W, C]
    if args.sweep:
        create_labelled_grid(images_np, NUM_ROWS, NUM_COLS, LIFT_VALUES, args.output_path)
    else:
        save_image_grid(images_np, args.output_path)
    print(f"Result saved to {args.output_path}")

if __name__ == "__main__":
    main()