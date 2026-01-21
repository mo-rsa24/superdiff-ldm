import matplotlib.pyplot as pl
import os
from operator import itemgetter

from notebooks.dynamics import get_latents, stochastic_super_diff_and
from notebooks.utils import get_sd_models
from diffusers import EulerDiscreteScheduler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

dtype = torch.float32
device = torch.device("cuda")
height, width = 512, 512
latent_height, latent_width = 64, 64
batch_size = 4

models = get_sd_models(dtype, device)
vae, tokenizer, text_encoder, unet = itemgetter(
    "vae", "tokenizer", "text_encoder", "unet"
)(models())
scheduler = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

obj_prompt = ["A Dog On The Left"]
bg_prompt = ["A Cat On The Right"]

latents = get_latents(scheduler, batch_size=batch_size, latent_height=latent_height, latent_width=latent_width)
latents, kappa, ll_obj, ll_bg = stochastic_super_diff_and(latents, obj_prompt, bg_prompt, scheduler, batch_size=batch_size)