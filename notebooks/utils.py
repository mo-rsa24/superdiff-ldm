import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import AutoencoderKL, UNet2DConditionModel, SD3Transformer2DModel, StableDiffusionGLIGENPipeline
from PIL import Image
import matplotlib.pyplot as plt


def get_sd_models(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    dtype=torch.float16,
    device=torch.device("cuda"),
):  # CompVis/stable-diffusion-v1-4
    is_sdxl = "stable-diffusion-xl" in model_id.lower()

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype, use_safetensors=True).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype, use_safetensors=True).to(device)

    if is_sdxl:
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True
        ).to(device)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=dtype, use_safetensors=True
        ).to(device)
        return {
            "vae": vae,
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "tokenizer_2": tokenizer_2,
            "text_encoder_2": text_encoder_2,
            "unet": unet,
            "is_sdxl": True,
        }

    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True
    ).to(device)
    return {
        "vae": vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "unet": unet,
        "is_sdxl": False,
    }


def get_gligen_models(
    model_id: str = "gligen/gligen-generation-text-box",
    dtype=torch.float16,
    device=torch.device("cuda"),
):
    """Load GLIGEN pipeline components (UNet + single CLIP text encoder + VAE)."""
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        model_id, torch_dtype=dtype,
    ).to(device)
    return {
        "vae": pipe.vae,
        "unet": pipe.unet,
        "tokenizer": pipe.tokenizer,
        "text_encoder": pipe.text_encoder,
        "scheduler": pipe.scheduler,
        "pipeline": pipe,
        "is_gligen": True,
    }


def get_sd3_models(
    model_id: str = "stabilityai/stable-diffusion-3.5-medium",
    dtype=torch.bfloat16,
    device=torch.device("cuda"),
):
    """Load SD3 transformer + triple text encoders + VAE."""
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype, use_safetensors=True,
    ).to(device)
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype, use_safetensors=True,
    ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True,
    ).to(device)

    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=dtype, use_safetensors=True,
    ).to(device)

    tokenizer_3 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3")
    text_encoder_3 = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder_3", torch_dtype=dtype, use_safetensors=True,
    ).to(device)

    return {
        "vae": vae,
        "transformer": transformer,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "tokenizer_2": tokenizer_2,
        "text_encoder_2": text_encoder_2,
        "tokenizer_3": tokenizer_3,
        "text_encoder_3": text_encoder_3,
        "is_sd3": True,
    }


@torch.no_grad()
def get_sd3_text_embedding(
    prompt,
    tokenizer,
    text_encoder,
    tokenizer_2,
    text_encoder_2,
    tokenizer_3,
    text_encoder_3,
    device=torch.device("cuda"),
    max_sequence_length: int = 256,
):
    """
    Encode prompt with SD3 triple text encoders.

    Returns (prompt_embeds, pooled_prompt_embeds):
        prompt_embeds:        (B, clip_seq + t5_seq, 4096)
        pooled_prompt_embeds: (B, pooled_dim)

    Matches official SD3 packing:
      1) CLIP-L and CLIP-G are concatenated on feature dim -> (B, 77, 2048)
      2) CLIP block is padded to T5 hidden dim (4096)
      3) CLIP block is concatenated with T5 block on sequence dim
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    # --- CLIP-L (text_encoder) ---
    text_input_1 = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    out_1 = text_encoder(text_input_1.input_ids.to(device), output_hidden_states=True)
    prompt_embeds_1 = out_1.hidden_states[-2]      # (B, seq, 768)
    pooled_1 = out_1[0]                             # (B, 768)

    # --- CLIP-G (text_encoder_2) ---
    text_input_2 = tokenizer_2(
        prompt, padding="max_length", max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    )
    out_2 = text_encoder_2(text_input_2.input_ids.to(device), output_hidden_states=True)
    prompt_embeds_2 = out_2.hidden_states[-2]      # (B, seq, 1280)
    pooled_2 = out_2[0]                             # (B, 1280)

    # --- T5-XXL (text_encoder_3) ---
    text_input_3 = tokenizer_3(
        prompt, padding="max_length", max_length=max_sequence_length,
        truncation=True, return_tensors="pt",
    )
    t5_embeds = text_encoder_3(text_input_3.input_ids.to(device))[0]  # (B, t5_seq, 4096)

    # Official SD3 packing:
    # CLIP-L and CLIP-G share the same 77 token positions and are merged on feature dim:
    #   (B, 77, 768) + (B, 77, 1280) -> (B, 77, 2048)
    clip_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

    # Pad merged CLIP block to T5 hidden size (4096)
    t5_dim = t5_embeds.shape[-1]
    clip_embeds = F.pad(clip_embeds, (0, t5_dim - clip_embeds.shape[-1]))

    # Concatenate along sequence: CLIP (77) + T5 (256) = 333
    prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)  # (B, 333, 4096)

    # Pooled = concat of both CLIP pooled
    pooled_prompt_embeds = torch.cat([pooled_1, pooled_2], dim=-1)  # (B, 2048)

    return prompt_embeds, pooled_prompt_embeds


@torch.no_grad()
def get_image(vae, latents, nrow, ncol):
    # Ensure latents are same dtype as VAE
    latents = latents.to(dtype=vae.dtype)

    # Ensure latents have batch dimension
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)

    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    image = vae.decode(latents / vae.config.scaling_factor + shift_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    # Don't squeeze! Keep batch dimension for permute
    image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)

    rows = []
    for row_i in range(nrow):
        row = []
        for col_i in range(ncol):
            i = row_i * nrow + col_i
            if i < len(image):
                row.append(image[i])
            else:
                row.append(torch.zeros_like(image[0]))
        rows.append(torch.hstack(row))
    image = torch.vstack(rows)
    return Image.fromarray(image.cpu().numpy())

# et_vel(unet, t, sigma, latents, embeddings, eps=None, get_div=False, device=torch.device("cuda"), dtype=torch.float16)
@torch.no_grad()
def get_text_embedding(
    prompt,
    tokenizer,
    text_encoder,
    device=torch.device("cuda"),
    tokenizer_2=None,
    text_encoder_2=None,
    return_pooled: bool = False,
):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    # SDXL requires concatenated embeddings from two text encoders.
    if tokenizer_2 is not None and text_encoder_2 is not None:
        text_input_2 = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        prompt_embeds_1 = text_encoder(
            text_input.input_ids.to(device),
            output_hidden_states=True,
        ).hidden_states[-2]

        prompt_outputs_2 = text_encoder_2(
            text_input_2.input_ids.to(device),
            output_hidden_states=True,
        )
        prompt_embeds_2 = prompt_outputs_2.hidden_states[-2]

        pooled_prompt_embeds = prompt_outputs_2[0]
        if pooled_prompt_embeds.ndim != 2:
            pooled_prompt_embeds = prompt_outputs_2.text_embeds

        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        if return_pooled:
            return prompt_embeds, pooled_prompt_embeds
        return prompt_embeds

    prompt_embeds = text_encoder(text_input.input_ids.to(device))[0]
    if return_pooled:
        return prompt_embeds, None
    return prompt_embeds

def plot_trajectories(ll_obj, ll_bg, kappa):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot((ll_obj - ll_bg).cpu().numpy(), c='royalblue')
    plt.ylabel('logp_obj - logp_bg')
    plt.xlabel('num iterations')
    plt.grid()
    plt.subplot(122)
    plt.plot(kappa.cpu().numpy(), c='royalblue')
    plt.ylabel('kappa')
    plt.xlabel('num iterations')
    plt.grid()
