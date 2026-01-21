
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image

dtype = torch.float32
device = torch.device("cuda")

def get_sd_models(dtype: torch.dtype = torch.float32, device = torch.device("cuda")):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=dtype, use_safetensors=True).to(device)
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=dtype,subfolder="text_encoder", use_safetensors=True).to(device)
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=type, subfolder="unet", use_safetensors=True).to(device)
    return {"vae": vae, "tokenizer": tokenizer, "text_encoder": text_encoder, "unet": unet,}


@torch.no_grad
def get_image(vae, latents, nrow, ncol):
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)
    rows = []
    for row_i in range(nrow):
        row = []
        for col_i in range(ncol):
            i = row_i*nrow + col_i
            row.append(image[i])
        rows.append(torch.hstack(row))
    image = torch.vstack(rows)
    return Image.fromarray(image.cpu().numpy())

@torch.no_grad
def get_text_embedding(prompt, tokenizer, text_encoder):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    return text_encoder(text_input.input_ids.to(device))[0]