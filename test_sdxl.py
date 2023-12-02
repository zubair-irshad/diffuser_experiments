import torch
from diffusers import DiffusionPipeline, LCMScheduler
import os

save_dir = "latent-consistency/lcm-lora-sdxl"

# create save directory relative to current working directory

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16,
).to("cuda")

# set scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# load LCM-LoRA
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

generator = torch.manual_seed(42)
image = pipe(
    prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=1.0
).images[0]


# save image
from PIL import Image

image.save(os.path.join(save_dir, "self-portrait-oil-painting.png"))
