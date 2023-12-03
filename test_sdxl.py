import torch
from diffusers import DiffusionPipeline, LCMScheduler
import os
import time

save_dir = "latent-consistency_output/lcm-lora-sdxl"

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

# prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"

generator = torch.manual_seed(42)

start_time = time.time()

all_times = []
for i in range(10):
    image = pipe(
        prompt=prompt, num_inference_steps=1, generator=generator, guidance_scale=1.0
    ).images[0]

    end_time = time.time()
    all_times.append(end_time - start_time)
    start_time = time.time()

    print("end time: ", end_time)


print("mean time taken for inference: ", sum(all_times) / len(all_times))
# image = pipe(
#     prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=1.0
# ).images[0]

# print("time taken for inference original: ", time.time() - start_time)


# save image
from PIL import Image

image.save(os.path.join(save_dir, "old_man.png"))
