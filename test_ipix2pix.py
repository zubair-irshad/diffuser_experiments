import torch
from diffusers import (
    AutoPipelineForImage2Image,
    LCMScheduler,
    StableDiffusionXLInstructPix2PixPipeline,
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.utils import make_image_grid, load_image
import os

save_dir = "latent-consistency_output/instructpix2pix"

# create save directory relative to current working directory

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# pipe = AutoPipelineForImage2Image.from_pretrained(
#     "Lykon/dreamshaper-7",
#     torch_dtype=torch.float16,
#     variant="fp16",
# ).to("cuda")

# # set scheduler
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# # load LCM-LoRA
# pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

# # prepare image
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
# init_image = load_image(url)
# prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"

# # pass prompt and image to pipeline
# generator = torch.manual_seed(0)
# image = pipe(
#     prompt,
#     image=init_image,
#     num_inference_steps=4,
#     guidance_scale=1,
#     strength=0.6,
#     generator=generator
# ).images[0]
# make_image_grid([init_image, image], rows=1, cols=2)


# pipe = AutoPipelineForText2Image.from_pretrained('lykon-models/dreamshaper-7', torch_dtype=torch.float16, variant="fp16")
# pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")

# prompt = "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"

# generator = torch.manual_seed(33)
# image = pipe(prompt, generator=generator, num_inference_steps=25).images[0]
# image.save("./image.png")

# original instructpix2pix

from PIL import Image

# edit_type = "original"
edit_type = "lcm-lora-sdxl"

resolution = 768
# image = load_image(
#     "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
# ).resize((resolution, resolution))

# data_path = "/home/ubuntu/zubair/diffuser_experiments/frame_00040.jpg"
# image = Image.open(data_path).resize((resolution, resolution))

data_folder = "/home/ubuntu/zubair/diffuser_experiments/face"

images_files = os.listdir(data_folder)

images = [
    Image.open(os.path.join(data_folder, file)).resize((resolution, resolution))
    for file in images_files
]
# measure time of inference

import time

all_times = []
if edit_type == "original":
    # image = load_image(
    #     "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
    # ).resize((resolution, resolution))

    save_dir_face = os.path.join(save_dir, "elf_out")
    os.makedirs(save_dir_face, exist_ok=True)

    # data_path = "/home/ubuntu/zubair/diffuser_experiments/frame_00040.jpg"
    # image = Image.open(data_path).resize((resolution, resolution))
    edit_instruction = "turn him into a tolkein elf"

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
    ).to("cuda")

    # pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
    #     "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
    # ).to("cuda")

    start_time = time.time()

    for i, image in enumerate(images):
        edited_image = pipe(prompt=edit_instruction, image=image).images[0]

        end_time = time.time()
        all_times.append(end_time - start_time)
        print("time per image: ", end_time - start_time)
        start_time = time.time()

        # print("time per image: ", end_time - start_time)

        # edited_image.save(os.path.join(save_dir, "face_edited_original.png"))

        # save each image in the folder with the different name

        edited_image.save(
            os.path.join(save_dir_face, "face_edited_original_" + str(i) + ".png")
        )

    print("mean time taken for inference: ", sum(all_times) / len(all_times))
    # edited_image = pipe(
    #     prompt=edit_instruction,
    #     image=image
    #     # height=resolution,
    #     # width=resolution,
    #     # guidance_scale=3.0,
    #     # image_guidance_scale=1.5,
    #     # num_inference_steps=30,
    # ).images[0]

    print("time taken for inference original: ", time.time() - start_time)

    # save image
    from PIL import Image

    image.save(os.path.join(save_dir, "face_original.png"))

    edited_image.save(os.path.join(save_dir, "face_edited_original.png"))
# edited_image.save("edited_image.png")


elif edit_type == "lcm-lora":
    resolution = 768
    # image = load_image(
    #     "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
    # ).resize((resolution, resolution))
    edit_instruction = "turn him into a tolkein elf"

    save_dir_face = os.path.join(save_dir, "elf_lora")
    os.makedirs(save_dir_face, exist_ok=True)

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
    ).to("cuda")

    # set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # load LCM-LoRA
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

    # edited_image = pipe(
    #     prompt=edit_instruction,
    #     image=image,
    #     height=resolution,
    #     width=resolution,
    #     guidance_scale=3.0,
    #     image_guidance_scale=1.5,
    #     num_inference_steps=30,
    # ).images[0]

    start_time = time.time()

    for i, image in enumerate(images):
        edited_image = pipe(
            prompt=edit_instruction,
            image=image,
            # height=resolution,
            # width=resolution,
            guidance_scale=1,
            image_guidance_scale=1.5,
            num_inference_steps=6,
        ).images[0]

        end_time = time.time()
        all_times.append(end_time - start_time)
        print("time per image: ", end_time - start_time)
        start_time = time.time()

        # edited_image.save(os.path.join(save_dir, "face_edited_original.png"))

        # save each image in the folder with the different name

        edited_image.save(
            os.path.join(save_dir_face, "face_edited_original_" + str(i) + ".png")
        )

    print("mean time taken for inference: ", sum(all_times) / len(all_times))

elif edit_type == "lcm-lora-sdxl":
    resolution = 768
    # image = load_image(
    #     "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
    # ).resize((resolution, resolution))
    edit_instruction = "turn him into a tolkein elf"

    save_dir_face = os.path.join(save_dir, "elf_lora_sdxl")
    os.makedirs(save_dir_face, exist_ok=True)

    pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        "diffusers/sdxl-instructpix2pix-768", torch_dtype=torch.float16
    ).to("cuda")

    # set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # load LCM-LoRA
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

    # edited_image = pipe(
    #     prompt=edit_instruction,
    #     image=image,
    #     height=resolution,
    #     width=resolution,
    #     guidance_scale=3.0,
    #     image_guidance_scale=1.5,
    #     num_inference_steps=30,
    # ).images[0]

    start_time = time.time()

    for i, image in enumerate(images):
        edited_image = pipe(
            prompt=edit_instruction,
            image=image,
            # height=resolution,
            # width=resolution,
            guidance_scale=1,
            image_guidance_scale=1.5,
            num_inference_steps=6,
        ).images[0]

        end_time = time.time()
        all_times.append(end_time - start_time)
        print("time per image: ", end_time - start_time)
        start_time = time.time()

        # edited_image.save(os.path.join(save_dir, "face_edited_original.png"))

        # save each image in the folder with the different name

        edited_image.save(
            os.path.join(save_dir_face, "face_edited_original_" + str(i) + ".png")
        )

    print("mean time taken for inference: ", sum(all_times) / len(all_times))

    # start_time = time.time()
    # edited_image = pipe(
    #     prompt=edit_instruction,
    #     image=image,
    #     # height=resolution,
    #     # width=resolution,
    #     guidance_scale=1,
    #     image_guidance_scale=1.5,
    #     num_inference_steps=4,
    # ).images[0]

    # print("time taken for inference lcm-lora: ", time.time() - start_time)

    # image.save(os.path.join(save_dir, "face_original.png"))

    # edited_image.save(os.path.join(save_dir, "face_edited_lcm-lora.png"))

# LCM_lora instructpix2pix
# resolution = 768
# image = load_image(
#     "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
# ).resize((resolution, resolution))
# edit_instruction = "Turn sky into a cloudy one"

# pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
#     "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
# ).to("cuda")

# # set scheduler
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# # load LCM-LoRA
# pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

# # edited_image = pipe(
# #     prompt=edit_instruction,
# #     image=image,
# #     height=resolution,
# #     width=resolution,
# #     guidance_scale=3.0,
# #     image_guidance_scale=1.5,
# #     num_inference_steps=30,
# # ).images[0]


# edited_image = pipe(
#     prompt=edit_instruction,
#     image=image,
#     height=resolution,
#     width=resolution,
#     guidance_scale=1.5,
#     image_guidance_scale=1.5,
#     num_inference_steps=4,
# ).images[0]

# edited_image.save(os.path.join(save_dir, "edited_image.png"))
