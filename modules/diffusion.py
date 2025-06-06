# https://github.com/xhinker/sd_embed
import gc
import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl

class Depth2ImageGenerator:
    def __init__(self, device="cuda"):
        self.device = device
        self._load_models()

    def _load_models(self):
        # Depth estimation models
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas"
        ).to(self.device)
        self.feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

        # ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )

        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )

        # SDXL Pipeline
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            vae=self.vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

    def _get_depth_map(self, image: Image.Image) -> Image.Image:
        image_tensor = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad(), torch.autocast(self.device):
            depth_map = self.depth_estimator(image_tensor).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        depth_image = torch.cat([depth_map] * 3, dim=1)
        depth_image = depth_image.permute(0, 2, 3, 1).cpu().numpy()[0]
        depth_image = Image.fromarray((depth_image * 255.0).clip(0, 255).astype(np.uint8))

        return depth_image

    def generate(self, prompt: str, image_path: str,
                 strength: float = 0.99,
                 controlnet_conditioning_scale: float = 0.4,
                 num_inference_steps: int = 50) -> Image.Image:
        # Load and resize image
        input_image = load_image(image_path).resize((1024, 1024))

        # Get depth map
        depth_image = self._get_depth_map(input_image)

        # prompt embedding to overcome input window length limitation
        prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = get_weighted_text_embeddings_sdxl(
            self.pipe,
            prompt=prompt,
            neg_prompt=""
        )


        # Generate image
        result = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
            image=input_image,
            control_image=depth_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale
        )
        return result.images[0]

if __name__ == "__main__":
    # the prompt token size must lower than 75 due to model capacity.
    prompt = "A captivating, incandescent fantasy portrait of a tranquil celestial woman emerging from a vibrant bed of fiery red flowers. Her radiant, cascading hair is a brilliant shade of molten gold, and she dons a delicate, gossamer-like gown of shimmering gold leaf. Majestic, fiery wings spread wide from her back, their intricate feather patterns resembling flickering flames. A circlet of twisted vines and smoldering embers adorns her brow. She gazes skyward, eyes shut, with arms outstretched in a serene, liberated stance. The backdrop showcases a dramatic, smoke-filled sky, intensifying the mystical and divine ambiance. The illumination is soft and hazy, casting an enchanting, otherworldly radiance over the entire scene."

    generator = Depth2ImageGenerator()
    image = generator.generate(prompt, "./images/2.jpg")

    image.save("./images/test.png")
    print("Image generation completed.")
