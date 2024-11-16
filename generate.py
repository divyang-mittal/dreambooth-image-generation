import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torch import nn
import os

# Load your LoRA layer class from the fine-tuning script
class LoRALayer(nn.Module):
    def __init__(self, rank=1):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.lora_down = nn.Linear(768, rank, bias=False)
        self.lora_up = nn.Linear(rank, 768, bias=False)

    def forward(self, x):
        return self.lora_up(self.lora_down(x)) + x

# Function to load LoRA weights into the UNet model
def load_lora_weights(unet, lora_weights_path):
    state_dict = torch.load(lora_weights_path)
    for name, module in unet.named_modules():
        if isinstance(module, LoRALayer) and name in state_dict:
            module.load_state_dict(state_dict[name])
    return unet

# Load the original Stable Diffusion pipeline
def load_finetuned_pipeline(lora_weights_path):
    # Load the base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and text encoder
    pipeline.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    pipeline.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")

    # Apply LoRA weights to the UNet model
    pipeline.unet = load_lora_weights(pipeline.unet, lora_weights_path)
    return pipeline

# Generate new images with the fine-tuned pipeline
def generate_image(pipeline, prompt, num_inference_steps=50, guidance_scale=7.5):
    with torch.no_grad():
        image = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

# Path to your fine-tuned LoRA weights
LORA_WEIGHTS_PATH = "./lora_weights/fine_tuned_lora.pt"

# Load the fine-tuned pipeline
pipeline = load_finetuned_pipeline(LORA_WEIGHTS_PATH)

# Define a text prompt to generate an image
# prompt = "A futuristic cityscape at sunset, with flying cars and neon lights"

# Generate and save the image
# generated_image = generate_image(pipeline, prompt)
# generated_image.save("generated_image.png")
# print("Image generated and saved as 'generated_image.png'.")


st.title("Generate Images with Fine-Tuned Stable Diffusion")
prompt = st.text_input("Enter your prompt", "A warrior")
generate_button = st.button("Generate Image")

if generate_button:
    st.write("Generating image...")
    image = generate_image(pipeline, prompt)
    st.image(image, caption="Generated Image")

