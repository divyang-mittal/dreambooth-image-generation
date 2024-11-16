import os
import torch
import numpy as np  # Ensure numpy is imported
import streamlit as st
from PIL import Image
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from torch import nn
import torchvision.transforms as transforms
from transformers import CLIPTextModel, CLIPTokenizer

# Define directories for saving images and LoRA weights
IMAGES_PATH = "./uploaded_images/"
LORA_WEIGHTS_PATH = "./lora_weights/"

# Ensure directories exist
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(LORA_WEIGHTS_PATH, exist_ok=True)


class LoRALayer(nn.Module):
    def __init__(self, rank=1):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.lora_down = None
        self.lora_up = None
        self.projection = None
        self.initialized = False

    def initialize_layers(self, in_features, out_features):
        """Dynamically initialize LoRA layers based on input features."""
        self.lora_down = nn.Linear(in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, out_features, bias=False)
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        # Initialize layers dynamically based on input shape if not already initialized
        if not self.initialized:
            in_features = x.shape[-1]
            out_features = in_features  # Assuming the output shape is the same initially
            self.initialize_layers(in_features, out_features)
            self.initialized = True
            # print(f"Initialized LoRA layers: in_features={in_features}, out_features={out_features}")

        # print(f"Input shape: {x.shape}")

        # Apply the LoRA down projection
        y = self.lora_down(x)
        # print(f"LoRA downsampled shape: {y.shape}")

        # Apply the LoRA up projection
        z = self.lora_up(y)
        # print(f"LoRA upsampled shape: {z.shape}")

        # Project input if necessary
        if self.projection:
            x = self.projection(x)
            # print(f"Projected input shape: {x.shape}")

        # Add residual connection if shapes match
        if x.shape == z.shape:
            return z + x
        else:
            # print(f"Shape mismatch: x.shape={x.shape}, z.shape={z.shape}")
            return z


# Function to apply LoRA to UNet
def apply_lora_to_unet(unet, rank=4):
    modules_to_replace = []

    # Gather linear layers to replace with LoRA layers
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_replace.append((name, module))

    # Replace linear layers with LoRA layers
    for name, module in modules_to_replace:
        # print(f"Applying LoRA to {name}")
        lora_layer = LoRALayer(rank)

        # Replace the original layer with a LoRA-enhanced layer
        parent_module, layer_name = get_parent_module(unet, name)
        setattr(parent_module, layer_name, nn.Sequential(module, lora_layer))

    return unet


def get_parent_module(root_module, module_name):
    parts = module_name.split(".")
    module = root_module
    for part in parts[:-1]:  # Go to the parent of the last part
        module = getattr(module, part)
    return module, parts[-1]


# Load base Stable Diffusion model and apply LoRA
def load_model_with_lora():
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    # Load the tokenizer
    pipeline.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    # Apply LoRA to the UNet model
    pipeline.unet = apply_lora_to_unet(pipeline.unet)
    return pipeline


# Helper function to preprocess images
def preprocess_image(pil_image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(pil_image).unsqueeze(0)  # Add batch dimension


# Fine-tune model with DreamBooth and LoRA
def fine_tune_with_dreambooth(pipeline, images, num_steps=100):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pipeline.unet.parameters()), lr=5e-5)

    # Freeze all layers except LoRA layers
    for param in pipeline.unet.parameters():
        param.requires_grad = False
    for name, module in pipeline.unet.named_modules():
        if isinstance(module, LoRALayer):
            for param in module.parameters():
                param.requires_grad = True

    pipeline.unet.train()
    for step in range(num_steps):
        for image in images:
            image_tensor = preprocess_image(image).to(pipeline.device)

            with torch.no_grad():
                image_latents = pipeline.vae.encode(image_tensor).latent_dist.sample()
                image_latents = image_latents * pipeline.vae.config.scaling_factor

            noise = torch.randn_like(image_latents)
            timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (1,)).long().to(pipeline.device)
            noisy_image = pipeline.scheduler.add_noise(image_latents, noise, timesteps)

            # Tokenize the description for encoder input
            text_inputs = pipeline.tokenizer("A description of the image", return_tensors="pt").to(pipeline.device)
            encoder_hidden_states = pipeline.text_encoder(text_inputs.input_ids).last_hidden_state

            output = pipeline.unet(noisy_image, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            loss = torch.nn.functional.mse_loss(output, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Step [{step}/{num_steps}], Loss: {loss.item()}")

    # Save LoRA weights
    lora_weights_path = os.path.join(LORA_WEIGHTS_PATH, "fine_tuned_lora.pt")
    torch.save(
        {name: module.state_dict() for name, module in pipeline.unet.named_modules() if isinstance(module, LoRALayer)},
        lora_weights_path
    )
    print(f"Fine-tuning complete, LoRA weights saved at {lora_weights_path}.")


# Streamlit Interface
st.title("DreamBooth Fine-Tuning with LoRA")
uploaded_files = st.file_uploader("Upload 10 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if len(uploaded_files) == 10:
    st.write("Processing images and fine-tuning model...")

    # Load model with LoRA applied
    pipeline = load_model_with_lora()

    # Save images and start fine-tuning
    images = [Image.open(file) for file in uploaded_files]
    fine_tune_with_dreambooth(pipeline, images)

    # Display download link for LoRA weights
    with open(os.path.join(LORA_WEIGHTS_PATH, "fine_tuned_lora.pt"), "rb") as f:
        st.download_button("Download LoRA Weights", f, file_name="fine_tuned_lora.pt")
else:
    st.warning("Please upload exactly 2 images.")
