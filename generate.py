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

def debug_lora_weights(state_dict):
    for layer_name, layer_params in state_dict.items():
        print(f"Layer: {layer_name}")
        if isinstance(layer_params, dict):
            for param_name, param in layer_params.items():
                print(f"  Param: {param_name}, Shape: {param.shape}")
        else:
            print(f"  Shape: {layer_params.shape}")


# Function to load weights into the UNet model
def load_weights(unet, lora_weights_path):
    state_dict = torch.load(lora_weights_path)
    # debug_lora_weights(state_dict)

    # Load whatever are available in state dict into the UNet model
    unet.load_state_dict(state_dict, strict=False)
    # print("------------------------------------------------")
    # debug_lora_weights(unet.state_dict())
    return unet

def load_model_with_lora():
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32
    )
    pipeline.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    # Apply LoRA to the UNet model
    pipeline.unet = apply_lora_to_unet(pipeline.unet)
    return pipeline

# Load the original Stable Diffusion pipeline
def load_finetuned_pipeline(lora_weights_path):
    # Load the base pipeline
    pipeline = load_model_with_lora()

    # Apply LoRA weights to the UNet model
    pipeline.unet = load_weights(pipeline.unet, lora_weights_path)
    return pipeline

# Generate new images with the fine-tuned pipeline
def generate_image(pipeline, prompt, num_inference_steps=50, guidance_scale=7.5):
    # print("\n=== Checking UNet Layers ===")
    # for name, module in pipeline.unet.named_modules():
    #     if isinstance(module, LoRALayer):
    #         print(f"LoRALayer found: {name}")
    #     else:
    #         print(f"Original layer: {name} - {module}")

    print("\n=== Generating Image ===")
    print(f"Prompt: {prompt}")
    print(f"Device: {pipeline.device}")

    try:
        with torch.no_grad():
            output = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
            image = output.images[0]
        print("Image generated successfully.")
        return image
    except Exception as e:
        print(f"Error during image generation: {e}")
        raise e



# Path to your fine-tuned LoRA weights
LORA_WEIGHTS_PATH = "./lora_weights/fine_tuned_lora_light.pt"

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

