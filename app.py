import os
import torch
import numpy as np  # Ensure numpy is imported
import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline
from torch import nn
import torchvision.transforms as transforms
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.tensorboard import SummaryWriter
import torch.nn.init as init

# Define directories for saving images and LoRA weights
IMAGES_PATH = "./uploaded_images/"
GENERATED_IMAGES_PATH = "./generated_images/"
LORA_WEIGHTS_PATH = "./lora_weights/"

# Ensure directories exist
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(LORA_WEIGHTS_PATH, exist_ok=True)


class LoRALayer(nn.Module):
    def __init__(self, rank=4):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.lora_down = None
        self.lora_up = None
        self.projection = None
        self.initialized = False

    def initialize_layers(self, in_features, out_features, device="cpu"):
        """Dynamically initialize LoRA layers based on input features."""
        self.lora_down = nn.Linear(in_features, self.rank, bias=False).to(device)
        self.lora_up = nn.Linear(self.rank, out_features, bias=False).to(device)

        init.xavier_uniform_(self.lora_down.weight)
        init.xavier_uniform_(self.lora_up.weight)
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features, bias=False).to(device)
            init.xavier_uniform_(self.projection.weight)

    def forward(self, x):
        # Initialize layers dynamically based on input shape if not already initialized
        if not self.initialized:
            in_features = x.shape[-1]
            out_features = in_features  # Assuming the output shape is the same initially
            self.initialize_layers(in_features, out_features, x.device)
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

# Function to apply LoRA to UNet
def apply_lora_to_unet(unet, rank=8):
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
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,  # Ensure the model uses float32

    ).to(device)
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

def debug_lora_weights(state_dict):
    for layer_name, layer_params in state_dict.items():
        print(f"Layer: {layer_name}")
        if isinstance(layer_params, dict):
            for param_name, param in layer_params.items():
                print(f"  Param: {param_name}, Shape: {param.shape}")
        else:
            print(f"  Shape: {layer_params.shape}")

def gradient_descent(pipeline, image_tensor, optimizer, step):
    with torch.no_grad():
        image_latents = pipeline.vae.encode(image_tensor).latent_dist.sample().to(device)
        image_latents = image_latents * pipeline.vae.config.scaling_factor

    image_latents = image_latents.to(device).to(torch.float32)
    noise = torch.randn_like(image_latents, device=device)
    noise = noise.to(torch.float32)
    timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (1,), device=device).long()
    noisy_image = pipeline.scheduler.add_noise(image_latents, noise, timesteps)

    # Tokenize and encode text
    text_inputs = pipeline.tokenizer("An image of Divyang", return_tensors="pt").to(device)
    encoder_hidden_states = pipeline.text_encoder(text_inputs.input_ids).last_hidden_state.to(device)

    output = pipeline.unet(noisy_image, timesteps, encoder_hidden_states=encoder_hidden_states).sample

    loss = torch.nn.functional.mse_loss(output, noise)
    before_training = {}
    for name, param in pipeline.unet.named_parameters():
        if param.requires_grad:
            before_training[name] = param.clone().detach()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(pipeline.unet.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    for name, param in pipeline.unet.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f"{name}_grad", param.grad, global_step=step)

    return loss


# Fine-tune model with DreamBooth and LoRA
def fine_tune_with_dreambooth(pipeline, images, num_steps=15):
    dummy_input = preprocess_image(images[0]).to(device).to(torch.float32)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pipeline.unet.parameters()), lr=1e-5)
    gradient_descent(pipeline, dummy_input, optimizer, 0)

    print("\n=== Fine-Tuning with DreamBooth ===")
    # Freeze all layers
    for param in pipeline.unet.parameters():
        param.requires_grad = False

    # Unfreeze LoRA layers
    for name, module in pipeline.unet.named_modules():
        if isinstance(module, LoRALayer):
            for name, param in module.named_parameters():
                param.requires_grad = True

    # for name, param in pipeline.unet.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable parameter: {name}, Shape: {param.shape}")

    pipeline.unet.train()
    augmentation_transform_1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
        transforms.Resize((512, 512)),  # Resize to desired input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.5], [0.5])
    ])

    augmentation_transform_2 = transforms.Compose([
        transforms.RandomVerticalFlip(),  # Randomly flip vertically
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformations
        transforms.Resize((512, 512)),  # Resize to desired input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.5], [0.5])
    ])

    augmentation_transform_3 = transforms.Compose([
        transforms.RandomRotation(30),  # Randomly rotate images by 30 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformations
        transforms.Resize((512, 512)),  # Resize to desired input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.5], [0.5])
    ])

    for step in range(num_steps):
        loss_sum = 0
        for im in images:
            image_tensor = preprocess_image(im).to(device).to(torch.float32) # Move to GPU
            image_tensor_1 = augmentation_transform_1(im).unsqueeze(0).to(device).to(torch.float32)
            # image_tensor_2 = augmentation_transform_2(im).unsqueeze(0).to(device).to(torch.float32)
            # image_tensor_3 = augmentation_transform_3(im).unsqueeze(0).to(device).to(torch.float32)
            # augmented_images = [image_tensor, image_tensor_1, image_tensor_2, image_tensor_3]
            augmented_images = [image_tensor, image_tensor_1]
            for image_tensor in augmented_images:
                loss = gradient_descent(pipeline, image_tensor, optimizer, step)
                loss_sum += loss.item()
                print(f"Step [{step}/{num_steps}], Loss: {loss.item()}")

        # Calculate average loss
        avg_loss = loss_sum / len(images)
        print(f"Step [{step}/{num_steps}], Average Loss: {avg_loss}")
        generated_image = generate_image(pipeline, "An image of Divyang sitting on the beach", guidance_scale=10)

        # Save generated image
        generated_image_path = os.path.join(GENERATED_IMAGES_PATH, f"generated_image_{step}.png")
        generated_image.save(generated_image_path)

    # Save the LoRA weights
    lora_weights_path = os.path.join(LORA_WEIGHTS_PATH, "fine_tuned_lora.pt")
    torch.save(
        {name: module.state_dict() for name, module in pipeline.unet.named_modules() if isinstance(module, LoRALayer)},
        lora_weights_path
    )
    print(f"Fine-tuning complete, LoRA weights saved at {lora_weights_path}.")


# Streamlit Interface
st.title("DreamBooth Fine-Tuning with LoRA")
device = "cpu" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
writer = SummaryWriter()
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

    # image = generate_image(pipeline, "A warrior")
    # Display generated image
    # st.image(image, caption="Generated Image", use_column_width=True)


else:
    st.warning("Please upload exactly 10 images.")
