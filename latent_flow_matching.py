import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image
from datasets import Image, load_dataset
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
import os
import copy

# Hyperparametrs


device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 512
batch_size = 8
epochs = 200
save_every = 20
learning_rate = 1e-4
latent_batch_size = 32
latent_dim = 4
latent_size = image_size // 8 # 64
sequence_length = latent_size * latent_size # 64*64 tokens
Hidden_Size = 256
Num_Layers = 12
Num_Heads = 16
Patch_Size = 2
Grid_Size = 4
# print(f"Is CUDA available? {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

# transform = transforms.Compose([
#     # 1. Resize the shortest edge to 512, keeping the original aspect ratio
#     transforms.Resize(image_size), 
    
#     # 2. Chop a perfect 512x512 square out of the center
#     transforms.CenterCrop(image_size), 
    
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# def preprocess(examples):
#     return {"images": [transform(image.convert("RGB")) for image in examples['image']]}

# dataset.set_transform(preprocess)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# print(f"Dataset loaded. Batches per epoch: {len(dataloader)}")



# Load the VAE and move to GPU
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

# Freeze the VAE to save memory and prevent accidental training
vae.eval()
vae.requires_grad_(False)

latent_cache_path = "cached_smithsonian_latents.pt"

# --- The Caching Logic ---
if os.path.exists(latent_cache_path):
    print(f"⚡ Found cached latents at {latent_cache_path}! Loading directly into RAM...")
    # This takes seconds instead of minutes!
    all_latents_tensor = torch.load(latent_cache_path)

else:
    print("⏳ No cache found. Downloading dataset and encoding via VAE...")
    
    # 1. Load Dataset & VAE (Only happens if cache is missing)
    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    # ... [Load your VAE here] ...
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    def preprocess(examples):
        processed = []
        for img in examples['image']:
            if isinstance(img, str):
                img = Image.open(img)
            processed.append(transform(img.convert("RGB")))
        return {"images": processed}

    dataset.set_transform(preprocess)
    temp_dataloader = DataLoader(dataset, batch_size=batch_size)
    
    scale_factor = vae.config.scaling_factor
    all_latents = []
    
    # 2. Run the VAE Encoding
    vae.eval()
    with torch.no_grad():
        for batch in tqdm(temp_dataloader, desc="Encoding Pixels to Latents"):
            pixel_images = batch["images"].to(device)
            latents = vae.encode(pixel_images).latent_dist.sample()
            latents = latents * scale_factor
            all_latents.append(latents.cpu())

    all_latents_tensor = torch.cat(all_latents, dim=0)
    
    # 3. SAVE TO DISK FOR NEXT TIME
    torch.save(all_latents_tensor, latent_cache_path)
    print(f"✅ Saved latents to {latent_cache_path}. Future runs will be instant!")

# --- Proceed to Training ---
# Create your final lightning-fast dataloader
latent_dataset = TensorDataset(all_latents_tensor)
latent_dataloader = DataLoader(latent_dataset, batch_size=latent_batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

print("Ready to train the DiT!")


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

import torch.nn as nn

class CustomDiT(nn.Module):
    def __init__(self, latent_dim=4, hidden_size=256, num_layers=6, num_heads=8, patch_size=4, grid_size=16):
        super().__init__()
        
        self.patch_size = patch_size
        self.grid_size = grid_size
        seq_len = grid_size * grid_size  # 16 * 16 = 256 tokens!
        
        # 1. Patch Embedding (Using a Convolutional stride to extract 4x4 blocks)
        self.patch_embed = nn.Conv2d(
            in_channels=latent_dim, 
            out_channels=hidden_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 2. Positional Embedding (Now matched to the new 256 token length)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_size) * 0.02)
        
        # 3. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # 4. Standard Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        # 5. Output Projection
        self.norm_out = nn.LayerNorm(hidden_size)
        
        # We project each hidden token back into 64 elements (4 channels * 4 height * 4 width)
        self.proj_out = nn.Linear(hidden_size, latent_dim * patch_size * patch_size)

    def forward(self, x, t):
        # Input: (Batch, 4 channels, 64 height, 64 width)
        B, C, H, W = x.shape
        
        # 1. Patchify: (Batch, 4, 64, 64) -> (Batch, hidden_size, 16, 16)
        x = self.patch_embed(x)
        
        # 2. Flatten spatial dimensions: (Batch, hidden_size, 256) -> (Batch, 256, hidden_size)
        x = x.flatten(2).permute(0, 2, 1)
        
        # 3. Add Positional and Time Embeddings
        x = x + self.pos_embed
        t_emb = self.time_mlp(t).unsqueeze(1)
        x = x + t_emb
        
        # 4. Transformer Attention Math
        x = self.transformer(x)
        
        # 5. Un-patchify back to original dimensions
        x = self.norm_out(x)
        x = self.proj_out(x) # Shape becomes: (Batch, 256, 64)
        
        # Reshape the flat 64 elements back into (4 channels, 4 height, 4 width)
        # and map the 256 tokens back into a 16x16 grid
        x = x.view(B, self.grid_size, self.grid_size, C, self.patch_size, self.patch_size)
        
        # Rearrange the dimensions to logically reconstruct the image:
        # (Batch, Channels, GridHeight, PatchHeight, GridWidth, PatchWidth)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        
        # Melt the grids and patches together: (Batch, 4 channels, 64 height, 64 width)
        x = x.view(B, C, self.grid_size * self.patch_size, self.grid_size * self.patch_size)
        
        return x

@torch.no_grad()
def generate_latent_samples(model, vae, device, num_samples=4, steps=20, noisy_samples=None):
    model.eval()
    
    # 1. Start with pure latent noise at the new 64x64 size!
    if noisy_samples is not None:
        x = noisy_samples.clone()
    else:
        x = torch.randn(num_samples, 4, 64, 64, device=device)

    dt = 1.0 / steps
    
    for i in range(steps):
        t_val = i / steps
        t_tensor = torch.full((num_samples,), t_val, device=device, dtype=torch.float32)
        
        velocity = model(x, t_tensor)
        x = x + velocity * dt
        
    x = x / vae.config.scaling_factor
    images = vae.decode(x).sample
    
    model.train()
    return images


model = CustomDiT(
    hidden_size=Hidden_Size,
    num_layers=Num_Layers,
    num_heads=Num_Heads,
    patch_size=Patch_Size,
    grid_size=Grid_Size
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

load_path = "/mnt/lab/asif/Introductory Task/latent_flow_matching_output/dit_best_epoch_968.pth"
start_epoch = 0 # Default starting point

if os.path.exists(load_path):
    print(f"Resuming! Loading weights from {load_path} into existing model...")
    # This single line pours the saved weights directly into your 'model' variable
    model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
    start_epoch = 980
else:
    print("No save file found. Starting completely fresh!")


total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params:,}")

output_dir = "/mnt/lab/asif/Introductory Task/latent_flow_matching_output"
os.makedirs(output_dir, exist_ok=True)

best_window_loss = float('inf')
best_window_weights = None
best_window_epoch = -1

noisy_samples = torch.randn(16, 4, 64, 64, device=device)

model.train()


for epoch in range(start_epoch, start_epoch + epochs):
    epoch_loss = 0.0
    # Notice we are now using the new latent_dataloader!
    progress_bar = tqdm(latent_dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in progress_bar:
        # 1. Grab pre-computed latents directly (no VAE needed here!)
        x1 = batch[0].to(device)
        b = x1.shape[0]
            
        # 2. Flow Matching Math
        x0 = torch.randn_like(x1)
        t = torch.rand(b, device=device)
        
        # Expand t to shape (Batch, 1, 1, 1)
        t_expand = t.view(-1, 1, 1, 1)
        
        xt = (1 - t_expand) * x0 + t_expand * x1
        target_velocity = x1 - x0
        
        # 3. Predict Velocity via DiT
        pred_velocity = model(xt, t)
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        # 4. Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    avg_loss = epoch_loss / len(latent_dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # --- Track Best Model in Window ---
    if avg_loss < best_window_loss:
        best_window_loss = avg_loss
        best_window_epoch = epoch + 1
        best_window_weights = copy.deepcopy(model.state_dict())
        
    # --- Window End: Checkpoint & Generate ---
    if (epoch + 1) % save_every == 0:
        print(f"\n--- Saving Best Model from past {save_every} Epochs ---")
        
        weight_path = os.path.join(output_dir, f"dit_best_epoch_{best_window_epoch}.pth")
        torch.save(best_window_weights, weight_path)
        
        # State Swap for Generation
        latest_weights = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_window_weights)
        
        # Generate Images (This still uses the VAE inside the function to decode)
        # Using 2 images here to save time and VRAM during mid-training checks
        samples = generate_latent_samples(model, vae, device, num_samples=16, steps=50, noisy_samples=noisy_samples)
        samples = (samples / 2 + 0.5).clamp(0, 1).cpu()
        
        image_path = os.path.join(output_dir, f"latent_generation_epoch_{best_window_epoch}.png")
        save_image(samples, image_path, nrow=4)

        image_path = os.path.join(output_dir, f"temp.png")
        save_image(samples, image_path, nrow=4)
        
        # # Plotting inline
        # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # for i, ax in enumerate(axes.flatten()):
        #     ax.imshow(samples[i].permute(1, 2, 0).numpy())
        #     ax.axis("off")
        # plt.suptitle(f"Latent DiT - Best Epoch {best_window_epoch}")
        # plt.tight_layout()
        # plt.show()
        
        # Restore latest weights
        model.load_state_dict(latest_weights)
        best_window_loss = float('inf')
        best_window_weights = None

torch.save(model.state_dict(), os.path.join(output_dir, "dit_final.pth"))
print("Latent Flow-Matching Training Complete!")
