import torch
from diffusers import AutoencoderKL
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision.utils import save_image
import math
import random



image_size = 512
latent_size = image_size // 8 # 64
sequence_length = latent_size * latent_size # 64*64 tokens
Hidden_Size = 512
Num_Layers = 8
Num_Heads = 16
Patch_Size = 2
Grid_Size = latent_size // Patch_Size # 16
device = "cuda" if torch.cuda.is_available() else "cpu"
load_path = "/mnt/lab/asif/Prompt/model_checkpoints_v2/dit_epoch_520.pt"


vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

# Freeze the VAE to save memory and prevent accidental training
vae.eval()
vae.requires_grad_(False)

def modulate(x, shift, scale):
    # Unsqueeze adds the sequence length dimension so the math broadcasts perfectly
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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


# 2. The New Cross-Attention Block
class PromptConditionedDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        
        # Self Attention (Image looks at Image)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # Cross Attention (Image looks at Text) <-- NEW!
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # AdaLN Modulator (Now generates 9 chunks instead of 6!)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        
        # Zero-init magic for stability
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, text_context): 
        # Chop the time embedding into 9 pieces
        (shift_msa, scale_msa, gate_msa, 
         shift_ca, scale_ca, gate_ca, 
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(9, dim=1)
        
        # Self-Attention Path
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out 
        
        # Cross-Attention Path (Reads the text_context)
        x_mod_ca = modulate(self.norm_cross(x), shift_ca, scale_ca)
        ca_out, _ = self.cross_attn(query=x_mod_ca, key=text_context, value=text_context, need_weights=False)
        x = x + gate_ca.unsqueeze(1) * ca_out
        
        # MLP Path
        x_mod_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mod_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

# 3. The Updated Main Class
class CustomDiT(nn.Module):
    def __init__(self, latent_dim=4, hidden_size=256, num_layers=12, num_heads=12, patch_size=4, grid_size=16):
        super().__init__()
        
        self.patch_size = patch_size
        self.grid_size = grid_size
        seq_len = grid_size * grid_size 
        
        self.patch_embed = nn.Conv2d(latent_dim, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_size) * 0.02)
        
        self.time_mlp = nn.Sequential(
            # Assuming you have your SinusoidalPositionEmbeddings defined above
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size) 
        )
        
        # --- SWAPPED TO NEW BLOCKS ---
        self.blocks = nn.ModuleList([
            PromptConditionedDiTBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        
        self.norm_out = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, latent_dim * patch_size * patch_size)

    # --- ADDED text_context HERE ---
    def forward(self, x, t, text_context):
        B, C, H, W = x.shape
        
        x = self.patch_embed(x).flatten(2).permute(0, 2, 1)
        x = x + self.pos_embed
        
        t_emb = self.time_mlp(t)
        
        # Pass both time (t_emb) and text (text_context) to every block
        for block in self.blocks:
            x = block(x, t_emb, text_context)
            
        x = self.norm_out(x)
        x = self.proj_out(x) 
        
        x = x.view(B, self.grid_size, self.grid_size, C, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, self.grid_size * self.patch_size, self.grid_size * self.patch_size)
        
        return x

@torch.no_grad()
def generate_latent_samples(model, vae, device, text_context, num_samples=4, steps=20, noisy_samples=None, seed=-1, guidance_scale=1.0, null_context=None):
    model.eval()
    
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    gen = torch.Generator(device=device).manual_seed(seed)

    if noisy_samples is not None:
        x = noisy_samples.clone()
    else:
        x = torch.randn(num_samples, 4, 64, 64, device=device, generator=gen)

    dt = 1.0 / steps
    
    for i in range(steps):
        t_val = i / steps
        t_tensor = torch.full((num_samples,), t_val, device=device, dtype=torch.float32)
        
        # --- CFG Logic ---
        if guidance_scale > 1.0 and null_context is not None:
            # Duplicate inputs for simultaneous conditional & unconditional processing
            x_double = torch.cat([x, x])
            t_double = torch.cat([t_tensor, t_tensor])
            context_double = torch.cat([null_context, text_context])
            
            # Predict both velocities at once
            v_pred = model(x_double, t_double, context_double)
            v_null, v_cond = v_pred.chunk(2)
            
            # CFG Math: Extrapolate away from the unconditional prediction
            velocity = v_null + guidance_scale * (v_cond - v_null)
        else:
            # Standard single-pass generation
            velocity = model(x, t_tensor, text_context)
        # -----------------
            
        x = x + velocity * dt
        
    x = x / vae.config.scaling_factor
    images = vae.decode(x).sample
    
    model.train()
    return images, seed


model = CustomDiT(
    hidden_size=Hidden_Size,
    num_layers=Num_Layers,
    num_heads=Num_Heads,
    patch_size=Patch_Size,
    grid_size=Grid_Size
).to(device)

#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)


print(f"Resuming! Loading weights from {load_path} into existing model...")
# This single line pours the saved weights directly into your 'model' variable
model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))


clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
clip_model.eval()

# --- 2. Define Fixed Evaluation Prompts ---
eval_prompts = [
    "brown yellow broad butterfly spots",
    "black green broad butterfly stripes",
    "a blue butterfly",
    "a butterfly with black wings and white stripes",
    "black yellow pointed butterfly stripes",
    "black blue broad butterfly spots",
    "black cream broad butterfly patches",
    "black blue broad butterfly patches",
    "black white pointed butterfly stripes",
    "black red broad butterfly bands",
    "black red broad butterfly patches",
    "black white broad butterfly spots",
    "white yellow broad butterfly spots",
    "black red broad butterfly bands",
    "black red broad butterfly stripes",
    "brown gold broad butterfly spots"  # The Null Token (Unconditional generation test!)
]

print("Generating fixed evaluation contexts...")
with torch.no_grad():
    clip_inputs = clip_tokenizer(
        eval_prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    ).to(device)
    # This matrix stays locked in memory for the whole training run
    fixed_text_context = clip_model(**clip_inputs).last_hidden_state 
    
    # Generate fixed noise (4 images, 4 channels, 64x64)

generated_images, _ = generate_latent_samples(model=model,vae=vae,device=device,text_context=fixed_text_context,num_samples=16,steps=80)
        
# Normalize images from [-1, 1] to [0, 1] for saving
generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
            
image_path = f"training_progress_v2/temp2.png"
save_image(generated_images, image_path, nrow=4)
