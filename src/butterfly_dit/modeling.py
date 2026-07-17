import math
import random

import torch
import torch.nn as nn

from src.butterfly_dit.config import  LATENT_DIM, LATENT_SIZE


def modulate(x, shift, scale):
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


class PromptConditionedDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, text_context):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)

        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_mod_ca = modulate(self.norm_cross(x), shift_ca, scale_ca)
        ca_out, _ = self.cross_attn(query=x_mod_ca, key=text_context, value=text_context, need_weights=False)
        x = x + gate_ca.unsqueeze(1) * ca_out

        x_mod_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mod_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class CustomDiT(nn.Module):
    def __init__(self, latent_dim=4, hidden_size=256, num_layers=12, num_heads=12, patch_size=4, grid_size=16):
        super().__init__()

        self.patch_size = patch_size
        self.grid_size = grid_size
        seq_len = grid_size * grid_size

        self.patch_embed = nn.Conv2d(latent_dim, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_size) * 0.02)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.blocks = nn.ModuleList(
            [PromptConditionedDiTBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )

        self.norm_out = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, latent_dim * patch_size * patch_size)

    def forward(self, x, t, text_context):
        B, C, H, W = x.shape

        x = self.patch_embed(x).flatten(2).permute(0, 2, 1)
        x = x + self.pos_embed

        t_emb = self.time_mlp(t)

        for block in self.blocks:
            x = block(x, t_emb, text_context)

        x = self.norm_out(x)
        x = self.proj_out(x)

        x = x.view(B, self.grid_size, self.grid_size, C, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, self.grid_size * self.patch_size, self.grid_size * self.patch_size)

        return x


@torch.no_grad()
def generate_samples(
    model,
    vae,
    device,
    text_context,
    num_samples=4,
    steps=20,
    noisy_samples=None,
    seed=-1,
    guidance_scale=1.0,
    null_context=None,
):
    was_training = model.training
    model.eval()

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    gen = torch.Generator(device=device).manual_seed(seed)

    if noisy_samples is not None:
        x = noisy_samples.clone()
    else:
        x = torch.randn(num_samples, LATENT_DIM, LATENT_SIZE, LATENT_SIZE, device=device, generator=gen)

    dt = 1.0 / steps

    for i in range(steps):
        t_val = i / steps
        t_tensor = torch.full((num_samples,), t_val, device=device, dtype=torch.float32)

        if guidance_scale > 1.0 and null_context is not None:
            x_double = torch.cat([x, x])
            t_double = torch.cat([t_tensor, t_tensor])
            context_double = torch.cat([null_context, text_context])

            v_pred = model(x_double, t_double, context_double)
            v_null, v_cond = v_pred.chunk(2)
            velocity = v_null + guidance_scale * (v_cond - v_null)
        else:
            velocity = model(x, t_tensor, text_context)

        x = x + velocity * dt

    x = x / vae.config.scaling_factor
    images = vae.decode(x).sample

    if was_training:
        model.train()
    else:
        model.eval()

    return images, seed
