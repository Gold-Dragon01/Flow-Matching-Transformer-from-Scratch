import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from tqdm.auto import tqdm

from src.butterfly_dit.config import CLIP_MODEL_NAME, GRID_SIZE, HIDDEN_SIZE, LATENT_BATCH_SIZE, NUM_HEADS, NUM_LAYERS, PATCH_SIZE, VAE_MODEL_NAME, EPOCHS, SAVE_EVERY, LEARNING_RATE, CHECKPOINT_PATH, START_EPOCH, COMBINED_CACHE_PATH, SAMPLE_IMAGE_DIRECTORY, MODEL_CHECKPOINT_DIRECTORY
from src.butterfly_dit.modeling import CustomDiT, generate_samples
from src.butterfly_dit.runtime import encode_text_context, get_device, load_clip_text_stack, load_model_weights, load_vae, set_global_seed

set_global_seed(24)

device = get_device()
vae = load_vae(VAE_MODEL_NAME, device)


def build_model():
    return CustomDiT(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        grid_size=GRID_SIZE,
    ).to(device)


model = build_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    print(f"Resuming! Loading weights from {CHECKPOINT_PATH} into existing model...")
    load_model_weights(model, CHECKPOINT_PATH, device)
    start_epoch = START_EPOCH
else:
    print("No saved file found. Starting completely fresh!")


total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params:,}")


print("Loading cached dataset into RAM...")
cache = torch.load(COMBINED_CACHE_PATH)

dataset = TensorDataset(cache["latents"], cache["text_embeddings"])
dataloader = DataLoader(dataset, batch_size=LATENT_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

clip_tokenizer, clip_model = load_clip_text_stack(CLIP_MODEL_NAME, device)

eval_prompts = [
    "a red butterfly",
    "an orange butterfly with black stripes",
    "a blue butterfly with yellow spots",
    "a yellow butterfly with black spots",
]

print("Generating fixed evaluation contexts...")
with torch.no_grad():
    fixed_text_context = encode_text_context(clip_tokenizer, clip_model, eval_prompts, device)
    fixed_noise = torch.randn(4, 4, 64, 64, device=device)

os.makedirs(SAMPLE_IMAGE_DIRECTORY, exist_ok=True)
os.makedirs(MODEL_CHECKPOINT_DIRECTORY, exist_ok=True)

for epoch in range(start_epoch, start_epoch + EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_latents, batch_texts in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        x_1 = batch_latents.to(device)
        text_context = batch_texts.to(device)

        batch_size_actual = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = torch.rand((batch_size_actual,), device=device)

        t_expanded = t.view(batch_size_actual, 1, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        target_velocity = x_1 - x_0

        pred_velocity = model(x_t, t, text_context)
        loss = F.mse_loss(pred_velocity, target_velocity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}/{start_epoch + EPOCHS} | Average Loss: {avg_loss:.4f}")

    if (epoch + 1) % SAVE_EVERY == 0:
        print(f"Saving checkpoint and generating samples for Epoch {epoch + 1}...")

        save_path = os.path.join(MODEL_CHECKPOINT_DIRECTORY, f"dit_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)

        generated_images, _ = generate_samples(
            model=model,
            vae=vae,
            device=device,
            text_context=fixed_text_context,
            num_samples=4,
            steps=50,
            noisy_samples=fixed_noise,
            seed=42,
        )

        generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
        image_path = os.path.join(SAMPLE_IMAGE_DIRECTORY, f"epoch_{epoch + 1}_samples.png")
        save_image(generated_images, image_path, nrow=2)

        image_path = os.path.join(SAMPLE_IMAGE_DIRECTORY, "temp.png")
        save_image(generated_images, image_path, nrow=2)

        print("✅ Checkpoint and Samples saved!")

torch.save(model.state_dict(), os.path.join(MODEL_CHECKPOINT_DIRECTORY, "dit_final.pth"))
print("Latent Flow-Matching Training Complete!")
