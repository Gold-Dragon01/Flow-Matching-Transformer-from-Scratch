import os
import torch
from torch.utils.data import DataLoader
from datasets import Image, load_dataset
from torchvision import transforms
from tqdm.auto import tqdm

from src.butterfly_dit.config import IMAGE_SIZE, IMAGE_BATCH_SIZE, VAE_MODEL_NAME, LATENT_CACHE_PATH, DATASET_NAME
from src.butterfly_dit.runtime import get_device, load_vae


device = get_device()

if os.path.exists(LATENT_CACHE_PATH):
    print(f"⚡ Found cached latents at {LATENT_CACHE_PATH}! Loading directly into RAM...")
    all_latents_tensor = torch.load(LATENT_CACHE_PATH)

else:
    print("⏳ No cache found. Downloading dataset and encoding via VAE...")

    dataset = load_dataset(DATASET_NAME, split="train")
    vae = load_vae(VAE_MODEL_NAME, device)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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
    temp_dataloader = DataLoader(dataset, batch_size=IMAGE_BATCH_SIZE)

    scale_factor = vae.config.scaling_factor
    all_latents = []

    vae.eval()
    with torch.no_grad():
        for batch in tqdm(temp_dataloader, desc="Encoding Pixels to Latents"):
            pixel_images = batch["images"].to(device)
            latents = vae.encode(pixel_images).latent_dist.sample()
            latents = latents * scale_factor
            all_latents.append(latents.cpu())

    all_latents_tensor = torch.cat(all_latents, dim=0)

    torch.save(all_latents_tensor, LATENT_CACHE_PATH)
    print(f"✅ Saved latents to {LATENT_CACHE_PATH}. Future runs will be instant!")
