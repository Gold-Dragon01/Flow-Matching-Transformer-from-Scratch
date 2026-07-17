import os

import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image
from transformers import CLIPTextModel, CLIPTokenizer

from src.butterfly_dit.config import CHECKPOINT_PATH, CLIP_MODEL_NAME, GRID_SIZE, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, PATCH_SIZE, SAMPLE_IMAGE_DIRECTORY, VAE_MODEL_NAME
from src.butterfly_dit.modeling import CustomDiT, generate_samples


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(SAMPLE_IMAGE_DIRECTORY, exist_ok=True)
    image_path = os.path.join(SAMPLE_IMAGE_DIRECTORY, "inference.png")

    vae = AutoencoderKL.from_pretrained(VAE_MODEL_NAME).to(device)
    vae.eval()
    vae.requires_grad_(False)

    clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to(device)
    clip_model.eval()

    model = CustomDiT(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        grid_size=GRID_SIZE,
    ).to(device)

    print(f"Resuming! Loading weights from {CHECKPOINT_PATH} into existing model...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))

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
        "brown gold broad butterfly spots",
    ]

    print("Generating fixed evaluation contexts...")
    with torch.no_grad():
        clip_inputs = clip_tokenizer(
            eval_prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to(device)
        fixed_text_context = clip_model(**clip_inputs).last_hidden_state

    generated_images, _ = generate_samples(
        model=model,
        vae=vae,
        device=device,
        text_context=fixed_text_context,
        num_samples=16,
        steps=80,
    )

    generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
    save_image(generated_images, image_path, nrow=4)


if __name__ == "__main__":
    main()
