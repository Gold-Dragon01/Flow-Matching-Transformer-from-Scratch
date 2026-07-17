import base64
import io
from contextlib import asynccontextmanager

import torch
from diffusers import AutoencoderKL
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from torchvision.transforms import ToPILImage
from transformers import CLIPTextModel, CLIPTokenizer

from src.butterfly_dit.config import CHECKPOINT_PATH, CLIP_MODEL_NAME, GRID_SIZE, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, PATCH_SIZE, VAE_MODEL_NAME
from src.butterfly_dit.modeling import CustomDiT, generate_samples


class GenerateRequest(BaseModel):
    prompt: str | None = None
    seed: int = -1
    num_steps: int = 30


class GenerateResponse(BaseModel):
    image: str
    prompt: str | None = None
    seed: int
    num_steps: int


device = "cuda" if torch.cuda.is_available() else "cpu"
dit_model = None
vae_model = None
tokenizer = None
text_encoder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global dit_model, vae_model, tokenizer, text_encoder

    print("Loading VAE...")
    vae_model = AutoencoderKL.from_pretrained(VAE_MODEL_NAME).to(device)
    vae_model.eval()
    vae_model.requires_grad_(False)

    print("Loading CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    text_encoder = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to(device)
    text_encoder.eval()

    print("Loading Custom DiT...")
    dit_model = CustomDiT(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        grid_size=GRID_SIZE,
    ).to(device)
    dit_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    dit_model.eval()

    print("API is ready to serve!")
    yield

    dit_model = vae_model = tokenizer = text_encoder = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    if dit_model is not None:
        return {"status": "running"}
    else:
        return {"status": "not running"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_image(req: GenerateRequest):
    if dit_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    prompt_text = req.prompt if req.prompt else ""

    with torch.no_grad():
        clip_inputs = tokenizer(
            [prompt_text], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to(device)
        text_context = text_encoder(**clip_inputs).last_hidden_state

        generated_images, actual_seed = generate_samples(
            model=dit_model,
            vae=vae_model,
            device=device,
            text_context=text_context,
            num_samples=1,
            steps=req.num_steps,
            seed=req.seed,
        )

        img_tensor = (generated_images[0] / 2 + 0.5).clamp(0, 1)
        to_pil = ToPILImage()
        pil_image = to_pil(img_tensor)

        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")
        base64_encoded = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    return GenerateResponse(
        image=base64_encoded,
        prompt=req.prompt,
        seed=actual_seed,
        num_steps=req.num_steps,
    )


@app.post("/generate-direct")
async def generate_image_direct(req: GenerateRequest):
    if dit_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    prompt_text = req.prompt if req.prompt else ""

    with torch.no_grad():
        clip_inputs = tokenizer(
            [prompt_text], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to(device)
        text_context = text_encoder(**clip_inputs).last_hidden_state

        generated_images, actual_seed = generate_samples(
            model=dit_model,
            vae=vae_model,
            device=device,
            text_context=text_context,
            num_samples=1,
            steps=req.num_steps,
            seed=req.seed,
        )

        img_tensor = (generated_images[0] / 2 + 0.5).clamp(0, 1)
        to_pil = ToPILImage()
        pil_image = to_pil(img_tensor)

        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")
