from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
import torch
from torchvision.transforms import ToPILImage
import io
import base64
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel


from inference import (
    CustomDiT, 
    generate_latent_samples, 
    Hidden_Size, 
    Num_Layers, 
    Num_Heads, 
    Patch_Size, 
    Grid_Size, 
    load_path
)

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
    vae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae_model.eval()
    vae_model.requires_grad_(False)

    print("Loading CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
    text_encoder.eval()

    print("Loading Custom DiT...")
    dit_model = CustomDiT(
        hidden_size=Hidden_Size,
        num_layers=Num_Layers,
        num_heads=Num_Heads,
        patch_size=Patch_Size,
        grid_size=Grid_Size
    ).to(device)
    dit_model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
    dit_model.eval()
    
    print("API is ready to serve!")
    yield
    
    # Cleanup memory on shutdown
    dit_model = vae_model = tokenizer = text_encoder = None


app = FastAPI(lifespan=lifespan)
 


@app.get("/health")
async def health_check():
    if dit_model is not None and torch.cuda.is_available():
        return {"status": "running"}
    else:
        return {"status": "not running"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_image(req: GenerateRequest):
    if dit_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    prompt_text = req.prompt if req.prompt else ""

    with torch.no_grad():
        # 1. Convert text prompt to CLIP embeddings
        clip_inputs = tokenizer(
            [prompt_text], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to(device)
        text_context = text_encoder(**clip_inputs).last_hidden_state 

        # 2. Run your DiT inference loop
        # Setting num_samples=1 because the API expects a single image back
        generated_images, actual_seed = generate_latent_samples(
            model=dit_model,
            vae=vae_model,
            device=device,
            text_context=text_context,
            num_samples=1, 
            steps=req.num_steps,
            seed=req.seed
        )

        # 3. Process the output tensor into a PIL Image
        # Grab the first image in the batch and un-normalize it
        img_tensor = (generated_images[0] / 2 + 0.5).clamp(0, 1)
        
        # Convert PyTorch tensor to PIL Image
        to_pil = ToPILImage()
        pil_image = to_pil(img_tensor)

        # 4. Convert PIL Image to Base64 string for the API response
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        base64_encoded = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return GenerateResponse(
        image=base64_encoded,
        prompt=req.prompt,
        seed=actual_seed,
        num_steps=req.num_steps
    )


@app.post("/generate-direct")
async def generate_image_direct(req: GenerateRequest):
    if dit_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    prompt_text = req.prompt if req.prompt else ""

    with torch.no_grad():
        # 1. Convert text prompt to CLIP embeddings
        clip_inputs = tokenizer(
            [prompt_text], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to(device)
        text_context = text_encoder(**clip_inputs).last_hidden_state 

        # 2. Run your DiT inference loop
        # Setting num_samples=1 because the API expects a single image back
        generated_images, actual_seed = generate_latent_samples(
            model=dit_model,
            vae=vae_model,
            device=device,
            text_context=text_context,
            num_samples=1, 
            steps=req.num_steps,
            seed=req.seed
        )

        # 3. Process the output tensor into a PIL Image
        # Grab the first image in the batch and un-normalize it
        img_tensor = (generated_images[0] / 2 + 0.5).clamp(0, 1)
        
        # Convert PyTorch tensor to PIL Image
        to_pil = ToPILImage()
        pil_image = to_pil(img_tensor)

        # 4. Convert PIL Image to Base64 string for the API response
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        #base64_encoded = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")
