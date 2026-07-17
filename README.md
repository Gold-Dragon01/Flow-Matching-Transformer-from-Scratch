# Butterfly DiT Image Generation Pipeline

This repository implements a Diffusion Transformer pipeline from scratch for butterfly image generation. It includes latent embedding generation and cache the embeddings, prompt generation for butterfly images and make combined embeddings for image and their corresponding prompts, training, offline inference, and a FastAPI inference server.

The project now follows a standard `src`-based Python layout:

- `src/butterfly_dit/` contains reusable library code
- `scripts/` contains runnable workflows
- `artifacts/` stores caches, checkpoints, and generated outputs

## Quick Start

### 1. Create an environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 2. Install Dependencies

Install the packages used by the project.

```bash
pip install torch torchvision transformers datasets diffusers fastapi uvicorn pydantic pillow tqdm qwen-vl-utils numpy safetensors
```

If you need a CUDA-specific PyTorch build, install the wheel that matches your system before the other packages.

### 3. Prepare the Embeddings and Prompts

The repository uses cached latents and prompt embeddings under `artifacts/cache/`.

Run latent generation first:

```bash
python scripts/data/prepare_latent_embedding.py
```

Then build the combined latent/text cache:

```bash
python scripts/data/prepare_dataset.py
```

### 4. Train the model

```bash
python scripts/train.py
```

Training checkpoints are written to `artifacts/checkpoints/`, and sample images are written to `artifacts/samples/`.

### 5. Run offline inference

```bash
python scripts/inference.py
```

This writes a generated image to the artifacts area configured by the scripts.

### 6. Start the API server

```bash
uvicorn scripts.api:app --host 0.0.0.0 --port 8000
```

## Artifact Locations

The default paths are defined in `src/butterfly_dit/config.py`.

- `artifacts/cache/` for cached latents and prompt CSV files
- `artifacts/checkpoints/` for model weights
- `artifacts/samples/` for generated training previews
- `artifacts/hf_cache/` for Hugging Face downloads used by the data-preparation workflow

If you want to relocate outputs, change `ARTIFACTS_DIR` in `src/butterfly_dit/config.py`. The derived cache and output paths will follow automatically.

## Project Structure

- `src/butterfly_dit/config.py` centralizes paths and model settings
- `src/butterfly_dit/modeling.py` defines the DiT model and sampling loop
- `src/butterfly_dit/runtime.py` holds reusable loading and encoding helpers
- `scripts/train.py` runs training
- `scripts/inference.py` runs batch inference
- `scripts/api.py` serves the API
- `scripts/data/prepare_latent_embedding.py` creates the latent cache
- `scripts/data/prepare_dataset.py` creates the prompt and combined training cache

## Configuration

The main settings you are most likely to change are:

- `ARTIFACTS_DIR` for the root output folder
- `HF_CACHE_DIR` for Hugging Face downloads
- `CHECKPOINT_PATH` for the model checkpoint used by inference and the API
- `DATASET_NAME` if you want a different dataset source
- `EPOCHS`, `SAVE_EVERY`, and `LEARNING_RATE` for training behavior

## Expected Workflow

1. Prepare latent caches.
2. Generate prompts and the combined training cache.
3. Train the model.
4. Run inference or start the API using the resulting checkpoint.

## Notes

Generated files should stay out of the repository root. The `artifacts/` directory is the canonical place for outputs, caches, and checkpoints.
