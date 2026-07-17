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

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install Dependencies

Install the packages used by the project.

```bash
pip install -r requirements.txt
```

If you need a CUDA-specific PyTorch build, install the wheel that matches your system before the other packages.

### 3. Configure Parameters and Paths

Change the configurations as you need from `src/butterfly_dit/config.py`


### 4. Prepare the Embeddings and Prompts

The repository uses cached latents and prompt embeddings under `artifacts/cache/`.

Run latent generation first:

```bash
python scripts/data/prepare_latent_embedding.py
```

Then build the combined latent/text cache:

```bash
python scripts/data/prepare_dataset.py
```

### 5. Train the model

```bash
python scripts/train.py
```

Training checkpoints are written to `artifacts/checkpoints/`, and sample images are written to `artifacts/samples/`.

### 6. Run offline inference

```bash
python scripts/inference.py
```

This writes a generated image to the artifacts area configured by the scripts.

### 7. Start the API server

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

The main settings you are most likely to change in `src/butterfly_dit/config.py` file are:

- `HYPERPARAMETERS` for the DiT Model 
- `ARTIFACTS_DIR` for the root output folder
- `HF_CACHE_DIR` for Hugging Face downloads
- `CHECKPOINT_PATH` for the model checkpoint used by inference and the API and `START_EPOCH` if starting training from that checkpoint
- `DATASET_NAME` if you want a different dataset source
- `PRE-TRAINED MODEL NAMES` if you want to use different pre-trained models
- `EPOCHS`, `SAVE_EVERY`, and `LEARNING_RATE` for training behavior

## Expected Workflow

1. Change the configuration as needed.
2. Prepare latent embeddings.
3. Generate prompts and the combined latent+prompt embeddings cache.
4. Train the model.
5. Run inference or start the API using the resulting checkpoint.

## Notes

Generated files should stay out of the repository root. The `artifacts/` directory is the canonical place for outputs, caches, and checkpoints.
