# Butterfly DiT Image Generation Pipeline

A complete end-to-end machine learning pipeline for generating high-quality butterfly images using a Diffusion Transformer (DiT). This project covers automated dataset captioning, model training conditioning, and a high-performance FastAPI inference server.

## Overview

This project is divided into two primary components:
1. **Automated Vision-Language Data Preparation:** Using `Qwen2-VL-2B-Instruct` to generate highly specific, rigidly formatted spatial and geometric captions (e.g., *"blue wings, white spots on outer edges"*) from the Smithsonian Butterflies dataset. 
2. **FastAPI Inference Server:** A production-ready API that loads the DiT, VAE, and CLIP models into global memory on startup and serves images via a robust generation endpoint.

## Features
* **Zero-Noise Captioning:** Employs Prompt Engineering (Domain Shift technique) on Qwen2-VL to enforce strict noun-and-feature-based outputs, avoiding conversational filler that degrades CLIP embeddings.
* **Smart Memory Management:** The API uses FastAPI's lifespan/startup events to load heavy PyTorch models once, preventing VRAM overflow and minimizing response times.
* **Half-Precision Optimization:** Uses `bfloat16` and Safetensors to safely run large vision-language models alongside diffusion pipelines without crashing standard GPUs.

## Installation

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install torch transformers datasets qwen-vl-utils pillow fastapi uvicorn pydantic
```
## Data Preparation

```bash
python3 prepare_dataset.py
```

## Run the model
Change the file paths and run the following command. 

```bash
python3 prompt_conditioned_flow_matching.py
```

## Inference
Change the file paths and run the following command. 
```bash
python3 inference.py
```
