import os

# Parameters for the dataset and latent space embeddings

IMAGE_SIZE = 512
LATENT_DIM = 4
LATENT_SIZE = IMAGE_SIZE // 8
SEQUENCE_LENGTH = LATENT_SIZE * LATENT_SIZE
LATENT_BATCH_SIZE = 32
IMAGE_BATCH_SIZE = 8

# Hyperparameters for the DiT model

HIDDEN_SIZE = 512
NUM_LAYERS = 8
NUM_HEADS = 16
PATCH_SIZE = 2
GRID_SIZE = LATENT_SIZE // PATCH_SIZE

# Hyperparameters for training and evaluation

EPOCHS = 500
SAVE_EVERY = 20
LEARNING_RATE = 1e-4

# Output Directories and File Paths

ARTIFACTS_DIR = "artifacts"
CACHE_DIR = os.path.join(ARTIFACTS_DIR, "cache")
SAMPLE_IMAGE_DIRECTORY = os.path.join(ARTIFACTS_DIR, "samples")
MODEL_CHECKPOINT_DIRECTORY = os.path.join(ARTIFACTS_DIR, "checkpoints")
HF_CACHE_DIR = os.path.join(ARTIFACTS_DIR, "hf_cache") 

CHECKPOINT_PATH = os.path.join(MODEL_CHECKPOINT_DIRECTORY, "dit_epoch_520.pt")
START_EPOCH = 520

LATENT_CACHE_PATH = os.path.join(CACHE_DIR, "cached_smithsonian_latents.pt")
PROMPT_CACHE_PATH = os.path.join(CACHE_DIR, "butterfly_prompts.csv")
COMBINED_CACHE_PATH = os.path.join(CACHE_DIR, "smithsonian_latents_and_prompts.pt")

# Dataset and Pre-trained Model Names

DATASET_NAME = "huggan/smithsonian_butterflies_subset"

VAE_MODEL_NAME = "stabilityai/sd-vae-ft-mse"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# Hyperparameters for Qwen2-VL model

PROMPT_DROPOUT_RATE = 0.08
QWEN_MAX_NEW_TOKENS = 25
QWEN_SYSTEM_PROMPT = (
    "You are an automated image captioner. Output a single, short phrase describing the visual features. "
    "Never use lists, bullet points, or introductory text."
)
QWEN_INSTRUCTION_PROMPT = (
    "Describe the color on the wings with patterns and spots. Be concise and tell ONLY the visual features of the butterfly."
)
