import random
import numpy as np
import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_vae(model_name, device):
    vae = AutoencoderKL.from_pretrained(model_name).to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def load_clip_text_stack(model_name, device):
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name, use_safetensors=True).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return tokenizer, text_encoder


def load_model_weights(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def encode_text_context(tokenizer, text_encoder, prompts, device):
    clip_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    return text_encoder(**clip_inputs).last_hidden_state
