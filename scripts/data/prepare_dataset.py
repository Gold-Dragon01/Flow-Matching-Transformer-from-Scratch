import csv
import os
import random

import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.butterfly_dit.config import (
    CLIP_MODEL_NAME,
    COMBINED_CACHE_PATH,
    DATASET_NAME,
    LATENT_CACHE_PATH,
    PROMPT_CACHE_PATH,
    PROMPT_DROPOUT_RATE,
    QWEN_INSTRUCTION_PROMPT,
    QWEN_MAX_NEW_TOKENS,
    QWEN_MODEL_NAME,
    QWEN_SYSTEM_PROMPT,
    HF_CACHE_DIR,
)
from src.butterfly_dit.runtime import encode_text_context, get_device, load_clip_text_stack, set_global_seed


os.environ["HF_HOME"] = HF_CACHE_DIR
set_global_seed(24)

device = get_device()
print(f"Loading models to {device}...")

print(f"Loading {QWEN_MODEL_NAME}...")
qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
qwen_model.eval()

print("Loading CLIP...")
clip_tokenizer, clip_model = load_clip_text_stack(CLIP_MODEL_NAME, device)

print(f"Loading existing latents from {LATENT_CACHE_PATH}...")
pre_saved_latents = torch.load(LATENT_CACHE_PATH)

print(f"Downloading/Loading {DATASET_NAME} for raw images...")
dataset = load_dataset(DATASET_NAME, split="train")

all_text_embeddings = []
csv_data = []

print("Starting the text encoding process...")

with torch.no_grad():
    for index, row in enumerate(tqdm(dataset, desc="Writing Prompts")):
        raw_image = row["image"].convert("RGB")

        if random.random() < PROMPT_DROPOUT_RATE:
            final_text = ""
        else:
            messages = [
                {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},
                        {"type": "text", "text": QWEN_INSTRUCTION_PROMPT},
                    ],
                },
            ]

            text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)

            inputs = qwen_processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            generated_ids = qwen_model.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            final_text = qwen_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            final_text = final_text.replace("\n", " ").replace("- ", "")

        csv_data.append({"image_index": index, "prompt": final_text})

        text_embedding = encode_text_context(clip_tokenizer, clip_model, [final_text], device)
        all_text_embeddings.append(text_embedding.cpu())

print(f"Saving {len(csv_data)} prompts to {PROMPT_CACHE_PATH}...")
with open(PROMPT_CACHE_PATH, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["image_index", "prompt"])
    writer.writeheader()
    writer.writerows(csv_data)

print("Stacking text tensors...")
final_text_tensor = torch.cat(all_text_embeddings, dim=0)

assert len(final_text_tensor) == len(pre_saved_latents), f"Mismatch! {len(final_text_tensor)} texts vs {len(pre_saved_latents)} latents."

cache_dict = {
    "latents": pre_saved_latents,
    "text_embeddings": final_text_tensor,
}

print(f"Saving final combined cache to {COMBINED_CACHE_PATH}...")
torch.save(cache_dict, COMBINED_CACHE_PATH)

print("✅ Success! Your dataset is fully prepared using Qwen2-VL!")
