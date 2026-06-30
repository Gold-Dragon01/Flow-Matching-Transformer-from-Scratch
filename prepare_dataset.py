import os
# This forces Hugging Face to download models to your roomy lab drive
os.environ["HF_HOME"] = "/mnt/lab/asif/hf_cache"

import csv
import torch
import random
from datasets import load_dataset
from tqdm.auto import tqdm

# Hugging Face Models
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import CLIPTokenizer, CLIPTextModel
from qwen_vl_utils import process_vision_info

# --- 1. Settings ---
device = "cuda" if torch.cuda.is_available() else "cpu"
existing_latents_path = "cached_smithsonian_latents.pt" 
new_cache_save_path = "smithsonian_latents_and_prompts.pt"
dropout_rate = 0.08 
csv_filename = "butterfly_prompts.csv"
print(f"Loading models to {device}...")

# --- 2. Load Text Models ONLY (No VAE!) ---
print("Loading Qwen2-VL-2B-Instruct...")
# We load Qwen in bfloat16 to save massive amounts of VRAM!
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
qwen_model.eval()

print("Loading CLIP...")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
clip_model.eval()

# --- 3. Load Existing Data ---
print(f"Loading existing latents from {existing_latents_path}...")
pre_saved_latents = torch.load(existing_latents_path)

print("Downloading/Loading Smithsonian Dataset for raw images...")
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

# --- 4. The Processing Loop ---
all_text_embeddings = []
csv_data = []

print("Starting the text encoding process...")

# The strict instruction for Qwen
system_prompt = "You are an automated image captioner. Output a single, short phrase describing the visual features. Never use lists, bullet points, or introductory text."
instruction_prompt = "Describe the color on the wings with patterns and spots. Be concise and tell ONLY the visual features of the butterfly."

with torch.no_grad():
    for index, row in enumerate(tqdm(dataset, desc="Writing Prompts")):
        raw_image = row['image'].convert("RGB")
        
        if random.random() < dropout_rate:
            final_text = ""
        else:
            # Qwen uses a strict Chat Template format
            messages = [
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},
                        {"type": "text", "text": instruction_prompt}
                    ]
                }
            ]
            
            # Prepare inputs using Qwen's processor
            text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = qwen_processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            # Generate the description
            generated_ids = qwen_model.generate(**inputs, max_new_tokens=25)
            
            # Trim the prompt tokens out of the output so we only get the new answer
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            final_text = qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            final_text = final_text.replace("\n", " ").replace("- ", "")

        csv_data.append({
            "image_index": index,
            "prompt": final_text
        })
            
        clip_inputs = clip_tokenizer(
            [final_text], 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        text_embedding = clip_model(**clip_inputs).last_hidden_state
        all_text_embeddings.append(text_embedding.cpu())


print(f"Saving {len(csv_data)} prompts to {csv_filename}...")

with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["image_index", "prompt"])
    writer.writeheader()
    writer.writerows(csv_data)

# --- 5. Save the Master Cache ---
print("Stacking text tensors...")
final_text_tensor = torch.cat(all_text_embeddings, dim=0)

assert len(final_text_tensor) == len(pre_saved_latents), f"Mismatch! {len(final_text_tensor)} texts vs {len(pre_saved_latents)} latents."

cache_dict = {
    "latents": pre_saved_latents,          
    "text_embeddings": final_text_tensor   
}

print(f"Saving final combined cache to {new_cache_save_path}...")
torch.save(cache_dict, new_cache_save_path)

print("✅ Success! Your dataset is fully prepared using Qwen2-VL!")