import os
# This forces Hugging Face to download models to your roomy lab drive
os.environ["HF_HOME"] = "/mnt/lab/asif/hf_cache"

import csv
import torch
import random
from datasets import load_dataset
from tqdm.auto import tqdm

# Hugging Face Models
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPTokenizer, CLIPTextModel

# --- 1. Settings ---
device = "cuda" if torch.cuda.is_available() else "cpu"
existing_latents_path = "cached_smithsonian_latents.pt" # Your previous file!
new_cache_save_path = "smithsonian_latents_and_prompts.pt"
dropout_rate = 0.1 
csv_filename = "butterfly_prompts.csv"
print(f"Loading models to {device}...")

# --- 2. Load Text Models ONLY (No VAE!) ---
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",use_safetensors=True).to(device)
blip_model.eval()

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
clip_model.eval()

# --- 3. Load Existing Data ---
print(f"Loading existing latents from {existing_latents_path}...")
# This loads your massive tensor directly into RAM instantly
pre_saved_latents = torch.load(existing_latents_path)

print("Downloading/Loading Smithsonian Dataset for raw images...")
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

# --- 4. The Processing Loop ---
all_text_embeddings = []

print("Starting the text encoding process...")

csv_data = []
guide_prompt = "a close-up photo of a butterfly with "

with torch.no_grad():
    for index, row in enumerate(tqdm(dataset, desc="Writing Prompts")):
        # Get raw image for BLIP
        raw_image = row['image'].convert("RGB")
        
        if random.random() < dropout_rate:
            final_text = ""
        else:
            # Pass BOTH the image and the text guide to the processor
            blip_inputs = blip_processor(images=raw_image, text=guide_prompt, return_tensors="pt").to(device)
            blip_out = blip_model.generate(**blip_inputs, max_new_tokens=25, num_beams=3, repetition_penalty=1.5, early_stopping=True)
            # Decode the final text
            final_text = blip_processor.decode(blip_out[0], skip_special_tokens=True)
            final_text = final_text.replace(guide_prompt, "").strip()

        csv_data.append({
            "image_index": index,
            "prompt": final_text
        })
            
        # Convert string to CLIP matrix
        clip_inputs = clip_tokenizer(
            [final_text], 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        text_embedding = clip_model(**clip_inputs).last_hidden_state # Shape: (1, 77, 768)
        
        all_text_embeddings.append(text_embedding.cpu())


print(f"Saving {len(csv_data)} prompts to {csv_filename}...")

# Write the list of dictionaries to a CSV file
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["image_index", "prompt"])
    writer.writeheader()
    writer.writerows(csv_data)
# --- 5. Save the Master Cache ---
print("Stacking text tensors...")
final_text_tensor = torch.cat(all_text_embeddings, dim=0)

# Safety check: Ensure we have the exact same number of text embeddings as image latents
assert len(final_text_tensor) == len(pre_saved_latents), f"Mismatch! {len(final_text_tensor)} texts vs {len(pre_saved_latents)} latents."

cache_dict = {
    "latents": pre_saved_latents,          # Re-using your hard work!
    "text_embeddings": final_text_tensor   # Adding the new text math!
}

print(f"Saving final combined cache to {new_cache_save_path}...")
torch.save(cache_dict, new_cache_save_path)

print("✅ Success! Your dataset is fully prepared for Prompt-Conditioned training!")