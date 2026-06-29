import torch
import csv
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
csv_filename = "butterfly_prompts.csv"

print(f"Loading BLIP to {device}...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# Using safetensors to avoid the PyTorch security error!
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base", 
    use_safetensors=True
).to(device)
blip_model.eval()

print("Downloading/Loading Smithsonian Dataset...")
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

# This list will hold our data rows before writing to CSV
csv_data = []
guide_prompt = "a close-up photo of a butterfly with "

print("Generating captions for CSV...")
with torch.no_grad():
    for index, row in enumerate(tqdm(dataset, desc="Writing Prompts")):
        raw_image = row['image'].convert("RGB")
        
        # Pass image and guide text to BLIP
        blip_inputs = blip_processor(images=raw_image, text=guide_prompt, return_tensors="pt").to(device)
        blip_out = blip_model.generate(**blip_inputs, max_new_tokens=25, num_beams=3, repetition_penalty=1.5, early_stopping=True)
        
        # Decode the text
        final_text = blip_processor.decode(blip_out[0], skip_special_tokens=True)
        
        # Optional: Clean up the prompt by removing the guide text so the CSV is clean
        clean_text = final_text.replace(guide_prompt, "").strip()
        
        # Add to our list
        csv_data.append({
            "image_index": index,
            "prompt": clean_text
        })

print(f"Saving {len(csv_data)} prompts to {csv_filename}...")

# Write the list of dictionaries to a CSV file
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["image_index", "prompt"])
    writer.writeheader()
    writer.writerows(csv_data)

print("✅ CSV generated successfully! You can now open it in Excel or Pandas.")