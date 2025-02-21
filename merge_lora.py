import argparse
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")

parser.add_argument("--base_model", type=str, required=True, help="Path to the base model.")
parser.add_argument("--lora_adapter", type=str, required=True, help="Path to the LoRA adapter (safetensors).")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model.")

args = parser.parse_args()

# Load base model
print(f"Loading base model from {args.base_model}...")
base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)

# Load LoRA adapter
print(f"Loading LoRA adapter from {args.lora_adapter}...")
config = LoraConfig.from_pretrained(args.lora_adapter)
model = get_peft_model(base_model, config)

# Merge LoRA into base model
print("Merging LoRA adapter into base model...")
model = model.merge_and_unload()

# Save the merged model
print(f"Saving merged model to {args.output_dir}...")
model.save_pretrained(args.output_dir)

print("✅ Model merging complete!")
