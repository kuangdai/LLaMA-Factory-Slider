import argparse
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")

parser.add_argument("-b", "--base_model", type=str, required=True, help="Path to the base model.")
parser.add_argument("-l", "--lora_adapter", type=str, required=True, help="Path to the LoRA adapter (safetensors).")
parser.add_argument("--slider_on", action="store_true", help="Whether slider is turned on.")
parser.add_argument("--slider_n_variables", type=int, default=3, help="Number of slider variables.")
parser.add_argument("--slider_n_heads_sharing_slider", type=int, default=2,
                    help="How many heads sharing the same slider kv.")
parser.add_argument("-o", "--output_dir", type=str, default=None, help="Directory to save the merged model.")

args = parser.parse_args()

# Step 1: Load Base Model
print(f"Loading base model from {args.base_model}...")
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype="auto",
    slider_on=args.slider_on,
    slider_n_variables=args.slider_n_variables,
    slider_n_heads_sharing_slider=args.slider_n_heads_sharing_slider
)

# Step 2: Load LoRA Adapter
print(f"Loading LoRA adapter from {args.lora_adapter}...")
config = LoraConfig.from_pretrained(args.lora_adapter)
model = get_peft_model(base_model, config)

# Step 3: Merge LoRA into Base Model
print("Merging LoRA adapter into base model...")
model = model.merge_and_unload()

# Step 4: Merge Slider Weights (if enabled)
if args.slider_on:
    print("Merging Slider weights...")

    # Load safetensor weights
    weights_path = f"{args.lora_adapter}/adapter_model.safetensors"
    state_dict = load_file(weights_path)

    # Remove "base_model.model." prefix from all keys
    prefix_to_remove = "base_model.model."
    slider_weights = {k[len(prefix_to_remove):]: v for k, v in state_dict.items() if k.startswith(prefix_to_remove)}

    # Load slider weights into model
    missing_keys, unexpected_keys = model.load_state_dict(slider_weights, strict=False)

    # Raise error if unexpected keys are found
    if unexpected_keys:
        raise ValueError(f"Unexpected keys found in safetensors: {unexpected_keys}")

# Step 5: Save Merged Model
if args.output_dir is None:
    args.output_dir = args.lora_adapter + "/merged"

print(f"Saving merged model to {args.output_dir}...")
model.save_pretrained(args.output_dir)

print("Model merging complete.")
