from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_name = "Qwen/Qwen3-4B-Instruct-2507"
save_dir = "./models/Qwen3-4B-Instruct-2507"

os.makedirs(save_dir, exist_ok=True)

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)

print("Saving tokenizer...")
tokenizer.save_pretrained(save_dir)

print("Saving model...")
model.save_pretrained(save_dir)

print(f"\nModel downloaded successfully!")
print(f"Location: {os.path.abspath(save_dir)}")