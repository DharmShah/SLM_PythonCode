"""
=========================================================
Merge LoRA Adapter with Base Model
=========================================================
"""

import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel

from config import *

# ==========================================================
# Output Directory
# ==========================================================

MERGED_MODEL_PATH = "./merged_model"

os.makedirs(MERGED_MODEL_PATH, exist_ok=True)

# ==========================================================
# GPU
# ==========================================================

print("=" * 60)

print("CUDA :", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU :", torch.cuda.get_device_name(0))

print("=" * 60)

# ==========================================================
# Tokenizer
# ==========================================================

print("Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# ==========================================================
# Base Model
# ==========================================================

print("Loading Base Model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

# ==========================================================
# Load Adapter
# ==========================================================

print("Loading LoRA Adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)

# ==========================================================
# Merge
# ==========================================================

print("Merging Adapter...")

merged_model = model.merge_and_unload()

print("Merge Completed!")

# ==========================================================
# Save Model
# ==========================================================

print("Saving Merged Model...")

merged_model.save_pretrained(
    MERGED_MODEL_PATH,
    safe_serialization=True,
)

tokenizer.save_pretrained(
    MERGED_MODEL_PATH
)

print()

print("=" * 60)

print("Merged Model Saved")

print(MERGED_MODEL_PATH)

print("=" * 60)