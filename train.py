"""
=========================================================
Qwen3-4B QLoRA Fine-Tuning
train.py (Part 1) - fixed for trl==1.8.0 / transformers==5.13.0
=========================================================
"""

import os

os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from trl import SFTTrainer, SFTConfig

from dataset import build_dataset
from config import *

# ==========================================================
# GPU INFO
# ==========================================================

print("=" * 60)
print("PyTorch :", torch.__version__)
print("CUDA Available :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")

print("GPU :", torch.cuda.get_device_name(0))
print("CUDA :", torch.version.cuda)

print(
    "VRAM :",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2,
    ),
    "GB",
)

print("=" * 60)

# ==========================================================
# Tokenizer
# ==========================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.eos_token

# ==========================================================
# Dataset
# ==========================================================

print("Loading Dataset...")

dataset = build_dataset()

print(dataset)

# ==========================================================
# Quantization
# ==========================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=LOAD_IN_4BIT,

    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,

    bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,

    bnb_4bit_use_double_quant=BNB_USE_DOUBLE_QUANT,

)

# ==========================================================
# Model
# ==========================================================

print("Loading Base Model...")

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",

    trust_remote_code=True,

)

model.config.use_cache = False

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# ==========================================================
# Prepare for QLoRA
# ==========================================================

model = prepare_model_for_kbit_training(model)

# ==========================================================
# LoRA
# ==========================================================

lora_config = LoraConfig(

    r=LORA_R,

    lora_alpha=LORA_ALPHA,

    target_modules=TARGET_MODULES,

    lora_dropout=LORA_DROPOUT,

    bias=BIAS,

    task_type="CAUSAL_LM",

)

model = get_peft_model(

    model,

    lora_config,

)

model.print_trainable_parameters()

print("=" * 60)
print("Model Ready For Training")
print("=" * 60)


# ==========================================================
# Training Arguments (SFTConfig, not plain TrainingArguments)
# ==========================================================
# SFTConfig subclasses TrainingArguments and adds SFT-specific
# fields (dataset_text_field, max_length, packing, etc.). Using
# plain TrainingArguments with SFTTrainer in trl>=1.x means those
# SFT-specific fields silently fall back to defaults.
#
# NOTE: set `dataset_text_field` (or `formatting_func` in
# SFTTrainer) to match whatever column build_dataset() produces,
# and set `max_length` explicitly (old `max_seq_length` was
# renamed).
training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    warmup_ratio=WARMUP_RATIO,

    logging_steps=LOGGING_STEPS,

    save_steps=SAVE_STEPS,

    save_total_limit=SAVE_TOTAL_LIMIT,

    fp16=False,

    bf16=False,

    optim="paged_adamw_8bit",

    lr_scheduler_type="cosine",

    report_to="none",

    seed=SEED,

    max_length=MAX_SEQ_LENGTH,

    max_grad_norm=MAX_GRAD_NORM,

    dataset_text_field="text",

    packing=False,

    save_only_model=True,

)
# ==========================================================
# Trainer
# ==========================================================
# No manual data_collator here: SFTTrainer builds its own
# collator from the raw dataset (handles tokenization + label
# masking internally). Passing DataCollatorForLanguageModeling
# expects pre-tokenized input_ids and will conflict with that
# pipeline.

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,

)
# ==========================================================
# Training
# ==========================================================

print("=" * 60)
print("Starting Training")
print("=" * 60)

trainer.train()

print("=" * 60)
print("Training Completed")
print("=" * 60)

# ==========================================================
# Save Adapter
# ==========================================================

print("Saving LoRA Adapter...")

trainer.model.save_pretrained(ADAPTER_DIR)

tokenizer.save_pretrained(ADAPTER_DIR)

print("Adapter Saved ->", ADAPTER_DIR)

# ==========================================================
# Save Trainer State
# ==========================================================

trainer.save_state()

print("=" * 60)
print("Training Finished Successfully")
print("=" * 60)