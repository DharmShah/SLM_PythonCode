"""
=========================================================
Configuration File
Qwen3-4B-Instruct-2507 QLoRA Fine-Tuning
=========================================================
"""

import os
import torch

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = os.path.join(
    BASE_DIR,
    "models",
    "Qwen3-4B-Instruct-2507"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "DataSet"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

ADAPTER_DIR = os.path.join(
    BASE_DIR,
    "adapters"
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

# ==========================================================
# Dataset
# ==========================================================

MAX_SEQ_LENGTH = 1024

TRAIN_SPLIT = 0.95

INSTRUCTION_FIELD = "instruction"
EXPLANATION_FIELD = "explanation"
CODE_FIELD = "code"
CATEGORY_FIELD = "category"
DIFFICULTY_FIELD = "difficulty"

# ==========================================================
# LoRA
# ==========================================================

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

BIAS = "none"

TARGET_MODULES = [

    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",

    "gate_proj",
    "up_proj",
    "down_proj",

]

# ==========================================================
# Quantization
# ==========================================================

LOAD_IN_4BIT = True

BNB_4BIT_QUANT_TYPE = "nf4"

BNB_4BIT_COMPUTE_DTYPE = torch.float16

BNB_USE_DOUBLE_QUANT = True

# ==========================================================
# Training
# ==========================================================

NUM_EPOCHS = 3

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.001

WARMUP_RATIO = 0.03

MAX_GRAD_NORM = 0.3

PER_DEVICE_TRAIN_BATCH_SIZE = 1

PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LOGGING_STEPS = 10

SAVE_STEPS = 100

SAVE_TOTAL_LIMIT = 3

OPTIMIZER = "paged_adamw_8bit"

LR_SCHEDULER = "cosine"

SEED = 42

# ==========================================================
# Trainer
# ==========================================================

USE_FP16 = True

USE_BF16 = False

GRADIENT_CHECKPOINTING = True

PACKING = False

SAVE_ONLY_MODEL = True

RESUME_FROM_CHECKPOINT = None

# ==========================================================
# Inference
# ==========================================================

MAX_NEW_TOKENS = 256

TEMPERATURE = 0.7

TOP_P = 0.9

TOP_K = 50

REPETITION_PENALTY = 1.1

# ==========================================================
# Create Directories
# ==========================================================

for folder in [

    OUTPUT_DIR,
    ADAPTER_DIR,
    LOG_DIR,
    CHECKPOINT_DIR,

]:

    os.makedirs(folder, exist_ok=True)

# ==========================================================
# Device
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# Preview
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print("Qwen3 QLoRA Configuration")

    print("=" * 60)

    print("Model           :", MODEL_NAME)
    print("Dataset         :", DATASET_PATH)
    print("Output          :", OUTPUT_DIR)
    print("Adapters        :", ADAPTER_DIR)
    print("Logs            :", LOG_DIR)
    print("Device          :", DEVICE)
    print("Epochs          :", NUM_EPOCHS)
    print("Batch Size      :", PER_DEVICE_TRAIN_BATCH_SIZE)
    print("Gradient Acc    :", GRADIENT_ACCUMULATION_STEPS)
    print("Learning Rate   :", LEARNING_RATE)
    print("Max Seq Length  :", MAX_SEQ_LENGTH)
    print("LoRA Rank       :", LORA_R)
    print("4-bit           :", LOAD_IN_4BIT)

    print("=" * 60)