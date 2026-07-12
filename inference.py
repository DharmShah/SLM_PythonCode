import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel

# =====================================================
# Paths
# =====================================================

BASE_MODEL = "./models/Qwen3-4B-Instruct-2507"
ADAPTER = "./adapters"

# =====================================================
# GPU Info
# =====================================================

print("=" * 60)

print("PyTorch :", torch.__version__)
print("CUDA Available :", torch.cuda.is_available())

if torch.cuda.is_available():

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

# =====================================================
# Quantization
# =====================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True,

)

# =====================================================
# Tokenizer
# =====================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# Base Model
# =====================================================

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    quantization_config=bnb_config,

    device_map="auto",

)

# =====================================================
# Load LoRA
# =====================================================

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(

    model,

    ADAPTER,

)

model.eval()

print("\nModel Loaded Successfully!")

# =====================================================
# Chat
# =====================================================

messages = []

print("=" * 60)
print("Qwen3 + LoRA Chat")
print("Type 'exit' to quit")
print("=" * 60)

while True:

    user = input("\nYou : ")

    if user.lower() == "exit":
        break

    messages.append(
        {
            "role": "user",
            "content": user,
        }
    )

    inputs = tokenizer.apply_chat_template(

        messages,

        tokenize=True,

        add_generation_prompt=True,

        return_tensors="pt",

        return_dict=True,

    )

    inputs = {

        k: v.to(model.device)

        for k, v in inputs.items()

    }

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=256,

            temperature=0.7,

            top_p=0.9,

            repetition_penalty=1.1,

            do_sample=True,

        )

    response = tokenizer.decode(

        outputs[0][inputs["input_ids"].shape[-1]:],

        skip_special_tokens=True,

    )

    print("\nAssistant :", response)

    messages.append(

        {

            "role": "assistant",

            "content": response,

        }

    )

    if torch.cuda.is_available():

        print(
            "\nGPU Memory :",
            round(torch.cuda.memory_allocated()/1024**3,2),
            "GB",
        )