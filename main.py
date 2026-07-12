import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ==========================================================
# Check GPU
# ==========================================================

print("=" * 60)
print("PyTorch :", torch.__version__)
print("CUDA Available :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available!")

print("GPU :", torch.cuda.get_device_name(0))
print("CUDA :", torch.version.cuda)

print(
    "VRAM :",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB",
)

print("=" * 60)

# ==========================================================
# Model
# ==========================================================

MODEL_PATH = "./models/Qwen3-4B-Instruct-2507"

# 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

print("Loading model (GPU)...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

print("\nModel loaded successfully!")

print(
    "GPU Memory Allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB",
)

print(
    "GPU Memory Reserved:",
    round(torch.cuda.memory_reserved() / 1024**3, 2),
    "GB",
)

# ==========================================================
# Chat
# ==========================================================

print("\n" + "=" * 60)
print("🤖 Qwen Local Chat")
print("Type 'exit' to quit")
print("=" * 60)

messages = []

while True:

    user_input = input("\nYou : ").strip()

    if user_input.lower() == "exit":
        print("\nGoodbye 👋")
        break

    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move tensors to GPU
    inputs = {
        k: v.to("cuda")
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    print("\nQwen :", response)

    messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    print(
        "\nGPU Allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )