"""
=========================================================
Dataset Loader for Qwen3 Fine-Tuning
=========================================================
"""

import os
import json
from glob import glob

from datasets import Dataset

from config import *


# ==========================================================
# Load JSONL Files
# ==========================================================

def load_jsonl_files():

    files = sorted(glob(os.path.join(DATASET_PATH, "*.jsonl")))

    if not files:
        raise FileNotFoundError(
            f"No JSONL files found in {DATASET_PATH}"
        )

    print("=" * 60)
    print("Loading Dataset Files")
    print("=" * 60)

    data = []

    for file in files:

        print(f"Loading : {os.path.basename(file)}")

        with open(file, "r", encoding="utf-8") as f:

            for line_number, line in enumerate(f, start=1):

                line = line.strip()

                if not line:
                    continue

                try:
                    sample = json.loads(line)

                    required = [
                        INSTRUCTION_FIELD,
                        EXPLANATION_FIELD,
                        CODE_FIELD,
                    ]

                    if not all(k in sample for k in required):
                        print(
                            f"Skipping line {line_number} in {os.path.basename(file)}"
                        )
                        continue

                    data.append(sample)

                except Exception as e:

                    print(
                        f"Invalid JSON in {os.path.basename(file)} "
                        f"Line {line_number}: {e}"
                    )

    print("=" * 60)
    print(f"Total Samples : {len(data)}")
    print("=" * 60)

    return data


# ==========================================================
# Create Prompt
# ==========================================================

def create_prompt(example):

    instruction = example[INSTRUCTION_FIELD].strip()
    explanation = example[EXPLANATION_FIELD].strip()
    code = example[CODE_FIELD].strip()

    category = example.get(CATEGORY_FIELD, "").strip()
    difficulty = example.get(DIFFICULTY_FIELD, "").strip()

    prompt = f"""<|im_start|>user
{instruction}

Category: {category}
Difficulty: {difficulty}
<|im_end|>

<|im_start|>assistant
Explanation:
{explanation}

Code:
{code}
<|im_end|>
"""

    return {
        "text": prompt
    }


# ==========================================================
# Build Dataset
# ==========================================================

def build_dataset():

    raw_data = load_jsonl_files()

    formatted = []

    for sample in raw_data:

        formatted.append(
            create_prompt(sample)
        )

    dataset = Dataset.from_list(formatted)

    return dataset


# ==========================================================
# Preview
# ==========================================================

def preview_dataset(dataset, n=3):

    print("=" * 60)

    print("Dataset Preview")

    print("=" * 60)

    for i in range(min(n, len(dataset))):

        print(dataset[i]["text"])

        print("-" * 60)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    dataset = build_dataset()

    print(dataset)

    print()

    preview_dataset(dataset)

    print()

    print("Total Samples :", len(dataset))