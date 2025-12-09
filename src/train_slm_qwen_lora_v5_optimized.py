"""
V5 OPTIMIZED TRAINING - Tối ưu hóa training cho Qwen2.5-0.5B
Target: 60-65% accuracy (tăng +9-14% so với 51%)

Key improvements:
1. Train full dataset (154k samples) with 3 epochs
2. Lower LR (5e-6) for stable learning
3. Cosine scheduler with longer warmup
4. Higher LoRA rank (64) for more capacity
5. Smart eval strategy (every 2000 steps)
6. Gradient clipping for stability
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

# ============================================================================
# V5 OPTIMIZED CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "slm_train_style_adapted.jsonl"
VAL_FILE = DATA_DIR / "slm_val.jsonl"

OUTPUT_DIR = BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v5-optimized"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ============================================================================
# OPTIMIZED TRAINING CONFIG
# ============================================================================
NUM_EPOCHS = 3                   # Tăng từ 1 → 3 epochs
BATCH_SIZE = 4                   # Keep small for memory
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 5e-6             # GIẢM từ 2e-5 → 5e-6 (học chậm, ổn định)
LOGGING_STEPS = 100
SAVE_STEPS = 2000                # Save every 2000 steps
EVAL_STEPS = 2000                # Eval every 2000 steps
MAX_LENGTH = 256
WARMUP_RATIO = 0.1               # Tăng từ 0.05 → 0.1 (warmup dài hơn)
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 16
MAX_GRAD_NORM = 1.0              # Gradient clipping

# ============================================================================
# IMPROVED LoRA CONFIG - Tăng capacity
# ============================================================================
LORA_R = 64                      # TĂNG từ 32 → 64 (model phức tạp hơn)
LORA_ALPHA = 128                 # TĂNG từ 64 → 128 (proportional to rank)
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(file_path: Path) -> List[Dict]:
    """Đọc file JSONL."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_prompt(input_text: str) -> str:
    """Tạo prompt."""
    return f"Bạn là một trợ lý y tế. Hãy trả lời Đúng hoặc Sai.\nNhận định: {input_text}\nĐáp án:"


class MedicalQADataset(Dataset):
    """Dataset cho medical QA."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = create_prompt(item["input"])
        full_text = prompt + " " + item["output"]

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Create labels - mask prompt part
        prompt_encoding = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length
        )
        prompt_length = len(prompt_encoding["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Mask prompt

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def compute_metrics(eval_pred):
    """Compute metrics cho evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    # Filter out -100 labels
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    # Accuracy
    accuracy = (predictions == labels).mean()

    return {"accuracy": accuracy}


def main():
    print("=" * 80)
    print("🚀 TRAINING V5 - OPTIMIZED FOR 0.5B MODEL")
    print("=" * 80)
    print(f"📊 Config:")
    print(f"  - Model: {MODEL_NAME}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - LoRA Rank: {LORA_R} (Alpha: {LORA_ALPHA})")
    print(f"  - Batch Size: {BATCH_SIZE} (Effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  - Device: {DEVICE}")
    print("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n📁 Loading data...")
    train_data = load_jsonl(TRAIN_FILE)
    val_data = load_jsonl(VAL_FILE) if VAL_FILE.exists() else []

    print(f"  ✅ Train samples: {len(train_data):,}")
    print(f"  ✅ Val samples: {len(val_data):,}")

    # Load tokenizer and model
    print("\n🔧 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    # Apply LoRA
    print("\n⚡ Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = MedicalQADataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = (
        MedicalQADataset(val_data, tokenizer, MAX_LENGTH)
        if val_data
        else None
    )

    # Training arguments
    print("\n⚙️  Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",        # Cosine scheduler
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,       # Gradient clipping
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS if val_dataset else None,
        save_total_limit=3,                # Keep only 3 best checkpoints
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        eval_strategy="steps" if val_dataset else "no",
        fp16=False,
        bf16=DEVICE == "cuda",             # Use bf16 on CUDA
        optim="adamw_torch",
        report_to="none",
        save_safetensors=True,
        gradient_checkpointing=True,       # Save memory
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Trainer
    print("\n🏋️  Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if val_dataset else None,
    )

    # Train
    print("\n" + "=" * 80)
    print("🔥 STARTING TRAINING")
    print("=" * 80)
    print(f"Total training steps: {len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS}")
    print(f"Expected time on T4: ~4-5 hours")
    print("=" * 80 + "\n")

    train_result = trainer.train()

    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save metrics
    metrics = {
        "train_loss": train_result.metrics["train_loss"],
        "train_runtime": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics["train_samples_per_second"],
    }

    if val_dataset:
        eval_results = trainer.evaluate()
        metrics.update(eval_results)

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED!")
    print("=" * 80)
    print(f"📊 Final Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.4f}")
    print(f"\n📁 Model saved to: {OUTPUT_DIR}")
    print(f"📊 Metrics saved to: {METRICS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
