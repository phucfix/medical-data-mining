"""
Script huấn luyện model Qwen2.5-0.5B-Instruct với LoRA - Phiên bản v3
Training với AUGMENTED DATA để cải thiện generalization
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

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
# CONFIGURATION - TRAINING WITH AUGMENTED DATA
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Sử dụng AUGMENTED DATA
TRAIN_FILE = DATA_DIR / "slm_train_augmented.jsonl"
VAL_FILE = DATA_DIR / "slm_val_augmented.jsonl"

OUTPUT_DIR = BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v3-augmented"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

# Model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ============================================================================
# LoRA Configuration - OPTIMIZED
# ============================================================================
LORA_R = 32  # Tăng lên 32 để tận dụng augmented data
LORA_ALPHA = 64  # Tăng tương ứng
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ============================================================================
# Training Configuration - OPTIMIZED FOR AUGMENTED DATA
# ============================================================================
NUM_EPOCHS = 5  # Tăng epochs vì có nhiều data hơn
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
LOGGING_STEPS = 100
SAVE_STEPS = 1000
EVAL_STEPS = 1000
MAX_LENGTH = 256
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# DATASET
# ============================================================================

def load_jsonl(file_path: Path) -> List[Dict]:
    """Đọc file JSONL và trả về list of dict."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_prompt(input_text: str) -> str:
    """Tạo prompt cho model."""
    return f"Bạn là một trợ lý y tế. Hãy trả lời Đúng hoặc Sai.\nNhận định: {input_text}\nĐáp án:"


class MedicalQADataset(Dataset):
    """Dataset cho bài toán True/False y khoa."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = MAX_LENGTH,
        is_train: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Lấy input và output
        input_text = item["input"]
        output_text = item["output"]  # "TRUE" hoặc "FALSE"
        
        # Tạo prompt
        prompt = create_prompt(input_text)
        
        if self.is_train:
            # Cho training: tokenize cả prompt + output
            full_text = prompt + " " + output_text
            
            # Tokenize prompt để biết độ dài
            prompt_tokens = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length - 10,
            )
            prompt_len = len(prompt_tokens["input_ids"])
            
            # Tokenize full text
            full_tokens = self.tokenizer(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = full_tokens["input_ids"].squeeze()
            attention_mask = full_tokens["attention_mask"].squeeze()
            
            # Tạo labels: mask prompt phần, chỉ tính loss cho output
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        else:
            tokens = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "input_ids": tokens["input_ids"].squeeze(),
                "attention_mask": tokens["attention_mask"].squeeze(),
            }


# ============================================================================
# TRAINING
# ============================================================================

def compute_metrics(eval_pred):
    """Tính metrics cho evaluation."""
    predictions, labels = eval_pred
    
    # Predictions là logits, lấy argmax
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Chỉ tính accuracy trên phần output (không tính prompt)
    predictions = np.argmax(predictions, axis=-1)
    
    # Tính accuracy
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum()
    
    return {"accuracy": accuracy}


def train_model():
    """Huấn luyện model với LoRA."""
    print("=" * 60)
    print("TRAINING QWEN 2.5-0.5B WITH LORA - VERSION 3 (AUGMENTED)")
    print("=" * 60)
    
    # Check files
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training file not found: {TRAIN_FILE}")
        print("Please run augmentation pipeline first:")
        print("  1. python src/augment_data.py")
        print("  2. python src/organize_augmented_data.py")
        return
    
    # Load data
    print(f"\nLoading augmented training data from {TRAIN_FILE}...")
    train_data = load_jsonl(TRAIN_FILE)
    print(f"Train samples: {len(train_data)}")
    
    print(f"\nLoading validation data from {VAL_FILE}...")
    val_data = load_jsonl(VAL_FILE)
    print(f"Validation samples: {len(val_data)}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print(f"\nLoading base model {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    
    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    
    print(f"  LoRA r: {LORA_R}")
    print(f"  LoRA alpha: {LORA_ALPHA}")
    print(f"  LoRA dropout: {LORA_DROPOUT}")
    print(f"  Target modules: {LORA_TARGET_MODULES}")
    
    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MedicalQADataset(train_data, tokenizer, is_train=True)
    val_dataset = MedicalQADataset(val_data, tokenizer, is_train=True)
    
    # Training arguments
    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=DEVICE == "cuda",
        report_to="none",
        save_total_limit=3,
    )
    
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Final validation accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Model saved to: {OUTPUT_DIR}")
    
    # Save metrics
    metrics = {
        "final_val_accuracy": eval_results["eval_accuracy"],
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "epochs": NUM_EPOCHS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
    }
    
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {METRICS_FILE}")
    print("\n✓ Training pipeline completed successfully!")


if __name__ == "__main__":
    train_model()
