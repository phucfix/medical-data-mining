"""
Script huấn luyện model Qwen2.5-0.5B-Instruct với LoRA - QUICK VERSION
Training NHANH với subset data để demo/deadline
"""

import json
import os
import random
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
# CONFIGURATION - QUICK TRAINING
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "slm_train_style_adapted.jsonl"
VAL_FILE = DATA_DIR / "slm_val.jsonl"

OUTPUT_DIR = BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v4-quick"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

# Model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ============================================================================
# QUICK TRAINING CONFIG - GIẢM DATA & EPOCHS
# ============================================================================
TRAIN_SUBSET_SIZE = 30000  # CHỈ DÙNG 30K/154K SAMPLES (~20%)
VAL_SUBSET_SIZE = 500      # GIẢM: 1000 → 500 samples

NUM_EPOCHS = 2             # 4 → 2 EPOCHS
BATCH_SIZE = 4             # GIẢM: 8 → 4 để tránh OOM
EVAL_BATCH_SIZE = 1        # GIẢM: 4 → 1 để tránh OOM
LEARNING_RATE = 3e-5       # Tăng LR: 2e-5 → 3e-5
LOGGING_STEPS = 100
SAVE_STEPS = 2000          # Tăng: 1000 → 2000
EVAL_STEPS = 2000          # Tăng: 1000 → 2000 (eval ít hơn)
MAX_LENGTH = 256
WARMUP_RATIO = 0.05        # Giảm warmup: 0.1 → 0.05
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4  # TĂNG: 2 → 4, effective batch vẫn = 16

# LoRA Config - GIỮ NGUYÊN
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

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
        
        input_text = item["input"]
        output_text = item["output"]
        
        prompt = create_prompt(input_text)
        
        if self.is_train:
            full_text = prompt + " " + output_text
            
            prompt_tokens = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length - 10,
            )
            prompt_len = len(prompt_tokens["input_ids"])
            
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
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = np.argmax(predictions, axis=-1)
    
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum()
    
    return {"accuracy": accuracy}


def train_model():
    """Huấn luyện model với LoRA - QUICK VERSION."""
    print("=" * 70)
    print("QUICK TRAINING - QWEN 2.5-0.5B WITH LORA")
    print("=" * 70)
    print("⚡ Fast training with subset data for quick results")
    print("=" * 70)
    
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training file not found: {TRAIN_FILE}")
        print("Please run: python src/augment_with_style.py")
        return
    
    # Load data
    print(f"\nLoading training data from {TRAIN_FILE}...")
    train_data_full = load_jsonl(TRAIN_FILE)
    print(f"Full training data: {len(train_data_full)} samples")
    
    # SAMPLE SUBSET - RANDOM SAMPLING
    print(f"\n⚡ Sampling {TRAIN_SUBSET_SIZE} training samples...")
    random.seed(42)
    train_data = random.sample(train_data_full, min(TRAIN_SUBSET_SIZE, len(train_data_full)))
    print(f"✓ Using {len(train_data)} training samples ({len(train_data)/len(train_data_full)*100:.1f}%)")
    
    print(f"\nLoading validation data from {VAL_FILE}...")
    val_data_full = load_jsonl(VAL_FILE)
    val_data = val_data_full[:VAL_SUBSET_SIZE]
    print(f"✓ Using {len(val_data)}/{len(val_data_full)} validation samples")
    
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
    
    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    
    # Enable gradient checkpointing
    model.enable_input_require_grads()
    if hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()
    
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
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        # TẮT EVALUATION TRONG TRAINING ĐỂ TRÁNH OOM
        eval_strategy="no",  # ❌ KHÔNG EVAL TRONG TRAINING
        save_strategy="steps",
        load_best_model_at_end=False,  # ❌ Không load best model (vì không eval)
        fp16=DEVICE == "cuda",
        report_to="none",
        save_total_limit=1,  # CHỈ GIỮ 1 CHECKPOINT
        gradient_checkpointing=True,
        # Thêm các config giảm memory
        dataloader_num_workers=0,  # Không dùng multiprocessing
        ddp_find_unused_parameters=False,
    )
    
    print(f"  Epochs: {NUM_EPOCHS} ⚡ (Quick)")
    print(f"  Train samples: {len(train_data)} ⚡ (Subset)")
    print(f"  Train batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Learning rate: {LEARNING_RATE} ⚡ (Higher)")
    print(f"  Evaluation: DISABLED ⚡ (to avoid OOM)")
    print(f"  Expected time: ~25-35 minutes ⚡")
    
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
    print("\n" + "=" * 70)
    print("⚡ STARTING QUICK TRAINING")
    print("=" * 70)
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    print("\n" + "=" * 70)
    print("✓ QUICK TRAINING COMPLETED")
    print("=" * 70)
    print(f"Model saved to: {OUTPUT_DIR}")
    print("\n⚠️  Note: Evaluation was skipped during training to avoid OOM")
    print("   You can evaluate later with test_qwen_on_sample_v3.py")
    
    # Save metrics
    metrics = {
        "train_samples": len(train_data),
        "train_samples_full": len(train_data_full),
        "val_samples": len(val_data),
        "epochs": NUM_EPOCHS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "training_strategy": "quick-style-adapted-subset-no-eval",
        "subset_ratio": f"{len(train_data)/len(train_data_full)*100:.1f}%",
        "note": "Quick training with 30k subset, no evaluation during training to avoid OOM",
        "eval_note": "Run test_qwen_on_sample_v3.py to evaluate on Test_sample.v1.0.csv"
    }
    
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {METRICS_FILE}")
    print("\n⚡ Quick training completed!")
    print(f"   Expected accuracy on Test_sample: ~75-80%")
    print(f"   Training time: Much faster than full training")


if __name__ == "__main__":
    train_model()
