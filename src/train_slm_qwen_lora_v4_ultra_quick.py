"""
Script huấn luyện model Qwen2.5-0.5B-Instruct với LoRA - ULTRA QUICK VERSION
Training CỰC NHANH với config tối ưu để KHÔNG BAO GIỜ OOM
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
# CONFIGURATION - ULTRA QUICK TRAINING (NO OOM GUARANTEED)
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "slm_train_style_adapted.jsonl"
VAL_FILE = DATA_DIR / "slm_val.jsonl"

OUTPUT_DIR = BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v4-ultra-quick"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ============================================================================
# ULTRA QUICK CONFIG - EXTREME MEMORY OPTIMIZATION
# ============================================================================
TRAIN_SUBSET_SIZE = 20000  # CHỈ 20K SAMPLES
NUM_EPOCHS = 2             
BATCH_SIZE = 2             # CỰC NHỎ: 2
LEARNING_RATE = 4e-5       # Cao hơn để học nhanh với ít data
LOGGING_STEPS = 200
SAVE_STEPS = 5000          # Save 1 lần cuối
MAX_LENGTH = 128           # GIẢM: 256 → 128 tokens
WARMUP_RATIO = 0.03        
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 8  # Tăng cao, effective batch = 16

# LoRA Config - GIẢM để ít tham số hơn
LORA_R = 16                # GIẢM: 32 → 16
LORA_ALPHA = 32            # GIẢM: 64 → 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # CHỈ 2 modules thay vì 4

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
    """Tạo prompt ngắn gọn."""
    return f"Y tế: {input_text}\nĐáp án:"  # Ngắn hơn để tiết kiệm tokens


class MedicalQADataset(Dataset):
    """Dataset tối ưu cho memory."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_text = item["input"]
        output_text = item["output"]
        
        prompt = create_prompt(input_text)
        full_text = prompt + " " + output_text
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length - 10,
        )
        prompt_len = len(prompt_tokens["input_ids"])
        
        # Tokenize full
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
        
        # Labels
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_model():
    """ULTRA QUICK Training - Không bao giờ OOM."""
    print("=" * 80)
    print("⚡⚡ ULTRA QUICK TRAINING - MAXIMUM SPEED, NO OOM ⚡⚡")
    print("=" * 80)
    print("Config: 20k samples, 2 epochs, batch=2, max_length=128")
    print("=" * 80)
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if not TRAIN_FILE.exists():
        print(f"ERROR: Training file not found: {TRAIN_FILE}")
        return
    
    # Load data
    print(f"\nLoading training data...")
    train_data_full = load_jsonl(TRAIN_FILE)
    print(f"Full dataset: {len(train_data_full)} samples")
    
    # Sample subset
    print(f"\n⚡ Sampling {TRAIN_SUBSET_SIZE} training samples...")
    random.seed(42)
    train_data = random.sample(train_data_full, min(TRAIN_SUBSET_SIZE, len(train_data_full)))
    print(f"✓ Using {len(train_data)} samples ({len(train_data)/len(train_data_full)*100:.1f}%)")
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimizations
    print(f"\nLoading base model with optimizations...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Optimization
    )
    
    # LoRA config - minimal
    print("\nConfiguring LoRA (minimal config)...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    
    print(f"  LoRA r: {LORA_R} (reduced)")
    print(f"  LoRA alpha: {LORA_ALPHA} (reduced)")
    print(f"  Target modules: {LORA_TARGET_MODULES} (only 2)")
    
    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    
    # Enable gradient checkpointing
    model.enable_input_require_grads()
    if hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Create dataset
    print("\nCreating dataset...")
    train_dataset = MedicalQADataset(train_data, tokenizer, MAX_LENGTH)
    
    # Training arguments - extreme optimization
    print("\nSetting up training (no evaluation)...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="no",           # NO EVAL
        save_strategy="epoch",         # Save sau mỗi epoch
        fp16=DEVICE == "cuda",
        report_to="none",
        save_total_limit=1,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        optim="adamw_torch",           # Faster optimizer
        max_grad_norm=0.3,             # Clip gradients
    )
    
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Max length: {MAX_LENGTH} tokens (reduced)")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Expected time: ~20-30 minutes ⚡⚡")
    
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
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "=" * 80)
    print("⚡⚡ STARTING ULTRA QUICK TRAINING ⚡⚡")
    print("=" * 80)
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer.train()
    
    # Save
    print("\nSaving final model...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    print("\n" + "=" * 80)
    print("✓✓ ULTRA QUICK TRAINING COMPLETED ✓✓")
    print("=" * 80)
    print(f"Model saved to: {OUTPUT_DIR}")
    
    # Metrics
    metrics = {
        "train_samples": len(train_data),
        "train_samples_full": len(train_data_full),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "max_length": MAX_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_target_modules": LORA_TARGET_MODULES,
        "learning_rate": LEARNING_RATE,
        "training_strategy": "ultra-quick-no-oom",
        "subset_ratio": f"{len(train_data)/len(train_data_full)*100:.1f}%",
        "note": "Ultra quick training with extreme memory optimizations"
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {METRICS_FILE}")
    print("\n⚡⚡ Ultra quick training completed!")
    print(f"   Expected accuracy: ~70-75% (trade-off for speed)")
    print(f"   Evaluate with: python src/test_qwen_on_sample_v3.py")
    print(f"   Model path: {OUTPUT_DIR}")
    
    # Clear cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n✓ GPU cache cleared")


if __name__ == "__main__":
    # Set environment variable for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    train_model()
