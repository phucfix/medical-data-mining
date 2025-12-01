"""
Script huấn luyện model Qwen2.5-0.5B-Instruct với LoRA cho bài toán True/False y khoa.
Sử dụng transformers, peft (LoRA), và Trainer.
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
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "slm_train.jsonl"
VAL_FILE = DATA_DIR / "slm_val.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

# Model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training Configuration
NUM_EPOCHS = 1  # Giảm số epoch để train nhanh hơn
BATCH_SIZE = 8  # Tăng batch size nếu GPU đủ VRAM
LEARNING_RATE = 5e-5  # Tăng learning rate để convergence nhanh hơn
LOGGING_STEPS = 50  # Giảm số lần logging
SAVE_STEPS = 500  # Giảm số lần save
EVAL_STEPS = 500  # Giảm số lần eval
MAX_LENGTH = 256
WARMUP_RATIO = 0.1

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
                max_length=self.max_length - 10,  # Để chừa chỗ cho output
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
            labels[:prompt_len] = -100  # Không tính loss cho prompt
            
            # Mask padding
            labels[attention_mask == 0] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        else:
            # Cho evaluation: chỉ tokenize prompt
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
                "label_text": output_text,  # Để so sánh khi evaluate
                "prompt": prompt,
            }


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer():
    """Load model và tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    # Đảm bảo có pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Parameters in billions: {total_params / 1e9:.3f}B")
    
    return model, tokenizer


def apply_lora(model):
    """Áp dụng LoRA vào model."""
    print("\nApplying LoRA configuration...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"All parameters: {all_params:,}")
    print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")
    
    return model


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, tokenizer, val_data: List[Dict], device: str) -> Dict:
    """
    Đánh giá model trên validation set.
    Generate câu trả lời và tính accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    print("\nEvaluating on validation set...")
    
    with torch.no_grad():
        for i, item in enumerate(val_data):
            input_text = item["input"]
            true_label = item["output"].upper()
            
            # Tạo prompt
            prompt = create_prompt(input_text)
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip().upper()
            
            # Extract TRUE/FALSE from generated text
            if "TRUE" in generated_text:
                pred_label = "TRUE"
            elif "FALSE" in generated_text:
                pred_label = "FALSE"
            else:
                pred_label = generated_text[:10]  # Fallback
            
            # Check correctness
            is_correct = pred_label == true_label
            if is_correct:
                correct += 1
            total += 1
            
            predictions.append({
                "input": input_text[:50] + "...",
                "true": true_label,
                "pred": pred_label,
                "correct": is_correct
            })
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"  Evaluated {i + 1}/{len(val_data)} samples...")
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nEvaluation Results:")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    # Print some examples
    print("\nSample predictions:")
    for pred in predictions[:5]:
        status = "✓" if pred["correct"] else "✗"
        print(f"  {status} True: {pred['true']}, Pred: {pred['pred']}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions
    }


# ============================================================================
# TRAINING
# ============================================================================

def train():
    """Hàm chính để huấn luyện model."""
    print("=" * 60)
    print("TRAINING QWEN2.5-0.5B WITH LORA FOR MEDICAL QA")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    train_data = load_jsonl(TRAIN_FILE)
    val_data = load_jsonl(VAL_FILE)
    print(f"   Train samples: {len(train_data)}")
    print(f"   Val samples: {len(val_data)}")
    
    # Load model and tokenizer
    print("\n2. Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Apply LoRA
    print("\n3. Applying LoRA...")
    model = apply_lora(model)
    
    # Create datasets
    print("\n4. Creating datasets...")
    train_dataset = MedicalQADataset(train_data, tokenizer, is_train=True)
    val_dataset = MedicalQADataset(val_data, tokenizer, is_train=True)
    
    print(f"   Train dataset size: {len(train_dataset)}")
    print(f"   Val dataset size: {len(val_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # Training arguments
    print("\n5. Setting up training...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=False,  # Tắt để giảm overhead
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=DEVICE == "cuda",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=2,  # Tăng batch logic mà không cần nhiều VRAM
    )
    
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Logging steps: {LOGGING_STEPS}")
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n6. Starting training...")
    print("=" * 60)
    
    train_result = trainer.train()
    
    print("=" * 60)
    print("Training completed!")
    
    # Get training metrics
    train_metrics = train_result.metrics
    print(f"\nTraining metrics:")
    print(f"   Train loss: {train_metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   Train runtime: {train_metrics.get('train_runtime', 'N/A'):.2f}s")
    
    # Evaluate on validation set
    print("\n7. Final evaluation on validation set...")
    eval_result = trainer.evaluate()
    print(f"   Eval loss: {eval_result.get('eval_loss', 'N/A'):.4f}")
    
    # Custom evaluation with accuracy
    print("\n8. Computing accuracy on validation set...")
    accuracy_result = evaluate_model(model, tokenizer, val_data, DEVICE)
    
    # Save LoRA adapter
    print("\n9. Saving LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"   Saved to: {OUTPUT_DIR}")
    
    # Save metrics
    metrics = {
        "train_loss": train_metrics.get("train_loss", None),
        "train_runtime": train_metrics.get("train_runtime", None),
        "eval_loss": eval_result.get("eval_loss", None),
        "eval_accuracy": accuracy_result["accuracy"],
        "eval_correct": accuracy_result["correct"],
        "eval_total": accuracy_result["total"],
        "config": {
            "model_name": MODEL_NAME,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
        }
    }
    
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"   Metrics saved to: {METRICS_FILE}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA rank (r): {LORA_R}")
    print(f"LoRA alpha: {LORA_ALPHA}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Train loss: {train_metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Eval loss: {eval_result.get('eval_loss', 'N/A'):.4f}")
    print(f"Eval accuracy: {accuracy_result['accuracy']:.4f} ({accuracy_result['accuracy'] * 100:.2f}%)")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    return model, tokenizer, metrics


if __name__ == "__main__":
    train()
