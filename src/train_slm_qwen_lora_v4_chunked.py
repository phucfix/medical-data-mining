"""
CHUNKED TRAINING - Train full 154k samples WITHOUT OOM
Strategy: Chia data thành chunks, train từng chunk, kế thừa weights
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List
import shutil

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np

# ============================================================================
# CHUNKED TRAINING CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "slm_train_style_adapted.jsonl"
VAL_FILE = DATA_DIR / "slm_val.jsonl"

OUTPUT_DIR = BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v4-chunked"
TEMP_DIR = BASE_DIR / "models" / "temp_chunks"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ============================================================================
# CHUNKED CONFIG - Train từng 30k samples một lần
# ============================================================================
CHUNK_SIZE = 30000          # Mỗi chunk 30k samples
NUM_EPOCHS_PER_CHUNK = 1    # 1 epoch mỗi chunk
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-5
LOGGING_STEPS = 100
MAX_LENGTH = 256
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4

# LoRA Config
LORA_R = 32
LORA_ALPHA = 64
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


def train_chunk(
    chunk_data: List[Dict],
    chunk_idx: int,
    total_chunks: int,
    tokenizer,
    previous_model_path: str = None
):
    """Train một chunk data."""
    
    print("\n" + "=" * 80)
    print(f"TRAINING CHUNK {chunk_idx + 1}/{total_chunks}")
    print(f"Samples in this chunk: {len(chunk_data)}")
    print("=" * 80)
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load base model hoặc model từ chunk trước
    if previous_model_path and os.path.exists(previous_model_path):
        print(f"Loading model from previous chunk: {previous_model_path}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
        )
        
        # Load LoRA weights từ chunk trước - SET IS_TRAINABLE=True
        model = PeftModel.from_pretrained(
            base_model, 
            previous_model_path,
            is_trainable=True  # FIX: Enable training on loaded weights
        )
        print("✓ Loaded LoRA weights from previous chunk")
    else:
        print(f"Loading fresh base model: {MODEL_NAME}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        
        model = get_peft_model(base_model, lora_config)
        print("✓ Initialized fresh LoRA model")
    
    # Enable gradient checkpointing
    model.enable_input_require_grads()
    if hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()
    
    # Create dataset
    train_dataset = MedicalQADataset(chunk_data, tokenizer, MAX_LENGTH)
    
    # Output path cho chunk này
    chunk_output_dir = TEMP_DIR / f"chunk_{chunk_idx}"
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(chunk_output_dir),
        num_train_epochs=NUM_EPOCHS_PER_CHUNK,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",  # Không save intermediate checkpoints
        eval_strategy="no",   # Không eval
        fp16=False,  # FIX: Disable fp16 to avoid mixed precision issues with chunked training
        bf16=DEVICE == "cuda",  # Use bf16 instead (more stable)
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )
    
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs for this chunk: {NUM_EPOCHS_PER_CHUNK}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n▶ Training chunk {chunk_idx + 1}...")
    trainer.train()
    
    # Save chunk model
    print(f"✓ Saving chunk {chunk_idx + 1} model...")
    trainer.save_model(str(chunk_output_dir))
    
    # Clear memory
    del model, trainer, train_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return str(chunk_output_dir)


def train_chunked():
    """Main chunked training function."""
    
    print("=" * 80)
    print("CHUNKED TRAINING - TRAIN FULL 154K SAMPLES WITHOUT OOM")
    print("=" * 80)
    print("Strategy: Split into 30k chunks, train sequentially, accumulate knowledge")
    print("=" * 80)
    
    # Set environment
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Check GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load full training data
    print(f"\nLoading full training data from {TRAIN_FILE}...")
    train_data_full = load_jsonl(TRAIN_FILE)
    print(f"✓ Total samples: {len(train_data_full)}")
    
    # Shuffle data
    print("\nShuffling data...")
    random.seed(42)
    random.shuffle(train_data_full)
    
    # Split into chunks
    num_chunks = (len(train_data_full) + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunks = []
    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, len(train_data_full))
        chunks.append(train_data_full[start_idx:end_idx])
    
    print(f"\n✓ Split into {num_chunks} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} samples")
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Train each chunk
    previous_model_path = None
    
    for i, chunk_data in enumerate(chunks):
        chunk_path = train_chunk(
            chunk_data=chunk_data,
            chunk_idx=i,
            total_chunks=num_chunks,
            tokenizer=tokenizer,
            previous_model_path=previous_model_path
        )
        previous_model_path = chunk_path
        
        print(f"\n✓ Chunk {i+1}/{num_chunks} completed")
        print(f"  Progress: {(i+1)/num_chunks*100:.1f}%")
    
    # Copy final model
    print("\n" + "=" * 80)
    print("FINALIZING MODEL")
    print("=" * 80)
    
    print(f"\nCopying final model to {OUTPUT_DIR}...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    shutil.copytree(previous_model_path, OUTPUT_DIR)
    
    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Clean up temp chunks (optional)
    print("\nCleaning up temporary chunks...")
    for i in range(num_chunks - 1):  # Keep last chunk as backup
        chunk_dir = TEMP_DIR / f"chunk_{i}"
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
    
    # Save metrics
    metrics = {
        "total_samples": len(train_data_full),
        "num_chunks": num_chunks,
        "chunk_size": CHUNK_SIZE,
        "epochs_per_chunk": NUM_EPOCHS_PER_CHUNK,
        "total_training_passes": num_chunks * NUM_EPOCHS_PER_CHUNK,
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "training_strategy": "chunked-full-dataset",
        "note": f"Trained on full {len(train_data_full)} samples via {num_chunks} chunks"
    }
    
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓✓ CHUNKED TRAINING COMPLETED ✓✓")
    print("=" * 80)
    print(f"Final model: {OUTPUT_DIR}")
    print(f"Total samples trained: {len(train_data_full)}")
    print(f"Number of chunks: {num_chunks}")
    print(f"Expected accuracy: 85-90% (full dataset)")
    print(f"\nEvaluate with: python src/test_qwen_on_sample_v3.py")
    print(f"Model path: {OUTPUT_DIR}")
    
    # Clear final cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train_chunked()
