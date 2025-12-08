"""
Script evaluate model v3 trên Test_sample.v1.0.csv
Test_sample.v1.0.csv được giữ hoàn toàn riêng biệt, KHÔNG merge vào training
"""

import csv
import json
import torch
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
TEST_FILE = BASE_DIR / "Test_sample.v1.0.csv"
LORA_MODEL_DIR = BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v3-augmented"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_FILE = BASE_DIR / "data" / "test_sample_v3_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 10


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_with_lora():
    """Load base model + LoRA adapter."""
    print(f"Loading base model: {BASE_MODEL_NAME}")
    print(f"Loading LoRA adapter from: {LORA_MODEL_DIR}")
    print(f"Device: {DEVICE}")
    
    # Load tokenizer from BASE MODEL (not LoRA dir)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,  # FIX: Load from base model, not LoRA dir
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, str(LORA_MODEL_DIR))
    model.eval()
    
    print("✓ Model loaded successfully!")
    return model, tokenizer


# ============================================================================
# PREDICTION
# ============================================================================

def create_prompt(input_text: str) -> str:
    """Tạo prompt cho model (giống lúc train)."""
    return f"Bạn là một trợ lý y tế. Hãy trả lời Đúng hoặc Sai.\nNhận định: {input_text}\nĐáp án:"


def extract_label(generated_text: str) -> str:
    """Trích xuất nhãn từ text được generate."""
    text = generated_text.upper().strip()
    
    # Tìm vị trí xuất hiện của TRUE/FALSE
    true_pos = float('inf')
    false_pos = float('inf')
    
    for keyword in ["TRUE", "ĐÚNG"]:
        pos = text.find(keyword)
        if pos != -1 and pos < true_pos:
            true_pos = pos
    
    for keyword in ["FALSE", "SAI"]:
        pos = text.find(keyword)
        if pos != -1 and pos < false_pos:
            false_pos = pos
    
    # Chọn label dựa trên từ xuất hiện đầu tiên
    if true_pos < false_pos:
        return "TRUE"
    elif false_pos < true_pos:
        return "FALSE"
    else:
        return "UNKNOWN"


def predict_single(model, tokenizer, input_text: str) -> tuple:
    """Dự đoán cho một mẫu."""
    prompt = create_prompt(input_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    predicted_label = extract_label(generated_text)
    return predicted_label, generated_text


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_on_test_sample():
    """Đánh giá model v3 trên Test_sample.v1.0.csv"""
    print("=" * 60)
    print("TESTING QWEN LORA V3 (AUGMENTED) ON Test_sample.v1.0.csv")
    print("=" * 60)
    print("⚠️  Test_sample.v1.0.csv is kept COMPLETELY SEPARATE")
    print("    NOT merged into training data")
    print("=" * 60)
    
    # Check files
    if not TEST_FILE.exists():
        print(f"ERROR: Test file not found: {TEST_FILE}")
        return None
    
    if not LORA_MODEL_DIR.exists():
        print(f"ERROR: Model v3 not found: {LORA_MODEL_DIR}")
        print("Please train model v3 first:")
        print("  python src/train_slm_qwen_lora_v3_augmented.py")
        return None
    
    # Load model
    model, tokenizer = load_model_with_lora()
    
    # Load test data
    print(f"\nLoading test data from: {TEST_FILE}")
    results = []
    
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Total samples: {len(rows)}")
    
    # Predict
    print("\nRunning predictions...")
    for i, row in enumerate(rows):
        text = row.get("Mệnh đề Câu hỏi (VIETNAMESE TEXT ONLY)", "").strip()
        true_label = row.get("Đáp án (TRUE/FALSE)", "").strip().upper()
        
        if not text or not true_label:
            continue
        
        pred_label, raw_output = predict_single(model, tokenizer, text)
        
        results.append({
            "input": text,
            "true_label": true_label,
            "pred_label": pred_label,
            "raw_output": raw_output,
            "correct": pred_label == true_label
        })
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(rows)} samples...")
    
    print(f"  Completed {len(results)} predictions")
    
    # Compute metrics
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    # Confusion matrix
    cm = Counter((r["true_label"], r["pred_label"]) for r in results)
    
    # Count unknown
    unknown_count = sum(1 for r in results if r["pred_label"] == "UNKNOWN")
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS ON Test_sample.v1.0.csv")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Unknown predictions: {unknown_count}")
    
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                  TRUE    FALSE   UNKNOWN")
    print(f"  Actual TRUE     {cm[('TRUE','TRUE')]:4d}    {cm[('TRUE','FALSE')]:4d}    {cm[('TRUE','UNKNOWN')]:4d}")
    print(f"  Actual FALSE    {cm[('FALSE','TRUE')]:4d}    {cm[('FALSE','FALSE')]:4d}    {cm[('FALSE','UNKNOWN')]:4d}")
    
    # Accuracy excluding UNKNOWN
    valid_results = [r for r in results if r["pred_label"] != "UNKNOWN"]
    if valid_results:
        valid_correct = sum(1 for r in valid_results if r["correct"])
        valid_accuracy = valid_correct / len(valid_results)
        print(f"\nAccuracy (excluding UNKNOWN): {valid_accuracy:.4f} ({valid_accuracy * 100:.2f}%)")
    
    # Sample predictions
    print("\n" + "-" * 60)
    print("SAMPLE PREDICTIONS (first 10)")
    print("-" * 60)
    for i, r in enumerate(results[:10]):
        status = "✓" if r["correct"] else "✗"
        print(f"\n{i+1}. {status}")
        print(f"   Input: {r['input'][:60]}...")
        print(f"   True: {r['true_label']}, Pred: {r['pred_label']}")
        print(f"   Raw output: {r['raw_output']}")
    
    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "model_version": "v3-augmented",
            "training_data": "augmented (105k samples)",
            "test_file": "Test_sample.v1.0.csv",
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "unknown_count": unknown_count,
            "confusion_matrix": {
                "TP": cm[("TRUE", "TRUE")],
                "FN": cm[("TRUE", "FALSE")],
                "FP": cm[("FALSE", "TRUE")],
                "TN": cm[("FALSE", "FALSE")],
                "TRUE_UNKNOWN": cm[("TRUE", "UNKNOWN")],
                "FALSE_UNKNOWN": cm[("FALSE", "UNKNOWN")]
            },
            "predictions": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {OUTPUT_FILE}")
    
    return results


if __name__ == "__main__":
    evaluate_on_test_sample()
