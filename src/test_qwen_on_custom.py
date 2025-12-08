"""
Script test model Qwen LoRA trên bộ test khách quan tự tạo
Các câu có cấu trúc hoàn toàn khác với dữ liệu train
"""

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
TEST_FILE = BASE_DIR / "data" / "custom_test_objective.jsonl"
LORA_MODEL_DIR = BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v2"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_FILE = BASE_DIR / "data" / "custom_test_results.json"

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
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(LORA_MODEL_DIR),
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
    
    print("Model loaded successfully!")
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
    if "TRUE" in text or "ĐÚNG" in text:
        return "TRUE"
    if "FALSE" in text or "SAI" in text:
        return "FALSE"
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

def evaluate_on_custom_test():
    """Đánh giá model trên bộ test khách quan."""
    print("=" * 60)
    print("TESTING QWEN LORA V2 ON CUSTOM OBJECTIVE TEST SET")
    print("=" * 60)
    
    # Check files
    if not TEST_FILE.exists():
        print(f"ERROR: Test file not found: {TEST_FILE}")
        return None
    
    if not LORA_MODEL_DIR.exists():
        print(f"ERROR: Model not found: {LORA_MODEL_DIR}")
        return None
    
    # Load model
    model, tokenizer = load_model_with_lora()
    
    # Load test data
    print(f"\nLoading test data from: {TEST_FILE}")
    samples = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Total samples: {len(samples)}")
    
    # Predict
    print("\nRunning predictions...")
    results = []
    
    for i, sample in enumerate(samples):
        input_text = sample["input"]
        true_label = sample["output"].strip().upper()
        
        pred_label, raw_output = predict_single(model, tokenizer, input_text)
        
        results.append({
            "input": input_text,
            "true_label": true_label,
            "pred_label": pred_label,
            "raw_output": raw_output,
            "correct": pred_label == true_label
        })
        
        # Progress
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")
    
    print(f"  Completed {len(results)} predictions")
    
    # Compute metrics
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    # Confusion matrix
    cm = Counter((r["true_label"], r["pred_label"]) for r in results)
    
    # Count by category
    unknown_count = sum(1 for r in results if r["pred_label"] == "UNKNOWN")
    true_correct = sum(1 for r in results if r["true_label"] == "TRUE" and r["correct"])
    false_correct = sum(1 for r in results if r["true_label"] == "FALSE" and r["correct"])
    true_total = sum(1 for r in results if r["true_label"] == "TRUE")
    false_total = sum(1 for r in results if r["true_label"] == "FALSE")
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS ON CUSTOM OBJECTIVE TEST")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Unknown predictions: {unknown_count}")
    
    print(f"\nTRUE statements: {true_correct}/{true_total} correct ({true_correct/true_total*100:.1f}%)")
    print(f"FALSE statements: {false_correct}/{false_total} correct ({false_correct/false_total*100:.1f}%)")
    
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
    
    # Show all predictions with status
    print("\n" + "-" * 60)
    print("ALL PREDICTIONS")
    print("-" * 60)
    for i, r in enumerate(results):
        status = "✓" if r["correct"] else "✗"
        print(f"\n{i+1}. {status} [{r['true_label']} → {r['pred_label']}]")
        print(f"   {r['input']}")
        if not r["correct"]:
            print(f"   Raw output: {r['raw_output']}")
    
    # Summary of wrong predictions
    wrong_predictions = [r for r in results if not r["correct"]]
    if wrong_predictions:
        print("\n" + "=" * 60)
        print(f"WRONG PREDICTIONS SUMMARY ({len(wrong_predictions)} errors)")
        print("=" * 60)
        for i, r in enumerate(wrong_predictions):
            print(f"\n{i+1}. Expected: {r['true_label']}, Got: {r['pred_label']}")
            print(f"   {r['input']}")
    
    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "unknown_count": unknown_count,
            "true_accuracy": true_correct / true_total if true_total > 0 else 0,
            "false_accuracy": false_correct / false_total if false_total > 0 else 0,
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
    evaluate_on_custom_test()
