"""
Script đánh giá model Qwen2.5-0.5B-Instruct + LoRA trên test_dev set.
Tính accuracy, confusion matrix và lưu kết quả.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TEST_DEV_FILE = DATA_DIR / "slm_test_dev.jsonl"
LORA_MODEL_DIR = BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora"
OUTPUT_METRICS_FILE = DATA_DIR / "metrics_eval_test_dev.json"

# Model
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Generation config
MAX_NEW_TOKENS = 10
MAX_LENGTH = 256

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# DATA LOADING
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
    """Tạo prompt cho model (giống lúc train)."""
    return f"Bạn là một trợ lý y tế. Hãy trả lời Đúng hoặc Sai.\nNhận định: {input_text}\nĐáp án:"


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_with_lora():
    """Load base model + LoRA adapter."""
    print(f"Loading base model: {BASE_MODEL_NAME}")
    print(f"Loading LoRA adapter from: {LORA_MODEL_DIR}")
    print(f"Device: {DEVICE}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(LORA_MODEL_DIR),
        trust_remote_code=True
    )
    
    # Đảm bảo có pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(LORA_MODEL_DIR))
    model.eval()
    
    # Print info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model loaded successfully!")
    
    return model, tokenizer


# ============================================================================
# PREDICTION
# ============================================================================

def extract_label(generated_text: str) -> str:
    """
    Trích xuất nhãn từ text được generate.
    
    Args:
        generated_text: Text được model generate.
        
    Returns:
        str: "TRUE" hoặc "FALSE" hoặc "UNKNOWN"
    """
    text = generated_text.upper().strip()
    
    # Check for TRUE variations
    if "TRUE" in text or "ĐÚNG" in text.upper():
        return "TRUE"
    
    # Check for FALSE variations
    if "FALSE" in text or "SAI" in text.upper():
        return "FALSE"
    
    return "UNKNOWN"


def predict_single(
    model,
    tokenizer,
    input_text: str,
    device: str
) -> Tuple[str, str]:
    """
    Dự đoán cho một mẫu.
    
    Args:
        model: Model đã load.
        tokenizer: Tokenizer.
        input_text: Text đầu vào.
        device: Device.
        
    Returns:
        Tuple[str, str]: (predicted_label, raw_generated_text)
    """
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
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    # Extract label
    predicted_label = extract_label(generated_text)
    
    return predicted_label, generated_text


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_confusion_matrix(
    predictions: List[str],
    labels: List[str]
) -> Dict:
    """
    Tính confusion matrix.
    
    Positive class = TRUE
    Negative class = FALSE
    
    Returns:
        Dict với TP, FP, TN, FN
    """
    tp = fp = tn = fn = 0
    
    for pred, label in zip(predictions, labels):
        if label == "TRUE":
            if pred == "TRUE":
                tp += 1
            else:
                fn += 1
        else:  # label == "FALSE"
            if pred == "FALSE":
                tn += 1
            else:
                fp += 1
    
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }


def compute_metrics(
    predictions: List[str],
    labels: List[str]
) -> Dict:
    """
    Tính các metrics đánh giá.
    
    Returns:
        Dict chứa accuracy, precision, recall, f1, confusion matrix
    """
    # Accuracy
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    total = len(labels)
    accuracy = correct / total if total > 0 else 0
    
    # Confusion matrix
    cm = compute_confusion_matrix(predictions, labels)
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    
    # Precision, Recall, F1 (cho class TRUE)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "total": total,
        "confusion_matrix": cm
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate():
    """Hàm chính để đánh giá model."""
    print("=" * 60)
    print("EVALUATING QWEN2.5-0.5B + LORA ON TEST_DEV SET")
    print("=" * 60)
    
    # 1. Load model
    print("\n1. Loading model with LoRA adapter...")
    model, tokenizer = load_model_with_lora()
    
    # 2. Load test data
    print("\n2. Loading test_dev data...")
    test_data = load_jsonl(TEST_DEV_FILE)
    print(f"   Test samples: {len(test_data)}")
    
    # 3. Predict
    print("\n3. Running predictions...")
    predictions = []
    labels = []
    raw_outputs = []
    
    for i, item in enumerate(test_data):
        input_text = item["input"]
        true_label = item["output"].upper()
        
        pred_label, raw_output = predict_single(model, tokenizer, input_text, DEVICE)
        
        predictions.append(pred_label)
        labels.append(true_label)
        raw_outputs.append(raw_output)
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(test_data)} samples...")
    
    print(f"   Completed {len(test_data)} predictions")
    
    # 4. Compute metrics
    print("\n4. Computing metrics...")
    metrics = compute_metrics(predictions, labels)
    
    # 5. Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    print(f"\nTotal samples: {metrics['total']}")
    print(f"Correct predictions: {metrics['correct']}")
    
    cm = metrics["confusion_matrix"]
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  TRUE    FALSE")
    print(f"  Actual TRUE     {cm['TP']:4d}    {cm['FN']:4d}")
    print(f"  Actual FALSE    {cm['FP']:4d}    {cm['TN']:4d}")
    
    # Count unknown predictions
    unknown_count = sum(1 for p in predictions if p == "UNKNOWN")
    if unknown_count > 0:
        print(f"\nWarning: {unknown_count} predictions were UNKNOWN")
    
    # 6. Show sample predictions
    print("\n" + "-" * 60)
    print("SAMPLE PREDICTIONS (first 10)")
    print("-" * 60)
    
    for i in range(min(10, len(test_data))):
        input_text = test_data[i]["input"][:60] + "..."
        true_label = labels[i]
        pred_label = predictions[i]
        raw_output = raw_outputs[i]
        
        status = "✓" if pred_label == true_label else "✗"
        print(f"\n{i+1}. {status}")
        print(f"   Input: {input_text}")
        print(f"   True: {true_label}, Pred: {pred_label}")
        print(f"   Raw output: {raw_output}")
    
    # 7. Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Prepare output data
    output_data = {
        "model": {
            "base_model": BASE_MODEL_NAME,
            "lora_adapter": str(LORA_MODEL_DIR)
        },
        "test_file": str(TEST_DEV_FILE),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "correct": metrics["correct"],
            "total": metrics["total"]
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "unknown_predictions": unknown_count,
        "sample_predictions": [
            {
                "input": test_data[i]["input"][:100],
                "true_label": labels[i],
                "predicted_label": predictions[i],
                "raw_output": raw_outputs[i],
                "correct": predictions[i] == labels[i]
            }
            for i in range(min(20, len(test_data)))
        ]
    }
    
    # Save to JSON
    OUTPUT_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {OUTPUT_METRICS_FILE}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {BASE_MODEL_NAME} + LoRA")
    print(f"Test samples: {metrics['total']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    evaluate()
