"""
Script evaluate model v4 trên Test_sample.v1.0.csv
Support multiple model versions: v2, v3, v4-quick, v4-ultra-quick, v4-chunked
"""

import csv
import json
import torch
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
TEST_FILE = BASE_DIR / "Test_sample.v1.0.csv"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Available model versions
MODEL_VERSIONS = {
    "v2": BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v2",
    "v3": BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v3-augmented",
    "v4-quick": BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v4-quick",
    "v4-ultra-quick": BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v4-ultra-quick",
    "v4-chunked": BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v4-chunked",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 10


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_with_lora(model_dir):
    """Load base model + LoRA adapter."""
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_dir}")
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")
    
    # Load tokenizer from BASE MODEL (more reliable)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,  # Load from base model, not LoRA dir
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    model.eval()
    
    print(f"✓ Model loaded successfully!\n")
    return model, tokenizer


# ============================================================================
# PREDICTION
# ============================================================================

def create_prompt(input_text: str) -> str:
    """Tạo prompt giống training."""
    return f"Bạn là một trợ lý y tế. Hãy trả lời Đúng hoặc Sai.\nNhận định: {input_text}\nĐáp án:"


def predict(model, tokenizer, input_text: str) -> str:
    """Predict TRUE or FALSE for given input."""
    prompt = create_prompt(input_text)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    if DEVICE == "cuda":
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].strip().upper()
    
    # Normalize answer
    if "ĐÚNG" in answer or "TRUE" in answer or answer.startswith("Đ"):
        return "TRUE"
    elif "SAI" in answer or "FALSE" in answer or answer.startswith("S"):
        return "FALSE"
    else:
        # Default: extract first word
        first_word = answer.split()[0] if answer else ""
        if first_word in ["ĐÚNG", "TRUE", "Đ"]:
            return "TRUE"
        elif first_word in ["SAI", "FALSE", "S"]:
            return "FALSE"
        else:
            return "TRUE"  # Default


# ============================================================================
# EVALUATION
# ============================================================================

def load_test_data():
    """Load Test_sample.v1.0.csv."""
    print(f"Loading test data from: {TEST_FILE}")
    
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    
    data = []
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different column names
            if 'input' in row:
                input_text = row['input']
                output_text = row['output']
            elif 'Mệnh đề Câu hỏi (VIETNAMESE TEXT ONLY)' in row:
                input_text = row['Mệnh đề Câu hỏi (VIETNAMESE TEXT ONLY)']
                output_text = row['Đáp án (TRUE/FALSE)']
            else:
                raise ValueError(f"Unknown CSV format. Columns: {list(row.keys())}")
            
            data.append({
                'input': input_text,
                'output': output_text
            })
    
    print(f"✓ Loaded {len(data)} test samples\n")
    return data


def evaluate(model, tokenizer, test_data, model_version="v4"):
    """Evaluate model on test data."""
    print(f"{'='*60}")
    print(f"EVALUATING MODEL {model_version.upper()}")
    print(f"{'='*60}\n")
    
    results = []
    correct = 0
    total = len(test_data)
    
    print(f"Processing {total} samples...")
    print(f"Progress: ", end="", flush=True)
    
    for i, item in enumerate(test_data):
        input_text = item['input']
        expected = item['output']
        
        # Predict
        predicted = predict(model, tokenizer, input_text)
        
        # Check
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        
        results.append({
            'input': input_text,
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct
        })
        
        # Progress bar
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{total}", end=" ", flush=True)
    
    print(f"\n\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    
    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%")
    
    # Analyze predictions
    pred_counter = Counter([r['predicted'] for r in results])
    expected_counter = Counter([r['expected'] for r in results])
    
    print(f"\nPrediction distribution:")
    print(f"  TRUE:  {pred_counter['TRUE']}")
    print(f"  FALSE: {pred_counter['FALSE']}")
    
    print(f"\nExpected distribution:")
    print(f"  TRUE:  {expected_counter['TRUE']}")
    print(f"  FALSE: {expected_counter['FALSE']}")
    
    # Confusion matrix
    tp = sum(1 for r in results if r['predicted'] == 'TRUE' and r['expected'] == 'TRUE')
    fp = sum(1 for r in results if r['predicted'] == 'TRUE' and r['expected'] == 'FALSE')
    tn = sum(1 for r in results if r['predicted'] == 'FALSE' and r['expected'] == 'FALSE')
    fn = sum(1 for r in results if r['predicted'] == 'FALSE' and r['expected'] == 'TRUE')
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Negatives: {fn}")
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return {
        'model_version': model_version,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        },
        'results': results
    }


def save_results(evaluation_results, model_version):
    """Save evaluation results to file."""
    output_file = BASE_DIR / "data" / f"test_sample_{model_version}_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen model on Test_sample.v1.0.csv")
    parser.add_argument(
        '--version',
        type=str,
        default='v4-ultra-quick',
        choices=list(MODEL_VERSIONS.keys()),
        help='Model version to evaluate'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Custom model path (overrides --version)'
    )
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model_path:
        model_dir = Path(args.model_path)
        model_version = "custom"
    else:
        model_dir = MODEL_VERSIONS[args.version]
        model_version = args.version
    
    # Check if model exists
    if not model_dir.exists():
        print(f"\n❌ ERROR: Model not found at {model_dir}")
        print(f"\nAvailable models:")
        for version, path in MODEL_VERSIONS.items():
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {version}: {path}")
        return
    
    # Load test data
    test_data = load_test_data()
    
    # Load model
    model, tokenizer = load_model_with_lora(model_dir)
    
    # Evaluate
    results = evaluate(model, tokenizer, test_data, model_version)
    
    # Save results
    save_results(results, model_version)
    
    print(f"\n{'='*60}")
    print(f"✓ EVALUATION COMPLETED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
