"""
Script để tương tác với model Qwen2.5-0.5B + LoRA đã train.
Cho phép nhập câu hỏi và xem model trả lời TRUE/FALSE.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
LORA_MODEL_DIR = BASE_DIR / "models" / "qwen2.5-0.5b-med-slm-lora-v2"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 150  # Tăng lên để có đủ chỗ cho giải thích


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load model với LoRA adapter."""
    print("=" * 50)
    print("Loading Medical QA Model...")
    print("=" * 50)
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"LoRA adapter: {LORA_MODEL_DIR}")
    print(f"Device: {DEVICE}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(LORA_MODEL_DIR),
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
    model = PeftModel.from_pretrained(base_model, str(LORA_MODEL_DIR))
    model.eval()
    
    print("\n✓ Model loaded successfully!\n")
    return model, tokenizer


def predict(model, tokenizer, statement: str) -> dict:
    """
    Dự đoán một mệnh đề là TRUE hay FALSE.
    
    Args:
        model: Model đã load
        tokenizer: Tokenizer
        statement: Mệnh đề y khoa cần kiểm tra
        
    Returns:
        dict với prediction và raw output
    """
    # Tạo prompt giống lúc train
    prompt = f"Bạn là một trợ lý y tế. Hãy trả lời Đúng hoặc Sai.\nNhận định: {statement}\nĐáp án:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    # Extract label - kiểm tra từ đầu tiên xuất hiện
    generated_upper = generated.upper()
    
    # Tìm vị trí xuất hiện của các từ khóa
    true_pos = float('inf')
    false_pos = float('inf')
    
    for keyword in ["TRUE", "ĐÚNG"]:
        pos = generated_upper.find(keyword)
        if pos != -1 and pos < true_pos:
            true_pos = pos
    
    for keyword in ["FALSE", "SAI"]:
        pos = generated_upper.find(keyword)
        if pos != -1 and pos < false_pos:
            false_pos = pos
    
    # Chọn label dựa trên từ xuất hiện đầu tiên
    if true_pos < false_pos:
        label = "TRUE"
        verdict = "✓ ĐÚNG"
    elif false_pos < true_pos:
        label = "FALSE"
        verdict = "✗ SAI"
    else:
        label = "UNKNOWN"
        verdict = "? KHÔNG XÁC ĐỊNH"
    
    return {
        "statement": statement,
        "prediction": label,
        "verdict": verdict,
        "raw_output": generated
    }


def interactive_mode(model, tokenizer):
    """Chế độ tương tác với người dùng."""
    print("=" * 50)
    print("MEDICAL TRUE/FALSE QA - Interactive Mode")
    print("=" * 50)
    print("Nhập một mệnh đề y khoa để kiểm tra đúng/sai.")
    print("Gõ 'quit' hoặc 'exit' để thoát.")
    print("Gõ 'demo' để xem các ví dụ mẫu.")
    print("=" * 50)
    
    while True:
        print()
        user_input = input("📝 Nhập mệnh đề: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Tạm biệt!")
            break
        
        if user_input.lower() == "demo":
            run_demo(model, tokenizer)
            continue
        
        # Predict
        result = predict(model, tokenizer, user_input)
        
        print(f"\n{'─' * 40}")
        print(f"📋 Mệnh đề: {result['statement']}")
        print(f"🤖 Kết quả: {result['verdict']}")
        print(f"📄 Raw output: {result['raw_output']}")
        print(f"{'─' * 40}")


def run_demo(model, tokenizer):
    """Chạy demo với các ví dụ mẫu."""
    examples = [
        "Tiểu đường là bệnh do thiếu insulin hoặc kháng insulin.",
        "Uống nhiều nước có thể chữa khỏi ung thư.",
        "Huyết áp cao có thể gây đột quỵ.",
        "Kháng sinh có thể điều trị được bệnh cúm do virus.",
        "Vitamin C giúp tăng cường hệ miễn dịch.",
        "Sốt là triệu chứng phổ biến của nhiễm trùng.",
        "Uống bia mỗi ngày tốt cho tim mạch.",
        "Tiêm vaccine có thể gây tự kỷ ở trẻ em.",
    ]
    
    print("\n" + "=" * 50)
    print("DEMO - Các ví dụ mẫu")
    print("=" * 50)
    
    for i, statement in enumerate(examples, 1):
        result = predict(model, tokenizer, statement)
        print(f"\n{i}. {statement}")
        print(f"   → {result['verdict']} (raw: {result['raw_output']})")
    
    print("\n" + "=" * 50)


def main():
    """Hàm main."""
    # Load model
    model, tokenizer = load_model()
    
    # Chạy interactive mode
    interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
