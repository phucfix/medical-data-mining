"""
Module để load model Qwen2.5-0.5B-Instruct cho fine-tuning.
Sử dụng Hugging Face Transformers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Tên model trên Hugging Face Hub
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def get_device() -> str:
    """
    Xác định device để chạy model.
    
    Returns:
        str: "cuda" nếu có GPU, "cpu" nếu không.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model) -> int:
    """
    Đếm tổng số tham số của model.
    
    Args:
        model: Model cần đếm tham số.
        
    Returns:
        int: Tổng số tham số.
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    """
    Đếm số tham số có thể train được của model.
    
    Args:
        model: Model cần đếm tham số.
        
    Returns:
        int: Số tham số trainable.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_base_model(device: str = None):
    """
    Load model Qwen2.5-0.5B-Instruct cho fine-tuning.
    
    Args:
        device: Device để load model ("cuda" hoặc "cpu").
                Nếu None, tự động detect.
    
    Returns:
        model: Model đã được load.
    """
    if device is None:
        device = get_device()
    
    print(f"Loading model {MODEL_NAME}...")
    print(f"Device: {device}")
    
    # Load model cho Causal Language Modeling (fine-tuning)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # Nếu device là cpu, chuyển model sang cpu
    if device == "cpu":
        model = model.to(device)
    
    # Đếm và in số lượng tham số
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    
    print(f"\n{'='*50}")
    print(f"Model: {MODEL_NAME}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameters in billions: {total_params / 1e9:.3f}B")
    print(f"{'='*50}")
    
    # Confirm model < 1 tỷ tham số
    if total_params < 1e9:
        print("✓ Confirmed: Model có ít hơn 1 tỷ tham số (< 1B)")
    else:
        print("✗ Warning: Model có nhiều hơn 1 tỷ tham số (>= 1B)")
    
    return model


def load_tokenizer():
    """
    Load tokenizer cho model Qwen2.5-0.5B-Instruct.
    
    Returns:
        tokenizer: Tokenizer đã được load.
    """
    print(f"Loading tokenizer {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    # Đảm bảo có pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer loaded successfully!")
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    
    return tokenizer


def load_model_and_tokenizer(device: str = None):
    """
    Load cả model và tokenizer cho fine-tuning.
    
    Args:
        device: Device để load model ("cuda" hoặc "cpu").
                Nếu None, tự động detect.
    
    Returns:
        tuple: (model, tokenizer)
    """
    if device is None:
        device = get_device()
    
    print(f"\n{'#'*60}")
    print(f"# Loading Qwen2.5-0.5B-Instruct for Fine-tuning")
    print(f"{'#'*60}\n")
    
    # Load tokenizer trước
    tokenizer = load_tokenizer()
    
    print()  # Dòng trống
    
    # Load model
    model = load_base_model(device)
    
    print(f"\n{'#'*60}")
    print(f"# Model and Tokenizer loaded successfully!")
    print(f"{'#'*60}\n")
    
    return model, tokenizer


# Test khi chạy trực tiếp file này
if __name__ == "__main__":
    print("Testing model loading...")
    print()
    
    # Load model và tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Test tokenizer
    print("\n--- Test Tokenizer ---")
    test_text = "Triệu chứng của bệnh tiểu đường là gì?"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"Input text: {test_text}")
    print(f"Token IDs: {tokens['input_ids'].tolist()}")
    print(f"Number of tokens: {tokens['input_ids'].shape[1]}")
    
    # Test model inference (không training)
    print("\n--- Test Model Inference ---")
    device = get_device()
    
    # Chuyển input sang device của model
    if device == "cuda":
        tokens = {k: v.to(device) for k, v in tokens.items()}
    
    # Generate với model
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **tokens,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    
    print("\n✓ All tests passed!")
