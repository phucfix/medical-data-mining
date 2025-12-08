"""
Data Augmentation cho Medical QA Dataset
Tạo thêm training samples bằng các kỹ thuật NLP
"""

import json
import random
from pathlib import Path
from typing import List, Dict
# from googletrans import Translator  # Optional, comment out if not needed
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data" / "slm_train.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "slm_train_augmented.jsonl"

# Augmentation config
AUGMENTATION_RATIO = 2.0  # Tạo thêm 2x dữ liệu gốc
METHODS = ["paraphrase", "synonym_replace", "add_context"]  # Skip back_translation

# translator = Translator()  # Optional, not used in lite version


# ============================================================================
# AUGMENTATION METHODS
# ============================================================================

def back_translate(text: str, source='vi', intermediate='en') -> str:
    """
    Dịch Vietnamese → English → Vietnamese để tạo variation.
    NOTE: Requires googletrans package. Skipped in lite version.
    """
    # Skip back-translation if googletrans not available
    return text
    
    # Uncomment below if you have googletrans installed
    # try:
    #     from googletrans import Translator
    #     translator = Translator()
    #     # VN → EN
    #     translated = translator.translate(text, src=source, dest=intermediate)
    #     time.sleep(0.1)  # Rate limiting
    #     
    #     # EN → VN
    #     back_translated = translator.translate(translated.text, src=intermediate, dest=source)
    #     time.sleep(0.1)
    #     
    #     return back_translated.text
    # except Exception as e:
    #     print(f"Error in back_translate: {e}")
    #     return text


def paraphrase_simple(text: str) -> str:
    """
    Paraphrase đơn giản bằng cách thay đổi cấu trúc câu.
    """
    # Pattern 1: "A là B" → "B là đặc điểm của A"
    if " là " in text:
        parts = text.split(" là ", 1)
        if len(parts) == 2:
            return f"{parts[1]} là đặc điểm của {parts[0]}"
    
    # Pattern 2: "A có B" → "B thuộc về A"
    if " có " in text:
        parts = text.split(" có ", 1)
        if len(parts) == 2:
            return f"{parts[1]} thuộc về {parts[0]}"
    
    # Pattern 3: "A gây ra B" → "B do A gây ra"
    if " gây ra " in text:
        parts = text.split(" gây ra ", 1)
        if len(parts) == 2:
            return f"{parts[1]} do {parts[0]} gây ra"
    
    return text


def synonym_replace(text: str) -> str:
    """
    Thay thế từ bằng từ đồng nghĩa trong y tế.
    """
    # Dictionary đồng nghĩa y tế tiếng Việt
    synonyms = {
        "bệnh": ["tình trạng", "chứng bệnh", "căn bệnh"],
        "triệu chứng": ["dấu hiệu", "biểu hiện", "triệu trứng"],
        "thuốc": ["dược phẩm", "hợp chất", "chế phẩm"],
        "điều trị": ["chữa trị", "trị liệu", "xử lý"],
        "gây ra": ["dẫn đến", "tạo ra", "sinh ra", "gây nên"],
        "sản xuất": ["tiết ra", "tạo ra", "sinh ra"],
        "cơ quan": ["bộ phận", "tổ chức"],
        "có thể": ["có khả năng", "được phép", "dễ dàng"],
        "không thể": ["không có khả năng", "bất khả thi"],
        "quan trọng": ["cần thiết", "thiết yếu", "chủ yếu"],
    }
    
    result = text
    for word, replacements in synonyms.items():
        if word in result:
            # Chọn ngẫu nhiên 1 từ thay thế
            replacement = random.choice(replacements)
            result = result.replace(word, replacement, 1)  # Chỉ thay 1 lần
            break  # Chỉ thay 1 từ mỗi lần
    
    return result


def add_context(text: str) -> str:
    """
    Thêm context y tế vào câu.
    """
    prefixes = [
        "Trong y học, ",
        "Theo y khoa, ",
        "Về mặt y tế, ",
        "Dựa trên kiến thức y học, ",
    ]
    
    return random.choice(prefixes) + text.lower()


def remove_redundant_words(text: str) -> str:
    """
    Loại bỏ từ thừa để tạo câu ngắn gọn hơn.
    """
    # Loại bỏ các từ không cần thiết
    redundant = ["rằng", "là việc", "có thể", "thường", "thông thường"]
    
    result = text
    for word in redundant:
        if word in result:
            result = result.replace(f" {word} ", " ", 1)
    
    return result.strip()


def negate_statement(text: str) -> tuple:
    """
    Tạo phủ định của câu và flip label.
    Returns: (negated_text, flipped_label)
    """
    if "không" in text:
        # Bỏ "không" để tạo positive
        negated = text.replace(" không ", " ", 1)
        return negated, "flip"
    else:
        # Thêm "không" để tạo negative
        # Tìm động từ và thêm "không" trước đó
        verbs = ["là", "có", "được", "gây", "tiết", "điều"]
        for verb in verbs:
            if f" {verb} " in text:
                negated = text.replace(f" {verb} ", f" không {verb} ", 1)
                return negated, "flip"
    
    return text, "no_flip"


# ============================================================================
# AUGMENTATION PIPELINE
# ============================================================================

def augment_single_sample(sample: Dict, num_augmentations: int = 2) -> List[Dict]:
    """
    Tạo nhiều augmented versions của 1 sample.
    """
    text = sample["input"]
    label = sample["output"]
    
    augmented_samples = [sample]  # Bao gồm cả bản gốc
    
    methods = [
        ("back_translate", back_translate),
        ("paraphrase", paraphrase_simple),
        ("synonym", synonym_replace),
        ("add_context", add_context),
        ("remove_redundant", remove_redundant_words),
    ]
    
    # Random chọn methods để apply
    selected_methods = random.sample(methods, min(num_augmentations, len(methods)))
    
    for method_name, method_func in selected_methods:
        try:
            if method_name == "negate":
                aug_text, flip_status = negate_statement(text)
                aug_label = "FALSE" if label == "TRUE" else "TRUE" if flip_status == "flip" else label
            else:
                aug_text = method_func(text)
                aug_label = label
            
            # Kiểm tra xem có khác gốc không
            if aug_text != text and len(aug_text) > 20:
                augmented_samples.append({
                    "input": aug_text,
                    "output": aug_label,
                    "augmentation_method": method_name
                })
        except Exception as e:
            print(f"Error in {method_name}: {e}")
            continue
    
    return augmented_samples


def augment_dataset(
    input_file: Path,
    output_file: Path,
    augmentation_ratio: float = 2.0
):
    """
    Augment toàn bộ dataset.
    """
    print("=" * 60)
    print("DATA AUGMENTATION FOR MEDICAL QA")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {input_file}...")
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Original samples: {len(samples)}")
    
    # Calculate số lượng augmentations cần tạo
    num_augmentations_per_sample = int(augmentation_ratio)
    print(f"Augmentations per sample: {num_augmentations_per_sample}")
    print(f"Expected total: {len(samples) * (1 + num_augmentations_per_sample)}")
    
    # Augment
    print("\nAugmenting dataset...")
    all_augmented = []
    
    for i, sample in enumerate(samples):
        augmented = augment_single_sample(sample, num_augmentations_per_sample)
        all_augmented.extend(augmented)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")
    
    # Shuffle
    random.shuffle(all_augmented)
    
    # Save
    print(f"\nSaving augmented data to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_augmented:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # Statistics
    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    print(f"Original samples: {len(samples)}")
    print(f"Augmented samples: {len(all_augmented)}")
    print(f"Increase: {len(all_augmented) - len(samples)} (+{(len(all_augmented)/len(samples) - 1)*100:.1f}%)")
    
    # Count by augmentation method
    method_counts = {}
    for sample in all_augmented:
        method = sample.get("augmentation_method", "original")
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print("\nAugmentation methods distribution:")
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method}: {count} ({count/len(all_augmented)*100:.1f}%)")
    
    print(f"\n✓ Augmentation completed!")
    print(f"Output file: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Demo mode: augment 1 sample
        print("DEMO: Augmenting a single sample\n")
        
        demo_sample = {
            "input": "Insulin được sản xuất bởi tuyến tụy.",
            "output": "TRUE"
        }
        
        print(f"Original: {demo_sample['input']}")
        print(f"Label: {demo_sample['output']}\n")
        
        augmented = augment_single_sample(demo_sample, num_augmentations=5)
        
        print("Augmented versions:")
        for i, sample in enumerate(augmented[1:], 1):  # Skip original
            print(f"{i}. {sample['input']}")
            print(f"   Method: {sample.get('augmentation_method', 'N/A')}")
            print(f"   Label: {sample['output']}\n")
    else:
        # Full augmentation mode
        augment_dataset(INPUT_FILE, OUTPUT_FILE, AUGMENTATION_RATIO)
