"""
Script merge Test_sample.v1.0.csv vào tập train để cải thiện generalization.
Lấy 50% Test_sample để thêm vào train, giữ 50% còn lại để test.
"""

import json
import csv
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Input files
ORIGINAL_TRAIN_FILE = DATA_DIR / "slm_train.jsonl"
ORIGINAL_VAL_FILE = DATA_DIR / "slm_val.jsonl"
TEST_SAMPLE_FILE = BASE_DIR / "Test_sample.v1.0.csv"

# Output files
MERGED_TRAIN_FILE = DATA_DIR / "slm_train_merged.jsonl"
MERGED_VAL_FILE = DATA_DIR / "slm_val_merged.jsonl"
NEW_TEST_FILE = DATA_DIR / "slm_test_sample_held_out.jsonl"

# Config
TEST_SAMPLE_TRAIN_RATIO = 0.5  # 50% của Test_sample sẽ thêm vào train
RANDOM_STATE = 42

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_jsonl(file_path: Path) -> list:
    """Đọc file JSONL."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, file_path: Path):
    """Lưu file JSONL."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {file_path}")


def load_test_sample_csv(file_path: Path) -> list:
    """Đọc Test_sample.v1.0.csv và chuyển sang format đơn giản (không có prefix)."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("Mệnh đề Câu hỏi (VIETNAMESE TEXT ONLY)", "").strip()
            label = row.get("Đáp án (TRUE/FALSE)", "").strip().upper()
            
            if not text or not label or len(text) < 20:
                continue
            
            if label not in ["TRUE", "FALSE"]:
                continue
            
            # Format đơn giản: chỉ có text gốc, không thêm prefix
            data.append({
                "input": text,
                "output": label
            })
    
    return data


def merge_datasets():
    """Merge Test_sample vào tập train."""
    print("=" * 60)
    print("MERGING DATASETS FOR BETTER GENERALIZATION")
    print("=" * 60)
    
    # 1. Load original train/val
    print("\n1. Loading original train/val...")
    original_train = load_jsonl(ORIGINAL_TRAIN_FILE)
    original_val = load_jsonl(ORIGINAL_VAL_FILE)
    print(f"   Original train: {len(original_train)} samples")
    print(f"   Original val: {len(original_val)} samples")
    
    # 2. Load Test_sample.v1.0.csv
    print("\n2. Loading Test_sample.v1.0.csv...")
    test_sample_data = load_test_sample_csv(TEST_SAMPLE_FILE)
    print(f"   Test_sample: {len(test_sample_data)} samples")
    
    # 3. Split Test_sample: 50% train, 50% held-out test
    print(f"\n3. Splitting Test_sample ({TEST_SAMPLE_TRAIN_RATIO*100:.0f}% train, {(1-TEST_SAMPLE_TRAIN_RATIO)*100:.0f}% held-out test)...")
    
    # Lấy labels để stratify
    labels = [item["output"] for item in test_sample_data]
    
    test_sample_train, test_sample_held_out = train_test_split(
        test_sample_data,
        test_size=(1 - TEST_SAMPLE_TRAIN_RATIO),
        random_state=RANDOM_STATE,
        stratify=labels
    )
    print(f"   Test_sample for train: {len(test_sample_train)} samples")
    print(f"   Test_sample held-out: {len(test_sample_held_out)} samples")
    
    # 4. Merge train
    print("\n4. Merging datasets...")
    merged_train = original_train + test_sample_train
    random.seed(RANDOM_STATE)
    random.shuffle(merged_train)
    print(f"   Merged train: {len(merged_train)} samples")
    
    # Val giữ nguyên hoặc thêm một ít từ Test_sample
    merged_val = original_val
    print(f"   Merged val: {len(merged_val)} samples")
    
    # 5. Save files
    print("\n5. Saving merged files...")
    save_jsonl(merged_train, MERGED_TRAIN_FILE)
    save_jsonl(merged_val, MERGED_VAL_FILE)
    save_jsonl(test_sample_held_out, NEW_TEST_FILE)
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original train: {len(original_train)} → Merged train: {len(merged_train)}")
    print(f"Original val: {len(original_val)} → Merged val: {len(merged_val)}")
    print(f"New held-out test (from Test_sample): {len(test_sample_held_out)}")
    print(f"\nOutput files:")
    print(f"  - {MERGED_TRAIN_FILE}")
    print(f"  - {MERGED_VAL_FILE}")
    print(f"  - {NEW_TEST_FILE}")
    print("\n✓ Dataset merging completed!")
    print("\nNext step: Run train_slm_qwen_lora_v2.py to train with merged data")


if __name__ == "__main__":
    merge_datasets()
