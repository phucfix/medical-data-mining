"""
Tạo dataset mới với data augmentation
KHÔNG merge Test_sample.v1.0.csv vào train
Test_sample.v1.0.csv chỉ dùng để evaluate cuối cùng
"""

import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Input: Dữ liệu gốc (KHÔNG bao gồm Test_sample)
ORIGINAL_TRAIN_FILE = DATA_DIR / "slm_train.jsonl"
ORIGINAL_VAL_FILE = DATA_DIR / "slm_val.jsonl" 
ORIGINAL_TEST_FILE = DATA_DIR / "slm_test_dev.jsonl"

# Output: Dữ liệu đã augment
AUGMENTED_TRAIN_FILE = DATA_DIR / "slm_train_augmented.jsonl"
AUGMENTED_VAL_FILE = DATA_DIR / "slm_val_augmented.jsonl"
AUGMENTED_TEST_FILE = DATA_DIR / "slm_test_augmented.jsonl"

# Config
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


def organize_augmented_data():
    """
    Tổ chức dữ liệu sau khi augment:
    - Train: Dữ liệu augmented (từ slm_train_augmented.jsonl)
    - Val: Giữ nguyên original validation set
    - Test: Giữ nguyên original test set
    """
    print("=" * 60)
    print("ORGANIZING AUGMENTED DATASET")
    print("=" * 60)
    
    # Load augmented train data
    print("\nLoading augmented training data...")
    if not (DATA_DIR / "slm_train_augmented.jsonl").exists():
        print("ERROR: slm_train_augmented.jsonl not found!")
        print("Please run: python src/augment_data.py first")
        return
    
    augmented_train = load_jsonl(DATA_DIR / "slm_train_augmented.jsonl")
    print(f"Augmented train: {len(augmented_train)} samples")
    
    # Load original val/test (NO augmentation for eval sets)
    print("\nLoading original validation and test sets...")
    original_val = load_jsonl(ORIGINAL_VAL_FILE)
    original_test = load_jsonl(ORIGINAL_TEST_FILE)
    print(f"Validation: {len(original_val)} samples")
    print(f"Test: {len(original_test)} samples")
    
    # Shuffle train
    random.seed(RANDOM_STATE)
    random.shuffle(augmented_train)
    
    # Save organized data
    print("\nSaving organized dataset...")
    save_jsonl(augmented_train, AUGMENTED_TRAIN_FILE)
    save_jsonl(original_val, AUGMENTED_VAL_FILE)
    save_jsonl(original_test, AUGMENTED_TEST_FILE)
    
    # Summary
    print("\n" + "=" * 60)
    print("DATASET ORGANIZATION SUMMARY")
    print("=" * 60)
    print(f"Training (augmented):   {len(augmented_train)} samples")
    print(f"Validation (original):  {len(original_val)} samples")
    print(f"Test (original):        {len(original_test)} samples")
    print(f"\nTotal samples: {len(augmented_train) + len(original_val) + len(original_test)}")
    
    print(f"\n✓ Dataset organization completed!")
    print(f"\nOutput files:")
    print(f"  - {AUGMENTED_TRAIN_FILE}")
    print(f"  - {AUGMENTED_VAL_FILE}")
    print(f"  - {AUGMENTED_TEST_FILE}")
    
    print(f"\n📝 Note: Test_sample.v1.0.csv is kept separate for final evaluation")


if __name__ == "__main__":
    organize_augmented_data()
