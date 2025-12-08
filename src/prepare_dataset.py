"""
Module để chuẩn bị dataset cho fine-tuning model.
Đọc dữ liệu từ medical_true_false_qa.csv và tạo format instruction.
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

# Đường dẫn
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "final" / "medical_true_false_qa.csv"

# Output files
TRAIN_FILE = DATA_DIR / "slm_train.jsonl"
VAL_FILE = DATA_DIR / "slm_val.jsonl"
TEST_DEV_FILE = DATA_DIR / "slm_test_dev.jsonl"

# Tỷ lệ chia dữ liệu
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_DEV_RATIO = 0.1

# Độ dài tối thiểu của text
MIN_TEXT_LENGTH = 20

# Random seed
RANDOM_STATE = 42


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Đọc dữ liệu từ file CSV.
    
    Args:
        file_path: Đường dẫn đến file CSV.
        
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu.
    """
    print(f"Loading data from {file_path}...")
    
    df = pd.read_csv(file_path)
    
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Original shape: {df.shape}")
    
    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa dữ liệu:
    - Đổi tên cột thành id, text, label.
    - Loại bỏ các dòng text bị trống hoặc chiều dài < 20 ký tự.
    
    Args:
        df: DataFrame gốc.
        
    Returns:
        pd.DataFrame: DataFrame đã chuẩn hóa.
    """
    print("\nNormalizing data...")
    
    # Mapping tên cột
    column_mapping = {
        "STT": "id",
        "Mệnh đề Câu hỏi (VIETNAMESE TEXT ONLY)": "text",
        "Đáp án (TRUE/FALSE)": "label"
    }
    
    # Đổi tên cột
    df = df.rename(columns=column_mapping)
    
    # Chỉ giữ các cột cần thiết
    df = df[["id", "text", "label"]]
    
    print(f"Columns after renaming: {df.columns.tolist()}")
    
    # Loại bỏ các dòng text bị null
    before_count = len(df)
    df = df.dropna(subset=["text"])
    after_null_removal = len(df)
    print(f"Removed {before_count - after_null_removal} rows with null text")
    
    # Chuyển text thành string và loại bỏ khoảng trắng thừa
    df["text"] = df["text"].astype(str).str.strip()
    
    # Loại bỏ các dòng text có chiều dài < MIN_TEXT_LENGTH
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    after_length_filter = len(df)
    print(f"Removed {after_null_removal - after_length_filter} rows with text length < {MIN_TEXT_LENGTH}")
    
    # Chuẩn hóa label thành uppercase
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    
    # Chỉ giữ các label hợp lệ (TRUE hoặc FALSE)
    valid_labels = ["TRUE", "FALSE"]
    df = df[df["label"].isin(valid_labels)]
    after_label_filter = len(df)
    print(f"Removed {after_length_filter - after_label_filter} rows with invalid labels")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Final shape after normalization: {df.shape}")
    
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """
    Chia dữ liệu thành train, val, test_dev với stratify theo label.
    
    Args:
        df: DataFrame đã chuẩn hóa.
        
    Returns:
        tuple: (train_df, val_df, test_dev_df)
    """
    print("\nSplitting data...")
    
    # Chia train và temp (val + test_dev)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_DEV_RATIO),
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )
    
    # Chia temp thành val và test_dev (50-50 vì mỗi cái là 10%)
    val_df, test_dev_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"]
    )
    
    # Reset index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_dev_df = test_dev_df.reset_index(drop=True)
    
    print(f"Train size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test_dev size: {len(test_dev_df)} ({len(test_dev_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_dev_df


def create_instruction_format(row: pd.Series) -> dict:
    """
    Tạo format huấn luyện kiểu instruction cho một mẫu.
    Format đơn giản: chỉ có text gốc, không thêm prefix.
    
    Args:
        row: Một dòng trong DataFrame.
        
    Returns:
        dict: Dict với input và output.
    """
    return {
        "input": row["text"],
        "output": row["label"]
    }


def save_to_jsonl(df: pd.DataFrame, file_path: Path):
    """
    Lưu DataFrame ra file JSONL.
    
    Args:
        df: DataFrame cần lưu.
        file_path: Đường dẫn file output.
    """
    print(f"Saving to {file_path}...")
    
    # Tạo thư mục nếu chưa tồn tại
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            sample = create_instruction_format(row)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(df)} samples to {file_path}")


def print_label_distribution(df: pd.DataFrame, name: str):
    """
    In phân phối nhãn của một DataFrame.
    
    Args:
        df: DataFrame cần phân tích.
        name: Tên của tập dữ liệu.
    """
    label_counts = Counter(df["label"])
    total = len(df)
    
    print(f"\n{name} label distribution:")
    for label in ["TRUE", "FALSE"]:
        count = label_counts.get(label, 0)
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count} ({percentage:.1f}%)")


def prepare_dataset():
    """
    Hàm chính để chuẩn bị dataset.
    """
    print("=" * 60)
    print("PREPARING DATASET FOR FINE-TUNING")
    print("=" * 60)
    
    # 1. Đọc dữ liệu
    df = load_data(INPUT_FILE)
    
    # 2. Chuẩn hóa dữ liệu
    df = normalize_data(df)
    
    # 3. Chia dữ liệu
    train_df, val_df, test_dev_df = split_data(df)
    
    # 4. In phân phối nhãn
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)
    print_label_distribution(train_df, "Train")
    print_label_distribution(val_df, "Val")
    print_label_distribution(test_dev_df, "Test_dev")
    
    # 5. Lưu ra file JSONL
    print("\n" + "=" * 60)
    print("SAVING FILES")
    print("=" * 60)
    save_to_jsonl(train_df, TRAIN_FILE)
    save_to_jsonl(val_df, VAL_FILE)
    save_to_jsonl(test_dev_df, TEST_DEV_FILE)
    
    # 6. In tổng kết
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test_dev samples: {len(test_dev_df)}")
    print(f"\nOutput files:")
    print(f"  - {TRAIN_FILE}")
    print(f"  - {VAL_FILE}")
    print(f"  - {TEST_DEV_FILE}")
    
    # In một vài ví dụ
    print("\n" + "=" * 60)
    print("SAMPLE DATA (first 3 from train)")
    print("=" * 60)
    for i, (_, row) in enumerate(train_df.head(3).iterrows()):
        sample = create_instruction_format(row)
        print(f"\nSample {i + 1}:")
        print(f"  Input: {sample['input'][:100]}...")
        print(f"  Output: {sample['output']}")
    
    print("\n✓ Dataset preparation completed!")
    
    return train_df, val_df, test_dev_df


if __name__ == "__main__":
    prepare_dataset()
