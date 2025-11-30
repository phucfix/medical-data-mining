"""Generate True/False QA dataset from Medical Knowledge Base.

This module creates a balanced True/False QA dataset:
- TRUE samples: Correct disease-symptom/drug relationships
- FALSE samples: Incorrect pairings (disease A with entity from disease B)

Input: data/processed/kb_medical.csv
Output: data/final/medical_true_false_qa.csv

VÍ DỤ CÂU TỐT (cần sinh ra):
- "Ho kéo dài trên 3 tuần là triệu chứng thường gặp của bệnh lao phổi." -> TRUE
- "Metformin thường được sử dụng trong điều trị bệnh đái tháo đường type 2." -> TRUE
- "Xuất tinh sớm là triệu chứng đặc trưng của tăng huyết áp nguyên phát." -> FALSE

VÍ DỤ CÂU XẤU (cần tránh):
- "Tiêm phòng vaccine thường được sử dụng trong điều trị bệnh Bệnh ung thư dạ dày." (lặp chữ, diễn đạt kém)
- "Bệnh Bệnh mô liên kết có thể gây ra triệu chứng Tôi hiện đang có triệu chứng khớp liên đốt ngón tay bị gập." (nguyên câu hỏi bệnh nhân)

Author: Medical Data Mining Project
Date: 2025-11-30
"""

import pandas as pd
import random
import re
from pathlib import Path
from typing import List, Dict, Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
FINAL_DIR = DATA_DIR / 'final'

# Random seed for reproducibility
RANDOM_SEED = 42


def clean_disease_name(disease: str) -> str:
    """Chuẩn hóa tên bệnh, loại bỏ từ 'Bệnh' ở đầu nếu có.
    
    Args:
        disease: Tên bệnh gốc
        
    Returns:
        Tên bệnh đã chuẩn hóa (không có 'Bệnh' ở đầu, viết thường)
        
    Ví dụ:
        "Bệnh mô liên kết" -> "mô liên kết"
        "Ung Thư Dạ Dày" -> "ung thư dạ dày"
        "bệnh Tiểu Đường" -> "tiểu đường"
    """
    if not disease:
        return ""
    
    disease = disease.strip()
    
    # Loại bỏ "Bệnh " hoặc "bệnh " ở đầu
    if disease.lower().startswith("bệnh "):
        disease = disease[5:]  # Bỏ 5 ký tự "bệnh " hoặc "Bệnh "
    
    # Strip lại sau khi cắt
    disease = disease.strip()
    
    # Viết thường toàn bộ để đồng nhất
    disease = disease.lower()
    
    return disease


def clean_symptom(text: str) -> str:
    """Làm sạch chuỗi triệu chứng từ câu hỏi bệnh nhân.
    
    Input: "Tôi hiện đang có các triệu chứng như vàng da, đau bụng và có khối u ở cổ. Tôi có thể đang bị bệnh gì?"
    Output: "vàng da, đau bụng và có khối u ở cổ"
    
    Quy tắc:
    - Cắt ở đoạn "Tôi có thể đang bị bệnh gì" (nếu có).
    - Xóa các cụm mở đầu "Tôi", "Tôi hiện đang", "Tôi đang", "Tôi hay bị", "Tôi bị".
    - Loại bỏ dấu chấm câu dư thừa ở đầu/cuối.
    
    Args:
        text: Chuỗi triệu chứng gốc
        
    Returns:
        Chuỗi đã làm sạch, hoặc "" nếu quá ngắn
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    # Cắt phần câu hỏi cuối (nếu có)
    question_patterns = [
        r'[.?!]*\s*tôi có thể đang bị bệnh gì[.?!]*',
        r'[.?!]*\s*tôi bị bệnh gì[.?!]*',
        r'[.?!]*\s*đây là bệnh gì[.?!]*',
        r'[.?!]*\s*có phải tôi bị[^.?!]*[.?!]*',
        r'[.?!]*\s*xin hỏi[^.?!]*[.?!]*$',
        r'[.?!]*\s*cho tôi hỏi[^.?!]*[.?!]*$',
    ]
    for pattern in question_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Xóa các cụm mở đầu
    start_patterns = [
        r'^tôi hiện đang có các triệu chứng như\s*',
        r'^tôi hiện đang có triệu chứng\s*',
        r'^tôi hiện đang có các triệu chứng\s*',
        r'^tôi hiện đang bị\s*',
        r'^tôi hiện đang\s*',
        r'^tôi đang cảm thấy\s*',
        r'^tôi đang bị\s*',
        r'^tôi đang có\s*',
        r'^tôi đang\s*',
        r'^tôi hay bị\s*',
        r'^tôi hay\s*',
        r'^tôi bị\s*',
        r'^tôi có\s*',
        r'^tôi\s+',
        r'^hiện đang\s*',
        r'^đang bị\s*',
        r'^đang\s*',
    ]
    
    for pattern in start_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Xóa "hiện đang" ở giữa câu (vd: "xuất tinh sớm và hiện đang né tránh")
    text = re.sub(r'\s+hiện đang\s+', ' ', text, flags=re.IGNORECASE)
    
    # Loại bỏ dấu chấm câu dư thừa ở đầu/cuối
    text = re.sub(r'^[.,;:!?\s]+', '', text)
    text = re.sub(r'[.,;:!?\s]+$', '', text)
    
    # Strip và chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Viết thường chữ đầu
    if text:
        text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
    
    # Kiểm tra độ dài tối thiểu và có chữ cái tiếng Việt
    if len(text) < 10:
        return ""
    
    # Kiểm tra có ít nhất một chữ cái
    if not re.search(r'[a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', text, re.IGNORECASE):
        return ""
    
    return text


def is_valid_entity(text: str) -> bool:
    """Kiểm tra entity có hợp lệ không.
    
    Args:
        text: Entity text
        
    Returns:
        True nếu hợp lệ
    """
    if not text or len(text) < 10:
        return False
    
    # Phải có ít nhất một chữ cái tiếng Việt
    if not re.search(r'[a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', text, re.IGNORECASE):
        return False
    
    return True


def filter_bad_sentences(rows: List[Dict]) -> List[Dict]:
    """Lọc bỏ các câu không hợp lệ.
    
    Loại bỏ câu có:
    - Từ lặp 2 lần liên tiếp: "Bệnh Bệnh", "bệnh bệnh"
    - Chứa "Tôi có thể đang bị bệnh gì", "?"
    - Độ dài text < 30 ký tự
    - label="FALSE" + chứa "thường gặp" (tránh nhiễu)
    
    Args:
        rows: List các dict {"text": ..., "label": ...}
        
    Returns:
        List đã lọc
    """
    filtered = []
    
    # Patterns để phát hiện từ lặp
    duplicate_patterns = [
        r'bệnh\s+bệnh',
        r'triệu chứng\s+triệu chứng',
        r'điều trị\s+điều trị',
    ]
    duplicate_regex = re.compile('|'.join(duplicate_patterns), re.IGNORECASE)
    
    # Patterns câu hỏi bệnh nhân (không phải mệnh đề kiến thức)
    question_patterns = [
        r'tôi có thể đang bị bệnh gì',
        r'tôi bị bệnh gì',
        r'\?',  # Dấu hỏi
    ]
    question_regex = re.compile('|'.join(question_patterns), re.IGNORECASE)
    
    for row in rows:
        text = row.get('text', '')
        label = row.get('label', '')
        
        # Kiểm tra độ dài
        if len(text) < 30:
            continue
        
        # Kiểm tra từ lặp
        if duplicate_regex.search(text):
            continue
        
        # Kiểm tra câu hỏi bệnh nhân
        if question_regex.search(text):
            continue
        
        # Với FALSE, loại bỏ câu có "thường gặp" để tránh nhiễu
        if label == 'FALSE' and 'thường gặp' in text.lower():
            continue
        
        filtered.append(row)
    
    return filtered


def load_knowledge_base(path: Path = None) -> pd.DataFrame:
    """Load the medical knowledge base.
    
    Args:
        path: Path to kb_medical.csv. If None, uses default path.
        
    Returns:
        pd.DataFrame with knowledge base data
    """
    if path is None:
        path = PROCESSED_DIR / 'kb_medical.csv'
    
    df = pd.read_csv(path, encoding='utf-8-sig')
    
    # Filter out rows with empty entity (ICD-10 placeholders)
    df = df[df['entity'].notna() & (df['entity'] != '')]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def generate_true_samples(kb: pd.DataFrame) -> List[Dict]:
    """Generate TRUE samples from knowledge base.
    
    Templates:
    - Triệu chứng:
        f"{entity} là triệu chứng thường gặp của bệnh {disease_clean}."
        f"Bệnh {disease_clean} có thể gây ra triệu chứng {entity}."
    - Thuốc:
        f"{entity} thường được sử dụng trong điều trị bệnh {disease_clean}."
    
    Args:
        kb: Knowledge base DataFrame
        
    Returns:
        List of dict with keys: text, label
    """
    records = []
    
    for _, row in kb.iterrows():
        disease_raw = str(row['disease']).strip()
        entity_raw = str(row['entity']).strip()
        relation = str(row['relation']).strip()
        
        if not disease_raw or not entity_raw or disease_raw == 'nan' or entity_raw == 'nan':
            continue
        
        # Chuẩn hóa disease (loại bỏ "Bệnh" ở đầu)
        disease_clean = clean_disease_name(disease_raw)
        if not disease_clean:
            continue
        
        # Chuẩn hóa entity (làm sạch câu hỏi bệnh nhân)
        entity = clean_symptom(entity_raw)
        if not is_valid_entity(entity):
            continue
        
        # Sinh câu theo relation
        if relation == 'has_symptom':
            # Template 1: "{Entity} là triệu chứng thường gặp của bệnh {disease_clean}."
            # Viết hoa chữ đầu của entity vì đứng đầu câu
            entity_cap = entity[0].upper() + entity[1:] if len(entity) > 1 else entity.upper()
            text1 = f"{entity_cap} là triệu chứng thường gặp của bệnh {disease_clean}."
            records.append({'text': text1, 'label': 'TRUE'})
            
            # Template 2: "Bệnh {disease_clean} có thể gây ra triệu chứng {entity}."
            text2 = f"Bệnh {disease_clean} có thể gây ra triệu chứng {entity}."
            records.append({'text': text2, 'label': 'TRUE'})
            
        elif relation == 'treated_by':
            # Template: "{Entity} thường được sử dụng trong điều trị bệnh {disease_clean}."
            entity_cap = entity[0].upper() + entity[1:] if len(entity) > 1 else entity.upper()
            text = f"{entity_cap} thường được sử dụng trong điều trị bệnh {disease_clean}."
            records.append({'text': text, 'label': 'TRUE'})
    
    return records


def generate_false_samples(kb: pd.DataFrame, n_samples: int) -> List[Dict]:
    """Generate FALSE samples by pairing disease A with entity from disease B.
    
    Sử dụng template khác với TRUE để tránh nhiễu:
    - Triệu chứng: "Bệnh {disease} có thể gây ra triệu chứng {entity}."
    - Thuốc: "{entity} được sử dụng trong điều trị bệnh {disease}."
    
    Args:
        kb: Knowledge base DataFrame
        n_samples: Number of FALSE samples to generate
        
    Returns:
        List of dict with keys: text, label
    """
    random.seed(RANDOM_SEED)
    
    # Separate symptoms and drugs
    symptoms_kb = kb[kb['relation'] == 'has_symptom'].copy()
    drugs_kb = kb[kb['relation'] == 'treated_by'].copy()
    
    # Clean all diseases and entities
    symptom_diseases = []
    symptom_entities = []
    symptom_map = {}  # disease -> set of entities
    
    for _, row in symptoms_kb.iterrows():
        disease_clean = clean_disease_name(str(row['disease']).strip())
        entity = clean_symptom(str(row['entity']).strip())
        
        if disease_clean and is_valid_entity(entity):
            if disease_clean not in symptom_map:
                symptom_map[disease_clean] = set()
                symptom_diseases.append(disease_clean)
            symptom_map[disease_clean].add(entity)
            if entity not in symptom_entities:
                symptom_entities.append(entity)
    
    drug_diseases = []
    drug_entities = []
    drug_map = {}  # disease -> set of entities
    
    for _, row in drugs_kb.iterrows():
        disease_clean = clean_disease_name(str(row['disease']).strip())
        entity = clean_symptom(str(row['entity']).strip())
        
        if disease_clean and is_valid_entity(entity):
            if disease_clean not in drug_map:
                drug_map[disease_clean] = set()
                drug_diseases.append(disease_clean)
            drug_map[disease_clean].add(entity)
            if entity not in drug_entities:
                drug_entities.append(entity)
    
    records = []
    used_pairs = set()
    
    # Calculate proportions
    n_symptom_false = int(n_samples * 0.9)  # 90% triệu chứng
    n_drug_false = n_samples - n_symptom_false
    
    # Generate FALSE symptom samples
    attempts = 0
    max_attempts = n_symptom_false * 20
    
    while len([r for r in records if 'triệu chứng' in r['text']]) < n_symptom_false and attempts < max_attempts:
        attempts += 1
        
        if not symptom_diseases or not symptom_entities:
            break
        
        disease = random.choice(symptom_diseases)
        entity = random.choice(symptom_entities)
        
        # Skip if this is a TRUE relationship
        if disease in symptom_map and entity in symptom_map[disease]:
            continue
        
        pair_key = (disease, entity, 'symptom')
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        
        # Template cho FALSE: không dùng "thường gặp"
        # disease đã là lowercase từ clean_disease_name
        text = f"Bệnh {disease} có thể gây ra triệu chứng {entity}."
        records.append({'text': text, 'label': 'FALSE'})
    
    # Generate FALSE drug samples
    attempts = 0
    max_attempts = n_drug_false * 20
    
    while len([r for r in records if 'điều trị' in r['text']]) < n_drug_false and attempts < max_attempts:
        attempts += 1
        
        if not drug_diseases or not drug_entities:
            break
        
        disease = random.choice(drug_diseases)
        entity = random.choice(drug_entities)
        
        # Skip if this is a TRUE relationship
        if disease in drug_map and entity in drug_map[disease]:
            continue
        
        pair_key = (disease, entity, 'drug')
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        
        # Template cho FALSE: không dùng "thường"
        # Viết hoa chữ đầu của entity vì đứng đầu câu
        entity_cap = entity[0].upper() + entity[1:] if len(entity) > 1 else entity.upper()
        text = f"{entity_cap} được sử dụng trong điều trị bệnh {disease}."
        records.append({'text': text, 'label': 'FALSE'})
    
    return records


def create_true_false_dataset(kb: pd.DataFrame) -> pd.DataFrame:
    """Create balanced True/False QA dataset.
    
    Args:
        kb: Knowledge base DataFrame
        
    Returns:
        pd.DataFrame with columns: STT, Mệnh đề Câu hỏi, Đáp án
    """
    print("\n" + "-" * 40)
    print("1. Sinh câu TRUE")
    print("-" * 40)
    
    # Generate TRUE samples
    true_records = generate_true_samples(kb)
    print(f"   ✓ Đã sinh {len(true_records)} câu TRUE (trước lọc)")
    
    print("\n" + "-" * 40)
    print("2. Sinh câu FALSE")
    print("-" * 40)
    
    # Generate FALSE samples (approximately equal to TRUE)
    n_false_target = len(true_records)
    false_records = generate_false_samples(kb, n_false_target)
    print(f"   ✓ Đã sinh {len(false_records)} câu FALSE (trước lọc)")
    
    print("\n" + "-" * 40)
    print("3. Lọc câu không hợp lệ")
    print("-" * 40)
    
    # Combine all records
    all_records = true_records + false_records
    print(f"   Tổng số trước khi lọc: {len(all_records)}")
    
    # Filter bad sentences
    filtered_records = filter_bad_sentences(all_records)
    print(f"   Tổng số sau khi lọc: {len(filtered_records)}")
    print(f"   Đã loại bỏ: {len(all_records) - len(filtered_records)} câu")
    
    print("\n" + "-" * 40)
    print("4. Gộp và shuffle dữ liệu")
    print("-" * 40)
    
    # Create DataFrame
    df_combined = pd.DataFrame(filtered_records)
    
    # Remove duplicates
    df_combined = df_combined.drop_duplicates(subset=['text'])
    print(f"   ✓ Tổng số sau khi loại trùng: {len(df_combined)}")
    
    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"   ✓ Đã shuffle dữ liệu")
    
    # Add STT column (1-indexed)
    df_combined['STT'] = range(1, len(df_combined) + 1)
    
    # Rename columns
    df_combined = df_combined.rename(columns={
        'text': 'Mệnh đề Câu hỏi (VIETNAMESE TEXT ONLY)',
        'label': 'Đáp án (TRUE/FALSE)'
    })
    
    # Reorder columns
    df_combined = df_combined[['STT', 'Mệnh đề Câu hỏi (VIETNAMESE TEXT ONLY)', 'Đáp án (TRUE/FALSE)']]
    
    return df_combined


def print_statistics(df: pd.DataFrame) -> None:
    """Print statistics about the dataset.
    
    Args:
        df: Final dataset DataFrame
    """
    print("\n" + "=" * 60)
    print("THỐNG KÊ DATASET TRUE/FALSE QA")
    print("=" * 60)
    
    total = len(df)
    n_true = len(df[df['Đáp án (TRUE/FALSE)'] == 'TRUE'])
    n_false = len(df[df['Đáp án (TRUE/FALSE)'] == 'FALSE'])
    
    print(f"\n📊 Tổng số dòng: {total}")
    print(f"📊 Số câu TRUE: {n_true} ({n_true/total*100:.1f}%)")
    print(f"📊 Số câu FALSE: {n_false} ({n_false/total*100:.1f}%)")
    
    print("\n📋 Mẫu dữ liệu (10 dòng đầu):")
    print(df.head(10).to_string())
    
    print("\n📋 Mẫu dữ liệu (10 dòng cuối):")
    print(df.tail(10).to_string())


def main():
    """Main function to generate True/False QA dataset."""
    print("=" * 60)
    print("TẠO DATASET TRUE/FALSE QA TỪ KNOWLEDGE BASE")
    print("=" * 60)
    
    # Load knowledge base
    print("\n" + "-" * 40)
    print("0. Đọc Knowledge Base")
    print("-" * 40)
    
    kb = load_knowledge_base()
    print(f"   ✓ Đã đọc {len(kb)} dòng từ kb_medical.csv")
    print(f"   ✓ Số dòng có relation 'has_symptom': {len(kb[kb['relation'] == 'has_symptom'])}")
    print(f"   ✓ Số dòng có relation 'treated_by': {len(kb[kb['relation'] == 'treated_by'])}")
    
    # Create dataset
    df = create_true_false_dataset(kb)
    
    # Print statistics
    print_statistics(df)
    
    # Save to CSV
    print("\n" + "-" * 40)
    print("5. Lưu file")
    print("-" * 40)
    
    # Create final directory if not exists
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = FINAL_DIR / 'medical_true_false_qa.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ Đã lưu vào: {output_path}")
    
    return df


if __name__ == '__main__':
    df = main()
    
    print("\n" + "=" * 60)
    print("HOÀN TẤT!")
    print("=" * 60)
