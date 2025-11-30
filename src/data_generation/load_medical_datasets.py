"""Load and prepare 3 types of medical datasets for Vietnamese.

This module provides functions to load:
1. ViMedical Disease dataset (Vietnamese diseases and symptoms)
2. ViMedNER dataset (Vietnamese medical NER)
3. ICD-10 codes dataset (English, to be translated to Vietnamese)

Author: Medical Data Mining Project
Date: 2025-11-30
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'


def load_vimedical(path: Optional[str] = None) -> pd.DataFrame:
    """Load ViMedical Disease dataset (Vietnamese diseases and symptoms).
    
    Args:
        path: Path to the CSV file. If None, uses default path.
              Expected columns: Disease, Question
    
    Returns:
        pd.DataFrame with columns:
            - disease: disease name (Vietnamese)
            - question: symptom description as patient narrative
            - source_name: "ViMedical_Disease"
            - source_type: "dataset_vi"
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if path is None:
        path = RAW_DIR / 'ViMedical_Disease.csv'
    else:
        path = Path(path)
    
    try:
        # Read CSV with UTF-8-SIG encoding for Vietnamese
        df = pd.read_csv(path, encoding='utf-8-sig')
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File không tồn tại: {path}\n"
            "Vui lòng tải file ViMedical_Disease.csv từ Kaggle/HuggingFace "
            "và lưu vào thư mục data/raw/"
        )
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file {path}: {e}")
    
    # Normalize column names (handle various cases)
    df.columns = df.columns.str.strip().str.lower()
    
    # Check required columns
    required_cols = ['disease', 'question']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Thiếu các cột bắt buộc: {missing_cols}\n"
            f"Các cột hiện có: {list(df.columns)}"
        )
    
    # Keep only required columns and rename
    df = df[['disease', 'question']].copy()
    
    # Add metadata columns
    df['source_name'] = 'ViMedical_Disease'
    df['source_type'] = 'dataset_vi'
    
    # Clean data
    df['disease'] = df['disease'].astype(str).str.strip()
    df['question'] = df['question'].astype(str).str.strip()
    
    return df


def load_vimedner(path: Optional[str] = None) -> pd.DataFrame:
    """Load ViMedNER dataset (Vietnamese medical NER).
    
    Supports multiple formats:
    1. ViMedNER format from tdtrinh11/ViMedNer: {"id": ..., "text": "...", "label": [[start, end, "TYPE"], ...]}
    2. Standard NER format: {"text": "...", "entities": [{"text": "...", "type": "..."}, ...]}
    
    Args:
        path: Path to the JSONL/TXT file. If None, uses default path.
    
    Returns:
        pd.DataFrame in "long" format with columns:
            - sentence: original sentence
            - entity: entity string (disease name, symptom, drug)
            - entity_type: one of ["disease", "symptom", "treatment", "cause", "diagnostic"]
            - source_name: "ViMedNER"
            - source_type: "dataset_vi"
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    if path is None:
        # Try different file names
        possible_paths = [
            RAW_DIR / 'ViMedNER.txt',
            RAW_DIR / 'ViMedNER_raw.txt',
            RAW_DIR / 'ViMedNER.jsonl',
        ]
        path = None
        for p in possible_paths:
            if p.exists():
                path = p
                break
        if path is None:
            raise FileNotFoundError(
                f"Không tìm thấy file ViMedNER. Đã tìm trong:\n"
                + "\n".join(f"  - {p}" for p in possible_paths) +
                "\nVui lòng tải file từ https://github.com/tdtrinh11/ViMedNer"
            )
    else:
        path = Path(path)
    
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File không tồn tại: {path}\n"
            "Vui lòng tải file ViMedNER.txt từ https://github.com/tdtrinh11/ViMedNer "
            "và lưu vào thư mục data/raw/"
        )
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file {path}: {e}")
    
    # Parse JSONL and extract entities
    records = []
    entity_type_mapping = {
        # Main types
        'DISEASE': 'disease',
        'SYMPTOM': 'symptom', 
        'TREATMENT': 'treatment',
        'CAUSE': 'cause',
        'DIAGNOSTIC': 'diagnostic',
        # Lowercase variants
        'disease': 'disease',
        'symptom': 'symptom',
        'treatment': 'treatment',
        'cause': 'cause',
        'diagnostic': 'diagnostic',
        # Vietnamese labels if present
        'BENH': 'disease',
        'TRIEU_CHUNG': 'symptom',
        'DIEU_TRI': 'treatment',
        'NGUYEN_NHAN': 'cause',
        'CHAN_DOAN': 'diagnostic',
    }
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Cảnh báo: Dòng {line_num} không phải JSON hợp lệ: {e}")
            continue
        
        sentence = data.get('text', data.get('sentence', ''))
        
        if not sentence:
            continue
        
        # Try to get labels in different formats
        # Format 1: ViMedNER format - "label": [[start, end, "TYPE"], ...]
        labels = data.get('label', [])
        
        # Format 2: Standard format - "entities": [{"text": "...", "type": "..."}, ...]
        entities = data.get('entities', data.get('ner', []))
        
        # Process ViMedNER format (label field with [start, end, type])
        if labels and isinstance(labels, list):
            for label in labels:
                if isinstance(label, (list, tuple)) and len(label) >= 3:
                    start, end, entity_label = label[0], label[1], label[2]
                    
                    # Extract entity text from sentence using positions
                    try:
                        entity_text = sentence[start:end]
                    except (IndexError, TypeError):
                        continue
                    
                    # Normalize entity type
                    entity_type = entity_type_mapping.get(
                        str(entity_label).upper(), 
                        str(entity_label).lower()
                    )
                    
                    if entity_text and entity_type:
                        records.append({
                            'sentence': sentence.strip(),
                            'entity': str(entity_text).strip(),
                            'entity_type': entity_type,
                            'source_name': 'ViMedNER',
                            'source_type': 'dataset_vi'
                        })
        
        # Process standard entities format
        if entities and isinstance(entities, list):
            for ent in entities:
                entity_text = None
                entity_label = None
                
                if isinstance(ent, dict):
                    # Format: {"text": "...", "type": "DISEASE"}
                    entity_text = ent.get('text', ent.get('entity', ent.get('word', '')))
                    entity_label = ent.get('type', ent.get('label', ent.get('entity_type', '')))
                elif isinstance(ent, (list, tuple)) and len(ent) >= 2:
                    if len(ent) >= 4:
                        entity_text = ent[3]
                        entity_label = ent[2]
                    else:
                        entity_text = ent[0]
                        entity_label = ent[1]
                
                if entity_text and entity_label:
                    entity_type = entity_type_mapping.get(
                        str(entity_label).upper(), 
                        str(entity_label).lower()
                    )
                    
                    if entity_type:
                        records.append({
                            'sentence': sentence.strip(),
                            'entity': str(entity_text).strip(),
                            'entity_type': entity_type,
                            'source_name': 'ViMedNER',
                            'source_type': 'dataset_vi'
                        })
    
    if not records:
        print(f"Cảnh báo: Không tìm thấy entity nào trong file {path}")
        return pd.DataFrame(columns=['sentence', 'entity', 'entity_type', 'source_name', 'source_type'])
    
    df = pd.DataFrame(records)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df


def load_icd10(path: Optional[str] = None) -> pd.DataFrame:
    """Load ICD-10 codes dataset (English, to be translated to Vietnamese).
    
    Args:
        path: Path to the CSV file. If None, uses default path.
              Expected format from k4m1113/ICD-10-CSV: no header, columns are:
              category, subcategory, code, short_description, long_description, category_title
    
    Returns:
        pd.DataFrame with columns:
            - diagnosis_code: ICD-10 code
            - full_description: disease/condition description in English
            - source_name: "ICD10_en"
            - source_type: "dataset_en"
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    if path is None:
        path = RAW_DIR / 'icd10_codes.csv'
    else:
        path = Path(path)
    
    try:
        # Try different encodings
        try:
            df = pd.read_csv(path, encoding='utf-8', header=None)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1', header=None)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File không tồn tại: {path}\n"
            "Vui lòng tải file codes.csv từ GitHub ICD-10-CSV "
            "và lưu vào data/raw/icd10_codes.csv"
        )
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file {path}: {e}")
    
    # Check if file has header or not
    first_val = str(df.iloc[0, 0]).strip().upper() if len(df) > 0 else ''
    
    # If first value looks like a header (contains letters like 'code', 'category', etc.)
    if first_val in ['CATEGORY', 'CODE', 'DIAGNOSIS_CODE', 'ICD10']:
        df = pd.read_csv(path, encoding='utf-8')
        df.columns = df.columns.str.strip().str.lower()
        
        # Find appropriate columns
        code_col = None
        desc_col = None
        for col_name in ['diagnosis_code', 'code', 'icd10_code', 'icd_code']:
            if col_name in df.columns:
                code_col = col_name
                break
        for col_name in ['full_description', 'long_description', 'description', 'name']:
            if col_name in df.columns:
                desc_col = col_name
                break
        
        if code_col and desc_col:
            df = df[[code_col, desc_col]].copy()
            df.columns = ['diagnosis_code', 'full_description']
    else:
        # No header - format from k4m1113/ICD-10-CSV:
        # category, subcategory, code, short_description, long_description, category_title
        if df.shape[1] >= 5:
            # Column 2 is the full code, column 4 is long_description
            df = df[[2, 4]].copy()
            df.columns = ['diagnosis_code', 'full_description']
        elif df.shape[1] >= 2:
            # Fallback: use first two columns
            df = df[[0, 1]].copy()
            df.columns = ['diagnosis_code', 'full_description']
        else:
            raise ValueError(f"File không đủ cột. Số cột: {df.shape[1]}")
    
    # Add metadata columns
    df['source_name'] = 'ICD10_en'
    df['source_type'] = 'dataset_en'
    
    # Clean data
    df['diagnosis_code'] = df['diagnosis_code'].astype(str).str.strip()
    df['full_description'] = df['full_description'].astype(str).str.strip()
    
    # Remove rows with missing data
    df = df[df['diagnosis_code'].notna() & (df['diagnosis_code'] != '')]
    df = df[df['full_description'].notna() & (df['full_description'] != '')]
    
    return df


def main():
    """Main function to load and display all 3 datasets."""
    print("=" * 80)
    print("LOADING MEDICAL DATASETS FOR VIETNAMESE")
    print("=" * 80)
    
    datasets = {}
    
    # 1. Load ViMedical Disease
    print("\n" + "-" * 40)
    print("1. ViMedical Disease Dataset")
    print("-" * 40)
    try:
        df_vimedical = load_vimedical()
        datasets['vimedical'] = df_vimedical
        print(f"✓ Số dòng: {len(df_vimedical)}")
        print(f"✓ Các cột: {list(df_vimedical.columns)}")
        print("\n5 dòng đầu tiên:")
        print(df_vimedical.head().to_string())
    except FileNotFoundError as e:
        print(f"✗ Lỗi: {e}")
    except ValueError as e:
        print(f"✗ Lỗi dữ liệu: {e}")
    
    # 2. Load ViMedNER
    print("\n" + "-" * 40)
    print("2. ViMedNER Dataset")
    print("-" * 40)
    try:
        df_vimedner = load_vimedner()
        datasets['vimedner'] = df_vimedner
        print(f"✓ Số dòng: {len(df_vimedner)}")
        print(f"✓ Các cột: {list(df_vimedner.columns)}")
        if len(df_vimedner) > 0:
            print("\n5 dòng đầu tiên:")
            print(df_vimedner.head().to_string())
            print(f"\nPhân bố entity_type:")
            print(df_vimedner['entity_type'].value_counts())
    except FileNotFoundError as e:
        print(f"✗ Lỗi: {e}")
    except ValueError as e:
        print(f"✗ Lỗi dữ liệu: {e}")
    
    # 3. Load ICD-10
    print("\n" + "-" * 40)
    print("3. ICD-10 Codes Dataset")
    print("-" * 40)
    try:
        df_icd10 = load_icd10()
        datasets['icd10'] = df_icd10
        print(f"✓ Số dòng: {len(df_icd10)}")
        print(f"✓ Các cột: {list(df_icd10.columns)}")
        print("\n5 dòng đầu tiên:")
        print(df_icd10.head().to_string())
    except FileNotFoundError as e:
        print(f"✗ Lỗi: {e}")
    except ValueError as e:
        print(f"✗ Lỗi dữ liệu: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TÓM TẮT")
    print("=" * 80)
    print(f"Số dataset đã tải thành công: {len(datasets)}/3")
    for name, df in datasets.items():
        print(f"  - {name}: {len(df)} dòng")
    
    return datasets


if __name__ == '__main__':
    main()
