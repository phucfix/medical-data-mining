"""Build Medical Knowledge Base from multiple datasets.

This module creates a unified knowledge base (kb_medical.csv) by:
1. Extracting symptoms from ViMedical Disease dataset
2. Extracting disease-symptom-drug relations from ViMedNER
3. Preparing ICD-10 data for future LLM expansion

Output: data/processed/kb_medical.csv

Author: Medical Data Mining Project
Date: 2025-11-30
"""

import pandas as pd
import re
import sys
import importlib.util
from pathlib import Path
from typing import List, Tuple

# Base directories
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

# Import the loading functions by directly loading the module file
# This avoids issues with the package's __init__.py
def import_module_from_path(module_name: str, file_path: Path):
    """Import a module directly from file path without going through __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load the module directly
load_datasets_module = import_module_from_path(
    "load_medical_datasets", 
    BASE_DIR / "src" / "data_generation" / "load_medical_datasets.py"
)

# Get the functions from the module
load_vimedical = load_datasets_module.load_vimedical
load_vimedner = load_datasets_module.load_vimedner
load_icd10 = load_datasets_module.load_icd10
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'


def extract_symptoms_from_vimedical(df_vimedical: pd.DataFrame) -> pd.DataFrame:
    """Extract symptom phrases from ViMedical Disease dataset.
    
    Process:
    1. Split question by comma and period
    2. Filter out question patterns like "Tôi có thể đang bị bệnh gì?"
    3. Keep symptom phrases with length > 5 characters
    
    Args:
        df_vimedical: DataFrame from load_vimedical()
        
    Returns:
        pd.DataFrame with columns: disease, entity, entity_type, relation, source_name, source_type
    """
    # Patterns to filter out (question phrases, not symptoms)
    question_patterns = [
        r'tôi có thể đang bị bệnh gì',
        r'tôi bị bệnh gì',
        r'đây là bệnh gì',
        r'bệnh gì',
        r'xin hỏi',
        r'cho tôi hỏi',
        r'làm sao',
        r'có phải',
        r'tôi nên',
        r'tôi cần',
        r'có cách nào',
        r'nên làm gì',
        r'phải làm gì',
        r'^\s*$'  # Empty strings
    ]
    question_regex = re.compile('|'.join(question_patterns), re.IGNORECASE)
    
    records = []
    
    for _, row in df_vimedical.iterrows():
        disease = str(row['disease']).strip()
        question = str(row['question']).strip()
        
        if not disease or not question or disease == 'nan' or question == 'nan':
            continue
        
        # Split by comma and period
        # Also split by semicolon and common Vietnamese separators
        phrases = re.split(r'[.,;!?\n]+', question)
        
        for phrase in phrases:
            phrase = phrase.strip()
            
            # Skip if too short
            if len(phrase) <= 5:
                continue
            
            # Skip if matches question patterns
            if question_regex.search(phrase):
                continue
            
            # Skip if mostly numbers or punctuation
            alpha_count = sum(1 for c in phrase if c.isalpha())
            if alpha_count < len(phrase) * 0.5:
                continue
            
            records.append({
                'disease': disease,
                'entity': phrase,
                'entity_type': 'symptom',
                'relation': 'has_symptom',
                'source_name': 'ViMedical_Disease',
                'source_type': 'dataset_vi'
            })
    
    df = pd.DataFrame(records)
    
    # Remove duplicates
    if len(df) > 0:
        df = df.drop_duplicates(subset=['disease', 'entity', 'relation'])
    
    return df


def extract_relations_from_vimedner(df_vimedner: pd.DataFrame) -> pd.DataFrame:
    """Extract disease-symptom and disease-drug relations from ViMedNER.
    
    Process:
    1. Group entities by sentence
    2. For each sentence with DISEASE and SYMPTOM, create (disease, symptom) pairs
    3. For each sentence with DISEASE and TREATMENT, create (disease, drug) pairs
    4. Normalize entities (lowercase, strip whitespace)
    
    Args:
        df_vimedner: DataFrame from load_vimedner()
        
    Returns:
        pd.DataFrame with columns: disease, entity, entity_type, relation, source_name, source_type
    """
    if len(df_vimedner) == 0:
        return pd.DataFrame(columns=['disease', 'entity', 'entity_type', 'relation', 'source_name', 'source_type'])
    
    records = []
    
    # Group entities by sentence
    grouped = df_vimedner.groupby('sentence')
    
    for sentence, group in grouped:
        # Get all diseases, symptoms, and treatments in this sentence
        diseases = group[group['entity_type'] == 'disease']['entity'].tolist()
        symptoms = group[group['entity_type'] == 'symptom']['entity'].tolist()
        treatments = group[group['entity_type'] == 'treatment']['entity'].tolist()
        
        # Create disease-symptom pairs
        for disease in diseases:
            disease_normalized = normalize_entity(disease)
            if not is_valid_entity(disease_normalized):
                continue
                
            for symptom in symptoms:
                symptom_normalized = normalize_entity(symptom)
                if not is_valid_entity(symptom_normalized):
                    continue
                    
                records.append({
                    'disease': disease_normalized,
                    'entity': symptom_normalized,
                    'entity_type': 'symptom',
                    'relation': 'has_symptom',
                    'source_name': 'ViMedNER',
                    'source_type': 'dataset_vi'
                })
            
            # Create disease-treatment pairs
            for treatment in treatments:
                treatment_normalized = normalize_entity(treatment)
                if not is_valid_entity(treatment_normalized):
                    continue
                    
                records.append({
                    'disease': disease_normalized,
                    'entity': treatment_normalized,
                    'entity_type': 'drug',
                    'relation': 'treated_by',
                    'source_name': 'ViMedNER',
                    'source_type': 'dataset_vi'
                })
    
    df = pd.DataFrame(records)
    
    # Remove duplicates
    if len(df) > 0:
        df = df.drop_duplicates(subset=['disease', 'entity', 'relation'])
    
    return df


def normalize_entity(text: str) -> str:
    """Normalize entity text.
    
    - Strip whitespace
    - Lowercase (except first letter of proper nouns if needed)
    - Remove extra whitespace
    
    Args:
        text: Raw entity text
        
    Returns:
        Normalized entity text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Strip and normalize whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase but keep first letter of sentence capitalized
    if text:
        text = text[0].upper() + text[1:].lower() if len(text) > 1 else text.upper()
    
    return text


def is_valid_entity(text: str) -> bool:
    """Check if entity is valid.
    
    Invalid entities:
    - Too short (< 4 characters)
    - Only numbers and punctuation
    - Empty
    
    Args:
        text: Entity text
        
    Returns:
        True if valid, False otherwise
    """
    if not text or len(text) < 4:
        return False
    
    # Check if only numbers and punctuation
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count == 0:
        return False
    
    return True


def prepare_icd10_for_translation(df_icd10: pd.DataFrame) -> pd.DataFrame:
    """Prepare ICD-10 data for future translation and LLM expansion.
    
    Note: This function creates a skeleton with description_vi placeholder.
    Actual translation using googletrans should be done separately.
    
    Args:
        df_icd10: DataFrame from load_icd10()
        
    Returns:
        pd.DataFrame with columns: disease, entity, entity_type, relation, source_name, source_type
    """
    records = []
    
    for _, row in df_icd10.iterrows():
        description = str(row['full_description']).strip()
        
        if not description or description == 'nan':
            continue
        
        # For now, use English description as disease name
        # In production, this would be translated to Vietnamese
        disease = description
        
        records.append({
            'disease': disease,
            'entity': '',  # Empty, to be filled by LLM later
            'entity_type': '',  # Empty
            'relation': '',  # Empty
            'source_name': 'ICD10_en',
            'source_type': 'dataset_en'
        })
    
    df = pd.DataFrame(records)
    
    # Remove duplicates
    if len(df) > 0:
        df = df.drop_duplicates(subset=['disease'])
    
    return df


def translate_icd10_to_vietnamese(df_icd10: pd.DataFrame) -> pd.DataFrame:
    """Translate ICD-10 descriptions from English to Vietnamese.
    
    Note: This is a skeleton function. In production, you would:
    1. Import googletrans: from googletrans import Translator
    2. Create translator: translator = Translator()
    3. Translate each description: translator.translate(text, dest='vi').text
    
    Args:
        df_icd10: DataFrame from load_icd10()
        
    Returns:
        pd.DataFrame with additional column 'description_vi'
    """
    # SKELETON FUNCTION - Translation not implemented
    # In production, use:
    #
    # from googletrans import Translator
    # translator = Translator()
    #
    # def translate_text(text):
    #     try:
    #         result = translator.translate(text, dest='vi')
    #         return result.text
    #     except Exception as e:
    #         print(f"Translation error: {e}")
    #         return text
    #
    # df_icd10['description_vi'] = df_icd10['full_description'].apply(translate_text)
    
    # For now, just copy English description
    df_result = df_icd10.copy()
    df_result['description_vi'] = df_result['full_description']
    
    return df_result


def filter_knowledge_base(kb: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean the knowledge base.
    
    Removes:
    - Duplicate (disease, entity, relation) combinations
    - Entities too short (< 4 characters)
    - Entities with only numbers/punctuation
    
    Args:
        kb: Raw knowledge base DataFrame
        
    Returns:
        Cleaned knowledge base DataFrame
    """
    if len(kb) == 0:
        return kb
    
    # Filter out invalid entities
    mask = kb['entity'].apply(lambda x: is_valid_entity(str(x)) if pd.notna(x) and x != '' else True)
    kb = kb[mask].copy()
    
    # Remove duplicates for same disease-entity-relation combination
    # Keep rows with empty entity (ICD-10 placeholders)
    kb_with_entity = kb[kb['entity'] != ''].drop_duplicates(subset=['disease', 'entity', 'relation'])
    kb_without_entity = kb[kb['entity'] == ''].drop_duplicates(subset=['disease'])
    
    kb = pd.concat([kb_with_entity, kb_without_entity], ignore_index=True)
    
    return kb


def print_statistics(kb: pd.DataFrame) -> None:
    """Print statistics about the knowledge base.
    
    Args:
        kb: Knowledge base DataFrame
    """
    print("\n" + "=" * 60)
    print("THỐNG KÊ KNOWLEDGE BASE")
    print("=" * 60)
    
    print(f"\n📊 Tổng số dòng: {len(kb)}")
    print(f"📊 Số bệnh khác nhau: {kb['disease'].nunique()}")
    
    # Statistics by entity type
    symptoms = kb[kb['entity_type'] == 'symptom']
    drugs = kb[kb['entity_type'] == 'drug']
    empty = kb[kb['entity_type'] == '']
    
    print(f"\n📋 Phân bố entity_type:")
    print(f"   - symptom: {len(symptoms)} dòng ({symptoms['entity'].nunique()} triệu chứng khác nhau)")
    print(f"   - drug: {len(drugs)} dòng ({drugs['entity'].nunique()} thuốc khác nhau)")
    print(f"   - (trống - ICD-10): {len(empty)} dòng")
    
    # Statistics by source
    print(f"\n📋 Phân bố theo nguồn:")
    source_counts = kb['source_name'].value_counts()
    for source, count in source_counts.items():
        print(f"   - {source}: {count} dòng")
    
    # Sample data
    print("\n📋 Mẫu dữ liệu:")
    print(kb.head(10).to_string())


def build_knowledge_base() -> pd.DataFrame:
    """Main function to build the medical knowledge base.
    
    Steps:
    1. Load all 3 datasets
    2. Extract symptoms from ViMedical
    3. Extract relations from ViMedNER
    4. Prepare ICD-10 placeholders
    5. Merge all data
    6. Filter and clean
    7. Save to CSV
    
    Returns:
        Final knowledge base DataFrame
    """
    print("=" * 60)
    print("XÂY DỰNG KNOWLEDGE BASE Y TẾ")
    print("=" * 60)
    
    kb_parts = []
    
    # 1. Process ViMedical Disease
    print("\n" + "-" * 40)
    print("1. Xử lý ViMedical Disease")
    print("-" * 40)
    try:
        df_vimedical = load_vimedical()
        print(f"   ✓ Đã tải {len(df_vimedical)} dòng từ ViMedical")
        
        kb_vimedical = extract_symptoms_from_vimedical(df_vimedical)
        print(f"   ✓ Trích xuất được {len(kb_vimedical)} cặp (bệnh, triệu chứng)")
        kb_parts.append(kb_vimedical)
    except Exception as e:
        print(f"   ✗ Lỗi: {e}")
    
    # 2. Process ViMedNER
    print("\n" + "-" * 40)
    print("2. Xử lý ViMedNER")
    print("-" * 40)
    try:
        df_vimedner = load_vimedner()
        print(f"   ✓ Đã tải {len(df_vimedner)} entity từ ViMedNER")
        
        kb_vimedner = extract_relations_from_vimedner(df_vimedner)
        print(f"   ✓ Trích xuất được {len(kb_vimedner)} quan hệ (bệnh-triệu chứng/thuốc)")
        kb_parts.append(kb_vimedner)
    except Exception as e:
        print(f"   ✗ Lỗi: {e}")
    
    # 3. Process ICD-10
    print("\n" + "-" * 40)
    print("3. Xử lý ICD-10")
    print("-" * 40)
    try:
        df_icd10 = load_icd10()
        print(f"   ✓ Đã tải {len(df_icd10)} mã ICD-10")
        
        # Note: Translation is not performed, just skeleton
        # df_icd10_vi = translate_icd10_to_vietnamese(df_icd10)
        
        kb_icd10 = prepare_icd10_for_translation(df_icd10)
        print(f"   ✓ Chuẩn bị {len(kb_icd10)} placeholder cho ICD-10 (chưa dịch)")
        kb_parts.append(kb_icd10)
    except Exception as e:
        print(f"   ✗ Lỗi: {e}")
    
    # 4. Merge all parts
    print("\n" + "-" * 40)
    print("4. Gộp và lọc dữ liệu")
    print("-" * 40)
    
    if not kb_parts:
        print("   ✗ Không có dữ liệu để gộp!")
        return pd.DataFrame()
    
    kb = pd.concat(kb_parts, ignore_index=True)
    print(f"   ✓ Tổng số dòng trước khi lọc: {len(kb)}")
    
    # 5. Filter and clean
    kb = filter_knowledge_base(kb)
    print(f"   ✓ Tổng số dòng sau khi lọc: {len(kb)}")
    
    # 6. Ensure correct column order
    columns = ['disease', 'entity', 'entity_type', 'relation', 'source_type', 'source_name']
    kb = kb[columns]
    
    # 7. Print statistics
    print_statistics(kb)
    
    # 8. Save to CSV
    print("\n" + "-" * 40)
    print("5. Lưu file")
    print("-" * 40)
    
    # Create processed directory if not exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = PROCESSED_DIR / 'kb_medical.csv'
    kb.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ Đã lưu vào: {output_path}")
    
    return kb


if __name__ == '__main__':
    kb = build_knowledge_base()
    
    print("\n" + "=" * 60)
    print("HOÀN TẤT!")
    print("=" * 60)
