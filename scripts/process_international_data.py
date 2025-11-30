#!/usr/bin/env python3
"""
Script xử lý dữ liệu quốc tế (ICD-10, MeSH, HPO)
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.translation import process_all_international_data
from loguru import logger

if __name__ == "__main__":
    logger.info("Processing international medical data...")
    
    try:
        data = process_all_international_data()
        
        print("\n" + "="*50)
        print("INTERNATIONAL DATA PROCESSING COMPLETED")
        print("="*50)
        print(f"Diseases: {len(data.get('diseases', []))}")
        print(f"Symptoms: {len(data.get('symptoms', []))}")
        print(f"Drugs: {len(data.get('drugs', []))}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
