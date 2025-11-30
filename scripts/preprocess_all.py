#!/usr/bin/env python3
"""
Script tiền xử lý tất cả dữ liệu
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import preprocess_all_data
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting data preprocessing...")
    
    try:
        preprocess_all_data()
        logger.info("Preprocessing completed successfully!")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
