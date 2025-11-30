#!/usr/bin/env python3
"""
Script đánh giá chất lượng dữ liệu
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import evaluate_data_quality
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting data quality evaluation...")
    
    try:
        evaluate_data_quality()
        logger.info("Evaluation completed! Check reports folder for detailed report.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
