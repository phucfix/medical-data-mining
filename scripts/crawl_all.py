#!/usr/bin/env python3
"""
Script chạy crawl tất cả các nguồn dữ liệu y tế
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crawler import crawl_all_sources
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting data crawling from all sources...")
    
    try:
        crawl_all_sources()
        logger.info("Crawling completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Crawling interrupted by user")
    except Exception as e:
        logger.error(f"Crawling failed: {e}")
        raise
