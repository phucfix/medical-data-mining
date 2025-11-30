#!/usr/bin/env python3
"""
Script sinh dữ liệu câu hỏi Q&A
Sử dụng: python generate_qa.py --llm gemini --max-items 100
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import generate_qa_data
from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description='Generate medical Q&A data')
    
    parser.add_argument(
        '--llm', 
        type=str, 
        default='gemini',
        choices=['openai', 'gemini', 'local'],
        help='LLM to use for generation'
    )
    
    parser.add_argument(
        '--questions-per-item',
        type=int,
        default=5,
        help='Number of questions to generate per medical item'
    )
    
    parser.add_argument(
        '--max-items',
        type=int,
        default=100,
        help='Maximum number of items to process'
    )
    
    parser.add_argument(
        '--additional-topics',
        action='store_true',
        help='Generate additional Q&A by medical topics'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    logger.info(f"Starting Q&A generation with {args.llm}...")
    logger.info(f"Max items: {args.max_items}, Questions per item: {args.questions_per_item}")
    
    try:
        qa_data = generate_qa_data(
            llm_type=args.llm,
            questions_per_item=args.questions_per_item,
            max_items=args.max_items,
            additional_topics=args.additional_topics
        )
        
        print("\n" + "="*50)
        print("Q&A GENERATION COMPLETED")
        print("="*50)
        print(f"Total Q&A pairs generated: {len(qa_data)}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
