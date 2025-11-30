"""
Evaluation module
"""
from .quality_evaluator import (
    DataQualityEvaluator,
    evaluate_data_quality
)

__all__ = [
    'DataQualityEvaluator',
    'evaluate_data_quality'
]
