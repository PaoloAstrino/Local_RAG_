#!/usr/bin/env python3
"""
Configuration management for RAG evaluation pipeline
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class EvaluationConfig:
    """Configuration for RAG evaluation pipeline"""
    # API Configuration
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = "gemini-1.5-flash"

    # Cost Management
    max_cost_usd: float = 10.0  # Reduced for 100-question test
    cost_per_gemini_call: float = 0.01

    # Evaluation Parameters
    queries_per_dataset: int = 50  # 50 per dataset x 2 datasets = ~100 total questions
    random_seed: int = 42
    max_consecutive_errors: int = 10
    state_save_interval: int = 5  # Save more frequently for smaller run

    # API Parameters
    gemini_temperature: float = 0.6
    gemini_top_p: float = 0.95
    gemini_top_k: int = 40
    gemini_max_tokens: int = 1024
    api_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    # Evaluation Thresholds
    hallucination_threshold: float = 3.0  # Scores below this indicate potential hallucination
    low_confidence_threshold: float = 3.0  # Scores below this indicate low confidence

    # Output Configuration
    enable_compression: bool = True
    detailed_logging: bool = True
    save_raw_responses: bool = True

    # Dataset Configuration
    datasets_to_evaluate: List[str] = None

    # Hybrid retrieval weights (always use 30/70)
    hybrid_sparse_weight: float = 0.3  # 30% sparse
    hybrid_dense_weight: float = 0.7   # 70% dense

    def __post_init__(self):
        if self.datasets_to_evaluate is None:
            # Temporarily excluding 'natural_questions' due to slow retrieval
            self.datasets_to_evaluate = ["squad", "ms_marco"]
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY not set in environment variables")
        
        if self.max_cost_usd <= 0:
            errors.append("max_cost_usd must be positive")
        
        if self.queries_per_dataset <= 0:
            errors.append("queries_per_dataset must be positive")
        
        if self.max_consecutive_errors <= 0:
            errors.append("max_consecutive_errors must be positive")
        
        if not (0.0 <= self.gemini_temperature <= 2.0):
            errors.append("gemini_temperature must be between 0.0 and 2.0")
        
        if not (0.0 <= self.gemini_top_p <= 1.0):
            errors.append("gemini_top_p must be between 0.0 and 1.0")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'gemini_model': self.gemini_model,
            'max_cost_usd': self.max_cost_usd,
            'queries_per_dataset': self.queries_per_dataset,
            'random_seed': self.random_seed,
            'gemini_temperature': self.gemini_temperature,
            'gemini_top_p': self.gemini_top_p,
            'gemini_top_k': self.gemini_top_k,
            'gemini_max_tokens': self.gemini_max_tokens,
            'datasets_to_evaluate': self.datasets_to_evaluate,
            'hallucination_threshold': self.hallucination_threshold,
            'low_confidence_threshold': self.low_confidence_threshold,
            'hybrid_sparse_weight': self.hybrid_sparse_weight,
            'hybrid_dense_weight': self.hybrid_dense_weight
        }

# Global configuration instance
config = EvaluationConfig()
