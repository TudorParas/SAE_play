"""
Evaluation utilities for SAEs.
"""

from .metrics import (
    compute_reconstruction_loss,
    compute_sparsity,
    compute_dead_features,
    get_spectral_stats,
)
from .analyzer import analyze_features, print_feature_analysis
from .report import ExperimentReport, create_experiment_id
from .evaluator import Evaluator, EvalConfig, EvalResults, AnalysisResults

__all__ = [
    # Unified evaluator (preferred)
    "Evaluator",
    "EvalConfig",
    "EvalResults",
    "AnalysisResults",
    # Individual metrics (for custom usage)
    "compute_reconstruction_loss",
    "compute_sparsity",
    "compute_dead_features",
    "get_spectral_stats",
    # Legacy analyzer functions
    "analyze_features",
    "print_feature_analysis",
    # Report
    "ExperimentReport",
    "create_experiment_id",
]
