"""
Evaluation configuration for SAE experiments.

Defines metrics computation settings, spectral analysis parameters, etc.
"""

from dataclasses import dataclass


@dataclass
class EvalConfig:
    """
    Configuration for evaluation.

    No defaults - set explicit values in baseline configs.

    Attributes:
        dead_feature_threshold: Minimum activation value to consider a feature "active"
        max_spectral_samples: Maximum samples to use for spectral stats (ELD computation)
        feature_analysis_top_k: Number of top features to show in text analysis
    """

    dead_feature_threshold: float
    max_spectral_samples: int
    feature_analysis_top_k: int
