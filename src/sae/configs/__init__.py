"""
SAE experiment configuration system.

Provides dataclass-based configs for defining experiments. Configs are composable
via dataclasses.replace() to create experiment variations.

Example:
    >>> from dataclasses import replace
    >>> from src.sae.configs import SIMPLE_SAE
    >>>
    >>> # Modify nested config
    >>> config = replace(SIMPLE_SAE,
    ...     training=replace(SIMPLE_SAE.training, num_epochs=50)
    ... )
"""

from .model import ModelConfig
from .data import DataConfig
from .sae import SimpleSAEConfig, DeepSAEConfig
from .training import TrainingConfig
from .experiment import SAEExperimentConfig
from .lr_schedule import LRScheduleConfig, OneCycleLRConfig
from .baselines import SIMPLE_SAE, DEEP_SAE

__all__ = [
    # Config components
    "ModelConfig",
    "DataConfig",
    "SimpleSAEConfig",
    "DeepSAEConfig",
    "TrainingConfig",
    "SAEExperimentConfig",
    # LR schedule configs
    "LRScheduleConfig",
    "OneCycleLRConfig",
    # Baseline configs
    "SIMPLE_SAE",
    "DEEP_SAE",
]
