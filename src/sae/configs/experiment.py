"""
Complete experiment configuration.

Composes all config components into a single experiment specification.
"""

from dataclasses import dataclass
from typing import Union

from .model import ModelConfig
from .data import DataConfig
from .sae import SimpleSAEConfig, DeepSAEConfig
from .training import TrainingConfig
from .evaluation import EvalConfig


@dataclass
class SAEExperimentConfig:
    """
    Complete configuration for an SAE experiment.

    Composes all configuration components: model, data, SAE architecture, training, and evaluation.
    Use dataclasses.replace() to modify nested configs.

    Attributes:
        project_name: Project identifier (e.g., "exp01a")
        experiment_name: Identifier for this experiment (e.g., "simple_sae_fast")
        model: Model and activation extraction config
        data: Data loading and processing config
        sae: SAE architecture config (SimpleSAE or DeepSAE)
        training: Training hyperparameters
        evaluation: Evaluation settings (metrics, spectral stats, etc.)
    """

    project_name: str
    experiment_name: str
    model: ModelConfig
    data: DataConfig
    sae: Union[SimpleSAEConfig, DeepSAEConfig]
    training: TrainingConfig
    evaluation: EvalConfig
