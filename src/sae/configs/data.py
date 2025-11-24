"""
Data configuration for SAE experiments.

Defines how to load and split activation data.
"""

from dataclasses import dataclass


@dataclass
class DataConfig:
    """
    Configuration for data loading and processing.

    Attributes:
        num_samples: Number of text samples to load from dataset
        train_frac: Fraction of data to use for training (rest is test)
        extraction_batch_size: Batch size for activation extraction
        training_batch_size: Batch size for SAE training
        seed: Random seed for reproducible splits
    """

    num_samples: int = 10000
    train_frac: float = 0.9
    extraction_batch_size: int = 8
    training_batch_size: int = 32
    seed: int = 42
