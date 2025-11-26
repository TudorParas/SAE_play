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
        num_workers: Number of DataLoader worker processes for parallel loading
    """

    num_samples: int
    train_frac: float
    extraction_batch_size: int
    training_batch_size: int
    seed: int
    num_workers: int = 0
