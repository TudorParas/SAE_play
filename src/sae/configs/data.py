"""
Data configuration for SAE experiments.

Defines how to load and split activation data.
"""

from dataclasses import dataclass


@dataclass
class SourceConfig:
    """
    Configuration for a single data source.

    Attributes:
        name: Dataset source name (e.g., "wikitext", "c4")
        train_frac: Fraction of this source for training (0.0 to 1.0)
        test_frac: Fraction of this source for testing (0.0 to 1.0)
    """

    name: str
    train_frac: float
    test_frac: float


@dataclass
class DataConfig:
    """
    Configuration for data loading and processing.

    Attributes:
        sources: List of data sources with per-source train/test fractions
        num_samples: Number of text samples to load per source (None = all)
        extraction_batch_size: Batch size for activation extraction
        training_batch_size: Batch size for SAE training
        seed: Random seed for reproducible splits
        max_length: Maximum sequence length for tokenization (tokens)
        num_workers: Number of DataLoader worker processes for parallel loading
    """

    sources: list[SourceConfig]
    num_samples: int | None
    extraction_batch_size: int
    training_batch_size: int
    seed: int
    max_length: int
    num_workers: int = 0
