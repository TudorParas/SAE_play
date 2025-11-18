"""
Load text samples from The Pile dataset.

This is a simple data loading utility that reads pre-downloaded Pile samples.
"""

import json
from pathlib import Path
from typing import List, Optional
import random


def load_pile_samples(
    data_file: str = "src/sae/data/pile_samples.json",
    num_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[str]:
    """
    Load text samples from a JSON file containing Pile dataset samples.

    Args:
        data_file: Path to the JSON file with samples (relative to project root)
        num_samples: How many samples to load (None = load all available)
        shuffle: Whether to shuffle the samples before selection
        seed: Random seed for reproducible shuffling

    Returns:
        List of text strings

    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    data_path = Path(data_file)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            f"Expected location: {data_path.absolute()}\n"
            f"Please ensure pile_samples.json exists at this location."
        )

    with open(data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Extract text field from each sample
    # Each sample is a dict with 'text' and 'meta' fields
    texts = [s['text'] for s in samples]

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(texts)

    # Limit to requested number
    if num_samples is not None:
        texts = texts[:num_samples]

    return texts
