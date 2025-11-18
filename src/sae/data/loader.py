"""
Data loader that handles path resolution automatically.

This finds the data file relative to this module's location, so it works
regardless of where you run your script from (PyCharm, command line, etc.).
"""

import json
from pathlib import Path
from typing import List, Optional
import random


# Find the data directory (same directory as this file)
_DATA_DIR = Path(__file__).parent
_DEFAULT_DATA_FILE = _DATA_DIR / "pile_samples.json"


def load_pile_samples(
    data_file: Optional[str] = None,
    num_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[str]:
    """
    Load text samples from The Pile dataset.

    This automatically finds the data file relative to the module location,
    so it works regardless of your current working directory.

    Args:
        data_file: Optional path to data file. If None, uses the default
                   pile_samples.json in this module's directory.
        num_samples: How many samples to load (None = load all available)
        shuffle: Whether to shuffle the samples before selection
        seed: Random seed for reproducible shuffling

    Returns:
        List of text strings

    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    # Use default data file if none specified
    if data_file is None:
        data_path = _DEFAULT_DATA_FILE
    else:
        data_path = Path(data_file)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Expected location: {data_path.absolute()}\n"
            f"Please ensure pile_samples.json exists at this location."
        )

    with open(data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Extract text field from each sample
    texts = [s['text'] for s in samples]

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(texts)

    # Limit to requested number
    if num_samples is not None:
        texts = texts[:num_samples]

    return texts


def get_data_dir() -> Path:
    """
    Get the path to the data directory.

    Returns:
        Path object pointing to the data directory
    """
    return _DATA_DIR


def get_default_data_file() -> Path:
    """
    Get the path to the default pile_samples.json file.

    Returns:
        Path object pointing to pile_samples.json
    """
    return _DEFAULT_DATA_FILE
