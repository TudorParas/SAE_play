"""
Helper functions to load training data from The Pile samples.

Use this after running download_pile_samples.py to get your training texts.
"""

import json
from pathlib import Path
from typing import List
import random


def load_pile_samples(
    data_file: str = "data/pile_samples.json",
    num_samples: int = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[str]:
    """
    Load text samples from the saved Pile dataset file.

    Args:
        data_file: Path to the JSON file with samples
        num_samples: How many samples to load (None = all)
        shuffle: Whether to shuffle the samples
        seed: Random seed for shuffling

    Returns:
        List of text strings
    """

    data_path = Path(data_file)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n\n"
            f"Please run this first:\n"
            f"  python src/1_initial/download_pile_samples.py\n"
        )

    print(f"Loading samples from {data_file}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Extract just the text (samples have 'text' and 'meta' fields)
    texts = [s['text'] for s in samples]

    if shuffle:
        random.seed(seed)
        random.shuffle(texts)

    # Limit to requested number
    if num_samples is not None:
        texts = texts[:num_samples]

    print(f"Loaded {len(texts)} text samples")

    return texts


if __name__ == "__main__":
    # Test loading
    try:
        texts = load_pile_samples(num_samples=5)
        print("\n✓ Successfully loaded samples!")
        print("\nFirst sample preview (200 chars):")
        print("-"*60)
        print(texts[0][:200] + "...")
        print("-"*60)
    except FileNotFoundError as e:
        print(f"\n{e}")