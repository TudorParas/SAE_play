"""
Data loader for text samples.

Loads from named sources (openwebtext, wikitext, c4) stored as Parquet in data/.
For multi-source mixing, see mixer.py.
"""

import random

from src.sae.data.sources.sources_registry import get_source_path, source_exists, list_available_sources
from src.sae.data.sources.base import load_samples as load_parquet


def load_samples(
    source: str,
    num_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[str]:
    """
    Load text samples from a named source.

    Sources are stored as Parquet files in the data/ directory.
    Use the downloader scripts to fetch data:
        python -m src.sae.data.sources.download_wikitext
        python -m src.sae.data.sources.download_c4

    Args:
        source: Source name (e.g., "openwebtext", "wikitext", "c4")
        num_samples: How many samples to load (None = load all)
        shuffle: Whether to shuffle before selection
        seed: Random seed for reproducible shuffling

    Returns:
        List of text strings

    Raises:
        FileNotFoundError: If the source data doesn't exist
    """
    if not source_exists(source):
        path = get_source_path(source)
        available = ", ".join(list_available_sources())
        raise FileNotFoundError(
            f"Source '{source}' not found at: {path}\n"
            f"Run: python -m src.sae.data.sources.{source}\n"
            f"Available sources: {available}"
        )

    # Load from Parquet
    path = get_source_path(source)
    df = load_parquet(path)
    texts = df["text"].tolist()

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(texts)

    # Limit to requested number
    if num_samples is not None:
        texts = texts[:num_samples]

    return texts
