"""
Base utilities for dataset downloaders.

Shared functions for saving/loading samples in Parquet format.
"""

import pandas as pd
from pathlib import Path
from typing import Iterator

from src.sae.data.sources.sources_registry import get_data_dir


def save_samples(
    texts: list[str],
    source_name: str,
    path: Path | None = None,
) -> Path:
    """
    Save text samples as a Parquet file.

    Args:
        texts: List of text strings
        source_name: Name of the data source (stored as metadata)
        path: Output path. If None, uses default location in data/ directory.

    Returns:
        Path to the saved file
    """
    if path is None:
        path = get_data_dir() / f"{source_name}.parquet"

    # Create directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame and save
    df = pd.DataFrame({
        "text": texts,
        "source": source_name,
    })
    df.to_parquet(path, index=False)

    return path


def load_samples(path: Path) -> pd.DataFrame:
    """
    Load samples from a Parquet file.

    Args:
        path: Path to Parquet file

    Returns:
        DataFrame with 'text' and 'source' columns
    """
    return pd.read_parquet(path)


def stream_and_filter(
    dataset_iterator: Iterator,
    text_key: str = "text",
    min_length: int = 100,
    max_length: int = 2000,
    num_samples: int = 10000,
    deduplicate: bool = True,
) -> list[str]:
    """
    Stream through a HuggingFace dataset and collect filtered samples.

    Common utility for all downloaders that stream from HuggingFace datasets.

    Args:
        dataset_iterator: Iterator over dataset items (streaming mode)
        text_key: Key to extract text from each item
        min_length: Minimum character length
        max_length: Maximum character length
        num_samples: Number of samples to collect
        deduplicate: If True, skip duplicate texts (by first 100 chars)

    Returns:
        List of filtered text strings
    """
    from tqdm import tqdm

    samples = []
    seen_fingerprints: set[str] = set()

    with tqdm(total=num_samples, desc="Collecting samples") as pbar:
        for item in dataset_iterator:
            text = item.get(text_key, "")

            # Skip empty
            if not text:
                continue

            # Filter by length
            if len(text) < min_length or len(text) > max_length:
                continue

            # Deduplicate by fingerprint
            if deduplicate:
                fingerprint = text[:100]
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)

            samples.append(text)
            pbar.update(1)

            if len(samples) >= num_samples:
                break

    return samples


def print_download_stats(texts: list[str], source_name: str, output_path: Path):
    """
    Print statistics about downloaded samples.

    Args:
        texts: List of downloaded texts
        source_name: Name of the data source
        output_path: Where the file was saved
    """
    total_chars = sum(len(t) for t in texts)
    avg_length = total_chars / len(texts) if texts else 0

    print("\n" + "=" * 60)
    print(f"Download Complete: {source_name}")
    print("=" * 60)
    print(f"Total samples: {len(texts):,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average length: {avg_length:.0f} characters")
    print(f"Saved to: {output_path.absolute()}")
    print("=" * 60)

    if texts:
        print("\nSample preview (first 200 chars):")
        print("-" * 60)
        print(texts[0][:200] + "...")
        print("-" * 60)
