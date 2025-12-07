"""
C4 (Colossal Clean Crawled Corpus) dataset downloader.

C4 is a cleaned version of Common Crawl, used to train T5 and other models.
Very large dataset (~750GB), uses streaming mode.

Usage:
    python -m src.sae.data.sources.download_c4
    python -m src.sae.data.sources.download_c4 --num-samples 20000
"""
import click
import random
from pathlib import Path
from datasets import load_dataset

from src.sae.data.sources.sources_registry import get_source_path
from src.sae.data.sources.base import stream_and_filter, save_samples, print_download_stats


SOURCE_NAME = "c4"


def download(
    num_samples: int = 10000,
    output_path: Path | str | None = None,
    min_length: int = 100,
    max_length: int = 2000,
    seed: int = 42,
) -> Path:
    """
    Download samples from C4 dataset.

    Args:
        num_samples: Number of samples to collect
        output_path: Output path. None = use default (data/c4.parquet)
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters
        seed: Random seed for shuffling

    Returns:
        Path to the saved Parquet file
    """
    random.seed(seed)

    if output_path is None:
        output_path = get_source_path(SOURCE_NAME)
    else:
        output_path = Path(output_path)

    print("=" * 60)
    print("Downloading C4 samples")
    print("=" * 60)
    print(f"Target samples: {num_samples:,}")
    print(f"Length range: {min_length}-{max_length} characters")
    print(f"Output: {output_path}")
    print()

    # Load dataset in streaming mode
    # C4 is huge, so streaming is essential
    print("Connecting to C4 dataset (streaming mode)...")
    print("Note: C4 is very large (~750GB), streaming avoids full download.\n")

    dataset = load_dataset(
        "allenai/c4",
        "en",  # English subset
        split="train",
        streaming=True,
    )

    # Stream and filter samples
    texts = stream_and_filter(
        dataset_iterator=iter(dataset),
        text_key="text",
        min_length=min_length,
        max_length=max_length,
        num_samples=num_samples,
        deduplicate=True,
    )

    # Shuffle for diversity
    random.shuffle(texts)

    # Save to Parquet
    output_path = save_samples(texts, SOURCE_NAME, output_path)

    # Print stats
    print_download_stats(texts, SOURCE_NAME, output_path)

    return output_path


@click.command()
@click.option(
    "--num-samples",
    default=10000,
    help="Number of samples to collect",
    show_default=True,
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output file path (default: data/c4.parquet)",
)
@click.option(
    "--min-length",
    default=100,
    help="Minimum text length in characters",
    show_default=True,
)
@click.option(
    "--max-length",
    default=2000,
    help="Maximum text length in characters",
    show_default=True,
)
@click.option(
    "--seed",
    default=42,
    help="Random seed",
    show_default=True,
)
def main(num_samples, output, min_length, max_length, seed):
    """Download samples from C4 dataset."""
    download(
        num_samples=num_samples,
        output_path=output,
        min_length=min_length,
        max_length=max_length,
        seed=seed,
    )

if __name__ == "__main__":
    main()
