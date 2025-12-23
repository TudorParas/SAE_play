"""
Factory for creating dataset downloaders.

Creates standardized download functions and CLI commands for HuggingFace datasets.
"""
import click
import random
from pathlib import Path
from datasets import load_dataset

from src.sae.data.sources.sources_registry import get_source_path
from src.sae.data.sources.base import stream_and_filter, save_samples, print_download_stats


def create_downloader(
    source_name: str,
    dataset_key: str,
    dataset_variant: str | None = None,
    text_key: str = "text",
    split: str = "train",
    description: str = "",
):
    """
    Factory to create dataset downloader functions and CLI commands.

    Args:
        source_name: Short name for the source (e.g., "wikitext", "c4")
        dataset_key: HuggingFace dataset key (e.g., "wikitext", "allenai/c4")
        dataset_variant: Dataset variant/config (e.g., "wikitext-103-v1", "en")
        text_key: Key for text field in dataset
        split: Dataset split to use
        description: Human-readable description for CLI help

    Returns:
        Tuple of (download_function, cli_main)
    """

    def download(
        num_samples: int = 10000,
        output_path: Path | str | None = None,
        min_length: int = 100,
        max_length: int = 2000,
        seed: int = 42,
    ) -> Path:
        """Download samples from the dataset."""
        random.seed(seed)

        if output_path is None:
            output_path_resolved = get_source_path(source_name)
        else:
            output_path_resolved = Path(output_path)

        print("=" * 60)
        print(f"Downloading {source_name} samples")
        print("=" * 60)
        print(f"Target samples: {num_samples:,}")
        print(f"Length range: {min_length}-{max_length} characters")
        print(f"Output: {output_path_resolved}")
        print()

        # Load dataset in streaming mode
        print(f"Connecting to {dataset_key} dataset (streaming mode)...")

        load_args = {"split": split, "streaming": True}
        if dataset_variant:
            dataset = load_dataset(dataset_key, dataset_variant, **load_args)
        else:
            dataset = load_dataset(dataset_key, **load_args)

        # Stream and filter samples
        texts = stream_and_filter(
            dataset_iterator=iter(dataset),
            text_key=text_key,
            min_length=min_length,
            max_length=max_length,
            num_samples=num_samples,
            deduplicate=True,
        )

        # Shuffle for diversity
        random.shuffle(texts)

        # Save to Parquet
        final_path = save_samples(texts, source_name, output_path_resolved)

        # Print stats
        print_download_stats(texts, source_name, final_path)

        return final_path

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
        help=f"Output file path (default: data/{source_name}.parquet)",
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
    def cli_main(num_samples, output, min_length, max_length, seed):
        """Download samples from the dataset."""
        download(
            num_samples=num_samples,
            output_path=output,
            min_length=min_length,
            max_length=max_length,
            seed=seed,
        )

    # Update docstrings with source-specific info
    download.__doc__ = f"Download samples from {source_name} dataset."
    cli_main.__doc__ = f"Download samples from {source_name} dataset. {description}"

    return download, cli_main
