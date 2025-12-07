"""
Dataset source registry.

Each source has a downloader module that can fetch samples from a specific dataset.
Sources are cached as Parquet files in the project's data/ directory.
"""

from pathlib import Path

# Registry of available dataset sources
SOURCES: dict[str, dict] = {
    "openwebtext": {
        "module": "openwebtext",
        "default_path": "openwebtext.parquet",
        "description": "OpenWebText corpus (similar to GPT-2 training data)",
    },
    "wikitext": {
        "module": "wikitext",
        "default_path": "wikitext.parquet",
        "description": "WikiText-103 (Wikipedia articles)",
    },
    "c4": {
        "module": "c4",
        "default_path": "c4.parquet",
        "description": "Colossal Clean Crawled Corpus (web text)",
    },
}


def get_data_dir() -> Path:
    """
    Get the data directory (project_root/data/).

    Returns:
        Path to the data directory
    """
    # Navigate from src/sae/data/sources/ to project root, then to data/
    return Path(__file__).parent.parent.parent.parent.parent / "data"


def get_source_path(name: str) -> Path:
    """
    Get the default file path for a dataset source.

    Args:
        name: Source name (e.g., "openwebtext", "wikitext")

    Returns:
        Path to the source's Parquet file

    Raises:
        ValueError: If source name is not registered
    """
    if name not in SOURCES:
        available = ", ".join(SOURCES.keys())
        raise ValueError(f"Unknown source '{name}'. Available: {available}")

    return get_data_dir() / SOURCES[name]["default_path"]


def list_available_sources() -> list[str]:
    """
    List all registered dataset sources.

    Returns:
        List of source names
    """
    return list(SOURCES.keys())


def source_exists(name: str) -> bool:
    """
    Check if a source's data file exists.

    Args:
        name: Source name

    Returns:
        True if the source file exists and can be loaded
    """
    path = get_source_path(name)
    return path.exists()
