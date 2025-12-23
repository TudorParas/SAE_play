"""
OpenWebText dataset downloader.

Usage:
    python -m src.sae.data.sources.download_openwebtext
    python -m src.sae.data.sources.download_openwebtext --num-samples 20000
"""
from src.sae.data.sources.downloader_factory import create_downloader

download, main = create_downloader(
    source_name="openwebtext",
    dataset_key="openwebtext",
    description="OpenWebText is a ~38GB web text corpus similar to GPT-2's training data.",
)

if __name__ == "__main__":
    main()
