"""
WikiText-103 dataset downloader.

Usage:
    python -m src.sae.data.sources.download_wikitext
    python -m src.sae.data.sources.download_wikitext --num-samples 5000
"""
from src.sae.data.sources.downloader_factory import create_downloader

download, main = create_downloader(
    source_name="wikitext",
    dataset_key="wikitext",
    dataset_variant="wikitext-103-v1",
    description="WikiText-103 contains verified Good and Featured Wikipedia articles.",
)

if __name__ == "__main__":
    main()
