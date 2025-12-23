"""
C4 (Colossal Clean Crawled Corpus) dataset downloader.

Usage:
    python -m src.sae.data.sources.download_c4
    python -m src.sae.data.sources.download_c4 --num-samples 20000
"""
from src.sae.data.sources.downloader_factory import create_downloader

download, main = create_downloader(
    source_name="c4",
    dataset_key="allenai/c4",
    dataset_variant="en",
    description="C4 is a cleaned version of Common Crawl (~750GB, uses streaming).",
)

if __name__ == "__main__":
    main()
