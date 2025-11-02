"""
Download training text samples for SAE training.

Uses OpenWebText (38GB web text corpus, similar to GPT-2 training data).
Streams samples without downloading the full dataset - works reliably on Windows.

Note: While Pythia was trained on The Pile, OpenWebText provides similar
diverse web text that's readily accessible and works across platforms.

Usage:
    python download_pile_samples.py

This will create: data/pile_samples.json with 10,000 text samples
"""

import json
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import random


def download_pile_samples(
    num_samples: int = 10000,
    output_file: str = "data/pile_samples.json",
    min_length: int = 100,
    max_length: int = 2000,
    seed: int = 42
):
    """
    Stream from The Pile and save a subset of samples to disk.

    Args:
        num_samples: Number of text samples to collect
        output_file: Where to save the samples (JSON format)
        min_length: Minimum character length for a sample (filter out tiny texts)
        max_length: Maximum character length (filter out huge texts)
        seed: Random seed for reproducibility
    """

    random.seed(seed)

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Downloading samples from The Pile")
    print("="*60)
    print(f"Target samples: {num_samples}")
    print(f"Length range: {min_length}-{max_length} characters")
    print(f"Output: {output_file}")
    print()

    # The Pile is available on HuggingFace, but it's huge
    # We'll use streaming mode to avoid downloading the whole thing
    print("Connecting to The Pile dataset (streaming mode)...")
    print("Note: First connection might take a moment...\n")

    # Use OpenWebText - it's reliable, works on Windows, and similar to GPT-2 training data
    # OpenWebText is ~38GB but we stream it so we don't download everything
    print("Using OpenWebText dataset (similar to GPT-2 training data)")
    print("Streaming mode - not downloading the full dataset\n")

    try:
        dataset = load_dataset(
            "openwebtext",
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"\n⚠️  Error with openwebtext: {e}")
        print("Trying alternative: wikitext-103-v1...")

        # Even more reliable fallback: WikiText
        dataset = load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split="train",
            streaming=True
        )

    samples = []
    seen_texts = set()  # Avoid duplicates

    # Stream through the dataset
    print("Streaming and collecting samples...")
    print("(This will take a few minutes)\n")

    with tqdm(total=num_samples, desc="Collecting samples") as pbar:
        for item in dataset:
            # OpenWebText format: {'text': '...'}
            text = item['text']

            # Filter by length
            if len(text) < min_length or len(text) > max_length:
                continue

            # Avoid duplicates (check first 100 chars as fingerprint)
            fingerprint = text[:100]
            if fingerprint in seen_texts:
                continue

            seen_texts.add(fingerprint)
            samples.append({
                'text': text,
                'meta': {'source': 'openwebtext'}  # Simple metadata
            })

            pbar.update(1)

            # Stop when we have enough
            if len(samples) >= num_samples:
                break

    # Shuffle samples for diversity
    random.shuffle(samples)

    # Save to disk
    print(f"\nSaving {len(samples)} samples to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print("✓ Done!")
    print()
    print("="*60)
    print("Dataset Statistics")
    print("="*60)

    # Show some statistics
    total_chars = sum(len(s['text']) for s in samples)
    avg_length = total_chars / len(samples)

    print(f"Total samples: {len(samples)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average length: {avg_length:.0f} characters")

    # Show source distribution if available
    if samples and 'meta' in samples[0] and 'pile_set_name' in samples[0]['meta']:
        sources = {}
        for s in samples:
            source = s['meta'].get('pile_set_name', 'unknown')
            sources[source] = sources.get(source, 0) + 1

        print("\nSource distribution:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {source}: {count} ({count/len(samples)*100:.1f}%)")

    print()
    print("Sample preview (first 200 chars of first sample):")
    print("-"*60)
    print(samples[0]['text'][:200] + "...")
    print("-"*60)
    print()
    print(f"✓ Samples ready! Use this file in your SAE training scripts.")
    print(f"  File location: {output_path.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download samples from The Pile dataset")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to collect (default: 10000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/pile_samples.json",
        help="Output file path (default: data/pile_samples.json)"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=100,
        help="Minimum text length in characters (default: 100)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2000,
        help="Maximum text length in characters (default: 2000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    download_pile_samples(
        num_samples=args.num_samples,
        output_file=args.output,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed
    )