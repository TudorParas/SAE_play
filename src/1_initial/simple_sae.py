"""
Simple Sparse Autoencoder (SAE) applied to Pythia model activations

This script demonstrates the basics of mechanistic interpretability with SAEs:
1. Load a small Pythia model
2. Extract activations from a specific layer
3. Train a sparse autoencoder on those activations
4. See what sparse features we learn

Keep it simple - this is for learning, not production!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from load_pile_data import load_pile_samples
from tqdm import tqdm
from pathlib import Path
import json
import signal
import sys
import time

# Add parent directory to path to import util module
sys.path.insert(0, str(Path(__file__).parent.parent))
from util.logging import setup_logger, format_time

# Setup logger for this module
logger = setup_logger("SAE")


# ============================================================================
# PART 0: Helper Functions
# ============================================================================

def format_number_si(num: int) -> str:
    """
    Format a number with SI suffixes (K, M, B).

    Examples:
        1500 -> "2k"
        262321 -> "262k"
        1500000 -> "2m"
    """
    if num >= 1_000_000_000:
        return f"{num // 1_000_000_000}b"
    elif num >= 1_000_000:
        return f"{num // 1_000_000}m"
    elif num >= 1_000:
        return f"{num // 1_000}k"
    else:
        return str(num)


class GracefulInterruptHandler:
    """
    Handler for graceful shutdown on interrupt (Ctrl+C or PyCharm Stop).

    Uses threading.Event to safely handle SIGINT/SIGTERM without doing I/O
    inside the signal handler.

    Waits for current epoch to complete before exiting - no stdin prompt
    needed (avoids stdin corruption from PyCharm Stop button).
    """

    def __init__(self):
        import threading
        self.shutdown_requested = threading.Event()

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._request_shutdown)
        try:
            signal.signal(signal.SIGTERM, self._request_shutdown)
        except Exception:
            pass  # SIGTERM not available on all platforms

    def _request_shutdown(self, signum, frame):
        """Signal handler: just set a flag, don't do I/O here"""
        self.shutdown_requested.set()

    def should_exit(self):
        """Check if shutdown was requested"""
        return self.shutdown_requested.is_set()


# ============================================================================
# PART 1: Define the Sparse Autoencoder
# ============================================================================

class SparseAutoencoder(nn.Module):
    """
    A simple sparse autoencoder.

    Architecture:
    - Encoder: Linear layer that expands dimensions (e.g., 512 -> 2048)
    - Decoder: Linear layer that contracts back (e.g., 2048 -> 512)
    - Sparsity: Enforced via L1 penalty on activations

    Why wider (overcomplete)? Gives space for different features to occupy
    separate dimensions instead of being "compressed" together.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Args:
            input_dim: Size of the activations we're encoding (e.g., 512)
            hidden_dim: Size of sparse representation (typically 2-8x larger)
        """
        super().__init__()

        # Encoder: maps activations to sparse features
        self.encoder = nn.Linear(input_dim, hidden_dim)

        # Decoder: reconstructs original activations from sparse features
        self.decoder = nn.Linear(hidden_dim, input_dim)

        # We'll use ReLU activation to ensure non-negative sparse features
        # (negative features are harder to interpret)

        # Initialize weights properly using Xavier initialization
        # This helps with optimization compared to PyTorch's default
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        """
        Args:
            x: Model activations, shape (batch_size, input_dim)

        Returns:
            reconstructed: Reconstructed activations
            sparse_features: The sparse representation (what we want to interpret!)
        """
        # Encode to sparse features
        sparse_features = F.relu(self.encoder(x))

        # Decode back to original space
        reconstructed = self.decoder(sparse_features)

        return reconstructed, sparse_features


# ============================================================================
# PART 2: Get activations from Pythia model
# ============================================================================

def get_model_activations(
    model,
    tokenizer,
    texts: List[str],
    layer_idx: int = None,
    batch_size: int = 8
) -> torch.Tensor:
    """
    Extract per-token activations from a specific layer.

    IMPORTANT: This processes texts token-by-token in an autoregressive manner.
    Each token position produces one activation vector (in context of previous tokens).

    Example:
        Text: "The cat sat"
        Tokens: [The, cat, sat]
        Returns: 3 activation vectors (one per token)

    This filtering of padding tokens is critical for SAE quality!

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        texts: List of text strings to process
        layer_idx: Which transformer layer to extract from (0 = first layer)
                   If None, uses the middle layer
        batch_size: How many texts to process at once (for GPU efficiency)

    Returns:
        Tensor of activations, shape (total_num_real_tokens, hidden_dim)
        Each row = activation for one token position (no padding!)
    """

    # Auto-detect number of layers and choose middle layer if not specified
    num_layers = model.config.num_hidden_layers
    if layer_idx is None:
        layer_idx = num_layers // 2  # Use middle layer

    # Validate layer index
    if layer_idx >= num_layers or layer_idx < 0:
        raise ValueError(f"layer_idx {layer_idx} is out of range. Model has {num_layers} layers (0-{num_layers-1})")

    logger.info(f"Model has {num_layers} layers, extracting from layer {layer_idx}")
    logger.info(f"Processing {len(texts)} texts in batches of {batch_size}...")

    device = next(model.parameters()).device
    all_activations = []

    # Process in batches for GPU efficiency
    # But we'll collect individual token activations (not whole text activations)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize this batch
        # This creates sequences with padding: [token1, token2, PAD, PAD, ...]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Move to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run forward pass
        # The model processes all tokens in parallel (with causal masking)
        # Each token "sees" the tokens before it (autoregressive)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract activations from the target layer
        # Shape: (batch_size, seq_len, hidden_dim)
        # Position [b, t, :] = activation when processing token t in text b
        hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 to skip embeddings

        # CRITICAL: Filter out padding tokens!
        # attention_mask[b, t] = 1 for real tokens, 0 for padding
        attention_mask = inputs['attention_mask']

        # Collect activations for real tokens only (skip padding)
        for b in range(hidden_states.shape[0]):
            # Find which positions are real tokens (not padding)
            real_token_positions = attention_mask[b] == 1
            num_real_tokens = real_token_positions.sum().item()

            # Extract activation vectors for real tokens only
            # Each real token contributes one activation vector to our dataset
            text_activations = hidden_states[b, :num_real_tokens, :]

            all_activations.append(text_activations)

    # Concatenate all token activations into one big tensor
    # Final shape: (total_num_tokens_across_all_texts, hidden_dim)
    # If you had 1000 texts with ~50 tokens each → ~50,000 activation vectors
    all_activations = torch.cat(all_activations, dim=0)

    logger.info(f"Collected {all_activations.shape[0]} token activations (padding filtered out)")
    logger.info(f"Average tokens per text: {all_activations.shape[0] / len(texts):.1f}")

    return all_activations


# ============================================================================
# PART 3: Train the SAE
# ============================================================================

def train_sae(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    sparsity_penalty: float = 1e-3,
    sparsity_penalty_end: float = None,
    warmup_epochs: int = 20,
    checkpoint_dir: str = None,
    model_name: str = "EleutherAI/pythia-70m",
    layer_idx: int = 3
):
    """
    Train the sparse autoencoder on collected activations.

    Loss = Reconstruction Loss + Sparsity Penalty (scheduled)

    - Reconstruction Loss: How well can we rebuild the original activations?
    - Sparsity Penalty: Encourages most features to be zero (L1 regularization)

    SPARSITY ANNEALING:
    We can use a schedule to gradually increase sparsity pressure:
    - Early epochs: Low penalty → learn good reconstruction first
    - Later epochs: High penalty → compress to sparse representation
    This curriculum approach often works better than fixed penalty!

    Args:
        sae: The sparse autoencoder to train
        activations: Collected activations from the model
        num_epochs: How many times to iterate over the data
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        sparsity_penalty: Starting weight of L1 penalty (or fixed if no schedule)
        sparsity_penalty_end: Final weight of L1 penalty (enables schedule if set)
        warmup_epochs: Number of epochs before starting to increase penalty
        checkpoint_dir: Directory to save checkpoint if interrupted (None = no checkpointing)
        model_name: Model name for checkpoint metadata
        layer_idx: Layer index for checkpoint metadata

    Returns:
        mean: The mean of the activations (needed for inference)
    """

    # IMPORTANT: Center the activations by subtracting the mean
    # This is standard practice for SAEs and helps with training stability
    # The SAE learns to reconstruct centered activations
    mean = activations.mean(dim=0, keepdim=True)
    centered_activations = activations - mean

    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    # Set up graceful interrupt handler (Ctrl+C / PyCharm Stop)
    interrupt_handler = GracefulInterruptHandler()

    num_samples = centered_activations.shape[0]

    logger.info(f"Training SAE on {num_samples} activation vectors...")
    logger.info(f"Input dim: {sae.encoder.in_features}, Hidden dim: {sae.encoder.out_features}")

    # Setup sparsity schedule
    use_schedule = sparsity_penalty_end is not None
    if use_schedule:
        logger.info("Using Sparsity Annealing Schedule:")
        logger.info(f"   Epochs 0-{warmup_epochs}: Fixed at {sparsity_penalty:.2e}")
        logger.info(f"   Epochs {warmup_epochs}-{num_epochs}: Ramp {sparsity_penalty:.2e} → {sparsity_penalty_end:.2e}")
    else:
        logger.info(f"Fixed sparsity penalty: {sparsity_penalty:.2e}")

    logger.info("="*60)
    logger.info("TRAINING PROGRESS")
    logger.info("="*60)
    logger.info("Goal: Active% ↓ (fewer active = more sparse = better)")
    logger.info("="*60)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # Calculate current sparsity penalty (with schedule if enabled)
        if use_schedule:
            if epoch < warmup_epochs:
                # Warmup phase: use starting penalty
                current_penalty = sparsity_penalty
            else:
                # Annealing phase: linear increase from start to end
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                current_penalty = sparsity_penalty + progress * (sparsity_penalty_end - sparsity_penalty)
        else:
            current_penalty = sparsity_penalty

        # Shuffle data each epoch
        perm = torch.randperm(num_samples)

        total_loss = 0
        total_recon_loss = 0
        total_sparsity_loss = 0
        total_pct_active = 0  # Track percentage of active features
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            # Get batch
            batch_indices = perm[i:i + batch_size]
            batch = centered_activations[batch_indices]

            # Forward pass
            reconstructed, sparse_features = sae(batch)

            # Reconstruction loss: mean squared error
            recon_loss = F.mse_loss(reconstructed, batch)

            # Sparsity loss: L1 norm (sum of absolute values)
            # We want this to be small = most features are zero
            sparsity_loss = torch.abs(sparse_features).mean()

            # Track actual sparsity: what % of features are active?
            # A feature is "active" if it's > 0.01
            num_active = (sparse_features > 0.01).float().sum(dim=1).mean().item()
            pct_active = (num_active / sparse_features.shape[1]) * 100

            # Total loss (using current scheduled penalty)
            loss = recon_loss + current_penalty * sparsity_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            total_pct_active += pct_active
            num_batches += 1

        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_recon = total_recon_loss / num_batches
        avg_sparsity = total_sparsity_loss / num_batches
        avg_pct_active = total_pct_active / num_batches
        avg_num_active = (avg_pct_active / 100) * sae.encoder.out_features

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Log epoch metrics with timing
        logger.info(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:7.4f} | Recon: {avg_recon:7.4f} | L1: {avg_sparsity:6.4f} | Active: {avg_num_active:6.0f}/{sae.encoder.out_features} ({avg_pct_active:4.1f}%) | Time: {format_time(epoch_time)}")

        # Check if interrupted by Ctrl+C or PyCharm Stop button
        if interrupt_handler.should_exit():
            logger.warning("Training interrupted by user")
            logger.info("="*60)
            logger.info("TRAINING INTERRUPTED")
            logger.info("="*60)

            if checkpoint_dir is not None:
                checkpoint_path = f"{checkpoint_dir}_epoch{epoch+1}"
                logger.info(f"Saving checkpoint to: {checkpoint_path}")

                try:
                    save_sae(sae, mean, checkpoint_path, model_name, layer_idx)
                    logger.info("Checkpoint saved successfully!")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")
            else:
                logger.info("(No checkpoint directory specified, cannot save)")

            logger.info("Exiting...")
            sys.exit(0)

    logger.info("="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)

    return mean


# ============================================================================
# PART 4: Save and Load SAE
# ============================================================================

def save_sae(
    sae: SparseAutoencoder,
    mean: torch.Tensor,
    save_dir: str,
    model_name: str = "EleutherAI/pythia-70m",
    layer_idx: int = 3
):
    """
    Save the trained SAE weights and metadata.

    Args:
        sae: Trained sparse autoencoder
        mean: Mean of training activations
        save_dir: Directory to save the SAE (e.g., "checkpoints/sae_001")
        model_name: Name of the model this SAE was trained on
        layer_idx: Which layer the SAE was trained on
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    weights_path = save_path / "sae_weights.pt"
    torch.save(sae.state_dict(), weights_path)

    # Save mean (needed for inference)
    mean_path = save_path / "activation_mean.pt"
    torch.save(mean, mean_path)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "layer_idx": layer_idx,
        "input_dim": sae.encoder.in_features,
        "hidden_dim": sae.encoder.out_features,
    }
    metadata_path = save_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ SAE saved to: {save_path.absolute()}")
    print(f"  - Weights: {weights_path.name}")
    print(f"  - Mean: {mean_path.name}")
    print(f"  - Metadata: {metadata_path.name}")


def load_sae(save_dir: str, device: str = "cpu"):
    """
    Load a trained SAE from disk.

    Args:
        save_dir: Directory where the SAE was saved
        device: Device to load the SAE onto ("cpu" or "cuda")

    Returns:
        sae: Loaded sparse autoencoder
        mean: Activation mean for centering
        metadata: Dictionary with training info
    """
    save_path = Path(save_dir)

    if not save_path.exists():
        raise FileNotFoundError(f"SAE directory not found: {save_dir}")

    # Load metadata
    metadata_path = save_path / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create SAE with same architecture
    sae = SparseAutoencoder(
        input_dim=metadata["input_dim"],
        hidden_dim=metadata["hidden_dim"]
    )

    # Load weights
    weights_path = save_path / "sae_weights.pt"
    sae.load_state_dict(torch.load(weights_path, map_location=device))
    sae.to(device)
    sae.eval()  # Set to evaluation mode

    # Load mean
    mean_path = save_path / "activation_mean.pt"
    mean = torch.load(mean_path, map_location=device)

    print(f"✓ SAE loaded from: {save_path.absolute()}")
    print(f"  - Model: {metadata['model_name']}")
    print(f"  - Layer: {metadata['layer_idx']}")
    print(f"  - Architecture: {metadata['input_dim']} → {metadata['hidden_dim']} → {metadata['input_dim']}")

    return sae, mean, metadata


# ============================================================================
# PART 5: Analyze what we learned
# ============================================================================

def analyze_features(
    sae: SparseAutoencoder,
    model,
    tokenizer,
    test_texts: List[str],
    mean: torch.Tensor,
    layer_idx: int = None,
    top_k: int = 10
):
    """
    See which features activate for different texts.

    This is where interpretability happens! We can see which sparse features
    "light up" for different inputs.

    Args:
        sae: Trained sparse autoencoder
        model: Language model
        tokenizer: Tokenizer for the model
        test_texts: List of texts to analyze
        mean: Mean of training activations (for centering)
        layer_idx: Which layer to extract activations from
        top_k: How many top features to display
    """

    print("\n" + "="*60)
    print("ANALYZING LEARNED FEATURES")
    print("="*60)

    for text in test_texts:
        print(f"\nText: '{text}'")

        # Get activations for this text
        activations = get_model_activations(model, tokenizer, [text], layer_idx)

        # Center activations using the training mean
        centered_activations = activations - mean

        # Pass through SAE to get sparse features
        with torch.no_grad():
            reconstructed, sparse_features = sae(centered_activations)

        # Average features across all tokens in this text
        avg_features = sparse_features.mean(dim=0)

        # Find top-k most active features
        top_values, top_indices = torch.topk(avg_features, k=top_k)

        print(f"  Top {top_k} active features:")
        for idx, val in zip(top_indices, top_values):
            print(f"    Feature {idx.item()}: {val.item():.3f}")

        # Show sparsity: what % of features are effectively zero?
        threshold = 0.01  # Consider values < 0.01 as "zero"
        pct_active = (avg_features > threshold).float().mean().item() * 100
        print(f"  Sparsity: {pct_active:.1f}% of features active (threshold={threshold})")


# ============================================================================
# MAIN: Put it all together
# ============================================================================

def main():
    """
    Main function: Load model, collect activations, train SAE, analyze results.
    """

    print("Loading Pythia-70M model...")
    # Using smallest Pythia model for speed (70M parameters)
    # You can use larger ones: pythia-160m, pythia-410m, etc.
    model_name = "EleutherAI/pythia-70m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-style models don't have a padding token by default
    # Standard practice: use the EOS (end of sequence) token for padding
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Load training texts from The Pile
    # This uses real data from Pythia's training distribution
    print("\nLoading training texts from The Pile...")
    training_texts = load_pile_samples(
        data_file="data/pile_samples.json",
        num_samples=10000,  # Use 1000 texts for training
        shuffle=True
    )

    print(f"\nCollecting activations from middle layer...")
    # Get activations from middle layer (auto-detected)
    activations = get_model_activations(model, tokenizer, training_texts)
    activations = activations.to(device)

    print(f"Collected {activations.shape[0]} activation vectors")
    print(f"Each vector has {activations.shape[1]} dimensions")

    # Create SAE with 4x expansion (overcomplete representation)
    """
      | Model Size     | Typical Expansion | Hidden Dim Example |
      |----------------|-------------------|--------------------|
      | Small (512d)   | 4x-8x             | 2k-4k              |
      | Medium (2048d) | 8x-16x            | 16k-32k            |
      | Large (4096d)  | 16x-32x           | 64k-128k           |
      | Gemma Scope    | Up to 64x         | 262k for 4k input  |
    """
    input_dim = activations.shape[1]
    hidden_dim = input_dim * 27 # input_dim ** 2  # 4x wider for sparse features

    print(f"\nCreating SAE: {input_dim} -> {hidden_dim} -> {input_dim}")
    sae = SparseAutoencoder(input_dim, hidden_dim).to(device)
    layer_idx = 3
    # Train the SAE with sparsity annealing
    print("\nTraining SAE...")
    mean = train_sae(
        sae,
        activations,
        num_epochs=20,
        batch_size=32,
        learning_rate=1e-3,
        sparsity_penalty=1e-2,  # Start
        sparsity_penalty_end=2.0,  # Much higher!
        warmup_epochs=2,             # Keep low penalty for first 20 epochs
        checkpoint_dir=f"checkpoints/pythia-70m_layer{layer_idx}_{format_number_si(hidden_dim)}_interrupted",  # Enable Ctrl+C checkpoints
        model_name=model_name,
        layer_idx=layer_idx
    )

    # Test on some new examples
    test_texts = [
        "Dogs are man's best friend.",
        "Neural networks process information.",
        "London is a major city.",
    ]

    analyze_features(sae, model, tokenizer, test_texts, mean)

    # Save the trained SAE
    save_sae(
        sae,
        mean,
        save_dir=f"checkpoints/pythia-70m_layer{layer_idx}_{format_number_si(hidden_dim)}",
        model_name=model_name,
        layer_idx=layer_idx  # Middle layer
    )

    print("\n" + "="*60)
    print("DONE! You've trained your first SAE on LLM activations!")
    print("="*60)


if __name__ == "__main__":
    main()
