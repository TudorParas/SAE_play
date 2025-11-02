"""
JumpReLU Sparse Autoencoder (SAE) applied to Pythia model activations

This implements the cutting-edge JumpReLU SAE from:
"Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders"
Rajamanoharan et al., Google DeepMind (July 2024)
https://arxiv.org/abs/2407.14435

Key innovations over standard SAE:
1. JumpReLU activation: Learnable thresholds instead of fixed ReLU
2. L0 penalty: Count active features directly, not L1 (avoids "shrinkage")
3. Straight-Through Estimator: Enables training through discontinuous function

This is state-of-the-art as of late 2024!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


# ============================================================================
# PART 1: JumpReLU Activation with Straight-Through Estimator
# ============================================================================

class JumpReLU(nn.Module):
    """
    JumpReLU activation: σ(z) = z ⊙ H(z - θ)

    Where:
    - H is the Heaviside step function (1 if input > 0, else 0)
    - θ is a learnable threshold (different for each feature)
    - ⊙ is element-wise multiplication

    Intuition: Like ReLU, but each feature learns its own threshold.
    Below threshold → hard zero. Above → pass through unchanged.

    The discontinuity requires Straight-Through Estimators (STEs) for training.
    """

    def __init__(self, num_features: int, initial_threshold: float = 0.001, bandwidth: float = 0.001):
        """
        Args:
            num_features: Number of features (hidden dimension of SAE)
            initial_threshold: Initial value for all thresholds
            bandwidth: ε parameter for STE gradient approximation
        """
        super().__init__()

        # Learnable threshold for each feature
        # Initialize small so most features start "on" (positive bias for learning)
        self.threshold = nn.Parameter(torch.ones(num_features) * initial_threshold)

        # Bandwidth for straight-through estimator gradient approximation
        self.bandwidth = bandwidth

    def forward(self, z):
        """
        Forward pass: Apply hard threshold
        Backward pass: Use smooth approximation via STE

        Args:
            z: Pre-activation values, shape (batch_size, num_features)

        Returns:
            Activated values (same shape as input)
        """
        # Compute z - θ for each feature
        z_minus_threshold = z - self.threshold

        # Forward pass: Hard Heaviside step function
        # This is what actually runs during inference
        heaviside = (z_minus_threshold > 0).float()

        # Apply JumpReLU: z ⊙ H(z - θ)
        activated = z * heaviside

        # Backward pass: Use Straight-Through Estimator
        # We can't backprop through the step function, so we approximate
        # the gradient using a smooth function (sigmoid)
        if self.training:
            # Detach the hard step, replace gradient with smooth approximation
            # This is the "straight-through" trick
            smooth_gate = torch.sigmoid(z_minus_threshold / self.bandwidth)
            activated = activated + (z * smooth_gate - z * smooth_gate.detach())

        return activated


# ============================================================================
# PART 2: Define the JumpReLU Sparse Autoencoder
# ============================================================================

class JumpReLUSparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder using JumpReLU activation and L0 penalty.

    Architecture:
    - Encoder: Linear layer with JumpReLU activation (learnable thresholds)
    - Decoder: Linear layer (no activation)
    - Sparsity: Enforced via L0 penalty (count of active features)

    Advantages over standard SAE:
    - L0 directly minimizes number of active features (not magnitude)
    - Avoids "shrinkage" problem where L1 makes features artificially small
    - State-of-the-art reconstruction quality at same sparsity level
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 initial_threshold: float = 0.001, bandwidth: float = 0.001):
        """
        Args:
            input_dim: Size of the activations we're encoding (e.g., 512)
            hidden_dim: Size of sparse representation (typically 2-8x larger)
            initial_threshold: Initial threshold for JumpReLU
            bandwidth: STE gradient approximation bandwidth
        """
        super().__init__()

        # Encoder: maps activations to sparse features (before activation)
        self.encoder = nn.Linear(input_dim, hidden_dim)

        # JumpReLU activation with learnable thresholds
        self.jumprelu = JumpReLU(hidden_dim, initial_threshold, bandwidth)

        # Decoder: reconstructs original activations from sparse features
        self.decoder = nn.Linear(hidden_dim, input_dim)

        # Initialize weights using Xavier initialization (same as standard SAE)
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
            sparse_features: The sparse representation after JumpReLU
            pre_activation: Pre-activation values (for computing L0)
        """
        # Encode to pre-activation features
        pre_activation = self.encoder(x)

        # Apply JumpReLU to get sparse features
        sparse_features = self.jumprelu(pre_activation)

        # Decode back to original space
        reconstructed = self.decoder(sparse_features)

        return reconstructed, sparse_features, pre_activation


# ============================================================================
# PART 3: Get activations from Pythia model (same as standard SAE)
# ============================================================================

def get_model_activations(
    model,
    tokenizer,
    texts: List[str],
    layer_idx: int = None
) -> torch.Tensor:
    """
    Run texts through the model and extract activations from a specific layer.

    (Same implementation as simple_sae.py - refer there for detailed comments)
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    num_layers = model.config.num_hidden_layers
    if layer_idx is None:
        layer_idx = num_layers // 2

    if layer_idx >= num_layers or layer_idx < 0:
        raise ValueError(f"layer_idx {layer_idx} out of range. Model has {num_layers} layers")

    print(f"Model has {num_layers} layers, extracting from layer {layer_idx}")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[layer_idx + 1]
    batch_size, seq_len, hidden_dim = hidden_states.shape
    all_activations = hidden_states.reshape(-1, hidden_dim)

    return all_activations


# ============================================================================
# PART 4: Train the JumpReLU SAE with L0 penalty
# ============================================================================

def train_sae(
    sae: JumpReLUSparseAutoencoder,
    activations: torch.Tensor,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    sparsity_penalty: float = 1e-3
):
    """
    Train the JumpReLU SAE on collected activations.

    Loss = Reconstruction Loss + L0 Sparsity Penalty

    Key differences from standard SAE:
    - Uses L0 penalty (count of active features) instead of L1 (sum of magnitudes)
    - Avoids "shrinkage" where L1 makes features artificially small
    - Directly optimizes what we care about: number of active features

    Args:
        sae: The JumpReLU sparse autoencoder to train
        activations: Collected activations from the model
        num_epochs: How many times to iterate over the data
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        sparsity_penalty: Weight of L0 penalty (higher = more sparse)

    Returns:
        mean: The mean of the activations (needed for inference)
    """

    # IMPORTANT: Center the activations (same as standard SAE)
    # This is critical for good feature learning
    mean = activations.mean(dim=0, keepdim=True)
    centered_activations = activations - mean

    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    num_samples = centered_activations.shape[0]

    print(f"Training JumpReLU SAE on {num_samples} activation vectors...")
    print(f"Input dim: {sae.encoder.in_features}, Hidden dim: {sae.encoder.out_features}")
    print(f"Using L0 penalty (counts active features, not magnitudes)")

    for epoch in range(num_epochs):
        # Shuffle data each epoch
        perm = torch.randperm(num_samples)

        total_loss = 0
        total_recon_loss = 0
        total_l0_loss = 0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            # Get batch
            batch_indices = perm[i:i + batch_size]
            batch = centered_activations[batch_indices]

            # Forward pass
            reconstructed, sparse_features, pre_activation = sae(batch)

            # Reconstruction loss: mean squared error
            recon_loss = F.mse_loss(reconstructed, batch)

            # L0 sparsity loss: count of active features
            # Active = where JumpReLU output is non-zero
            # = where pre_activation > threshold
            # This is the key innovation: we penalize COUNT, not MAGNITUDE
            l0_loss = (sparse_features != 0).float().mean()

            # Total loss
            loss = recon_loss + sparsity_penalty * l0_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Optional: Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)

            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_l0_loss += l0_loss.item()
            num_batches += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            avg_recon = total_recon_loss / num_batches
            avg_l0 = total_l0_loss / num_batches

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  Reconstruction: {avg_recon:.4f}")
            print(f"  L0 (fraction active): {avg_l0:.4f} ({avg_l0*100:.1f}%)")

            # Show learned threshold statistics
            thresholds = sae.jumprelu.threshold.detach()
            print(f"  Thresholds - min: {thresholds.min():.4f}, max: {thresholds.max():.4f}, mean: {thresholds.mean():.4f}")

    return mean


# ============================================================================
# PART 5: Analyze what we learned
# ============================================================================

def analyze_features(
    sae: JumpReLUSparseAutoencoder,
    model,
    tokenizer,
    test_texts: List[str],
    mean: torch.Tensor,
    layer_idx: int = None,
    top_k: int = 10
):
    """
    See which features activate for different texts.

    (Similar to standard SAE, but now features have learned thresholds)
    """

    print("\n" + "="*60)
    print("ANALYZING LEARNED FEATURES (JumpReLU SAE)")
    print("="*60)

    for text in test_texts:
        print(f"\nText: '{text}'")

        # Get activations for this text
        activations = get_model_activations(model, tokenizer, [text], layer_idx)

        # Center activations using the training mean
        centered_activations = activations - mean

        # Pass through SAE to get sparse features
        with torch.no_grad():
            reconstructed, sparse_features, pre_activation = sae(centered_activations)

        # Average features across all tokens in this text
        avg_features = sparse_features.mean(dim=0)

        # Find top-k most active features
        top_values, top_indices = torch.topk(avg_features, k=top_k)

        print(f"  Top {top_k} active features:")
        for idx, val in zip(top_indices, top_values):
            threshold = sae.jumprelu.threshold[idx].item()
            print(f"    Feature {idx.item()}: {val.item():.3f} (threshold: {threshold:.4f})")

        # Show sparsity: what % of features are active?
        num_active = (avg_features > 0).float().sum().item()
        pct_active = (num_active / len(avg_features)) * 100
        print(f"  Sparsity: {pct_active:.1f}% of features active ({int(num_active)}/{len(avg_features)})")


# ============================================================================
# MAIN: Put it all together
# ============================================================================

def main():
    """
    Main function: Load model, collect activations, train JumpReLU SAE, analyze results.
    """

    print("="*60)
    print("JumpReLU Sparse Autoencoder")
    print("State-of-the-art SAE architecture (July 2024)")
    print("="*60)

    print("\nLoading Pythia-70M model...")
    model_name = "EleutherAI/pythia-70m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Prepare training texts
    training_texts = [
        "The cat sat on the mat.",
        "Machine learning is a subset of artificial intelligence.",
        "Paris is the capital of France.",
        "To be or not to be, that is the question.",
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning was the Word.",
        "E equals mc squared is Einstein's famous equation.",
        "The mitochondria is the powerhouse of the cell.",
        "Breaking news: Scientists discover new particle.",
        "Once upon a time in a land far away.",
    ]

    print(f"\nCollecting activations from middle layer...")
    activations = get_model_activations(model, tokenizer, training_texts)
    activations = activations.to(device)

    print(f"Collected {activations.shape[0]} activation vectors")
    print(f"Each vector has {activations.shape[1]} dimensions")

    # Create JumpReLU SAE with 4x expansion
    input_dim = activations.shape[1]
    hidden_dim = input_dim ** 2

    print(f"\nCreating JumpReLU SAE: {input_dim} -> {hidden_dim} -> {input_dim}")
    sae = JumpReLUSparseAutoencoder(
        input_dim,
        hidden_dim,
        initial_threshold=0.01,  # Start with small thresholds
        bandwidth=0.001  # STE gradient approximation bandwidth
    ).to(device)

    # Train the SAE
    print("\nTraining JumpReLU SAE...")
    print("Key innovation: L0 penalty counts active features, doesn't penalize magnitude")
    mean = train_sae(
        sae,
        activations,
        num_epochs=200,
        batch_size=32,
        learning_rate=1e-3,
        sparsity_penalty=5e-2  # Can be higher than L1 since we're counting, not summing
    )

    # Test on some new examples
    test_texts = [
        "Dogs are man's best friend.",
        "Neural networks process information.",
        "London is a major city.",
    ]

    analyze_features(sae, model, tokenizer, test_texts, mean)

    print("\n" + "="*60)
    print("DONE! You've trained a state-of-the-art JumpReLU SAE!")
    print("="*60)
    print("\nCompare this with simple_sae.py to see the differences:")
    print("- JumpReLU learns per-feature thresholds (vs fixed ReLU)")
    print("- L0 penalty counts active features (vs L1 summing magnitudes)")
    print("- Better reconstruction at same sparsity level")


if __name__ == "__main__":
    main()