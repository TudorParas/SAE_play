"""
Training loop for SAEs.

This is the thin "pipeline" layer that coordinates training.
"""

import torch
from typing import Dict, Any, List
from ..models.base import BaseSAE
from .trainer import SAETrainer
from .train_metrics import TrainMetrics


def train_sae(
    sae: BaseSAE,
    trainer: SAETrainer,
    activations: torch.Tensor,
    num_epochs: int = 100,
    batch_size: int = 32,
    use_l1_penalty: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a Sparse Autoencoder on collected activations.

    This is the main training loop that:
    1. Centers the activations (subtracts mean)
    2. Runs training epochs with batching
    3. Tracks metrics
    4. Returns results

    Args:
        sae: The SAE model to train
        trainer: Trainer object (holds optimizer and training logic)
        activations: Tensor of activations, shape (num_samples, input_dim)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        use_l1_penalty: Whether to use L1 sparsity penalty.
                       Set to False if using TopK activation.
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
            - 'activation_mean': Mean of input activations (needed for inference)
            - 'loss_history': List of average losses per epoch
            - 'recon_loss_history': List of reconstruction losses
            - 'sparsity_history': List of sparsity percentages
            - 'final_metrics': TrainMetrics object with final epoch metrics
    """
    # Center activations (standard practice for SAEs)
    activation_mean = activations.mean(dim=0, keepdim=True)
    centered_activations = activations - activation_mean

    num_samples = centered_activations.shape[0]
    device = next(sae.parameters()).device

    # Move activations to same device as model
    centered_activations = centered_activations.to(device)

    if verbose:
        print(f"\nTraining SAE on {num_samples:,} activation vectors")
        print(f"Input dim: {sae.input_dim}, Latent dim: {sae.latent_dim}")
        print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
        print("=" * 60)

    # Track metrics over time
    loss_history = []
    recon_loss_history = []
    sparsity_history = []

    # Training loop
    for epoch in range(num_epochs):
        # Shuffle data each epoch
        perm = torch.randperm(num_samples)

        # Initialize epoch metrics
        epoch_metrics = TrainMetrics()
        num_batches = 0

        # Batch iteration
        for i in range(0, num_samples, batch_size):
            batch_indices = perm[i:i + batch_size]
            batch = centered_activations[batch_indices]

            # Training step
            batch_metrics = trainer.train_step(batch, use_l1_penalty=use_l1_penalty)

            # Accumulate metrics
            epoch_metrics.update(batch_metrics)
            num_batches += 1

        # Average metrics over batches
        epoch_metrics.scale(1.0 / num_batches)

        # Store history
        loss_history.append(epoch_metrics.loss)
        recon_loss_history.append(epoch_metrics.recon_loss)
        sparsity_history.append(epoch_metrics.pct_active)

        # Print progress
        if verbose:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Loss: {epoch_metrics.loss:.4f} | "
                f"Recon: {epoch_metrics.recon_loss:.4f} | "
                f"Active: {epoch_metrics.num_active:.0f}/{sae.latent_dim} "
                f"({epoch_metrics.pct_active:.1f}%)"
            )

    if verbose:
        print("=" * 60)
        print("Training complete!\n")

    return {
        'activation_mean': activation_mean,
        'loss_history': loss_history,
        'recon_loss_history': recon_loss_history,
        'sparsity_history': sparsity_history,
        'final_metrics': epoch_metrics,
    }
