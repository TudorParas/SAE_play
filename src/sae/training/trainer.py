"""
SAE Trainer class.

Handles optimization and training step logic.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from ..models.base import BaseSAE
from .schedules import Schedule, ConstantSchedule
from .train_metrics import TrainMetrics


class SAETrainer:
    """
    Trainer for Sparse Autoencoders.

    Holds the optimizer and handles the training step logic (forward pass,
    loss computation, backward pass).
    """

    def __init__(
        self,
        sae: BaseSAE,
        lr: float = 1e-3,
        sparsity_penalty: Optional[Schedule] = None,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the trainer.

        Args:
            sae: The sparse autoencoder to train
            lr: Learning rate
            sparsity_penalty: Schedule for sparsity penalty (L1 coefficient).
                             If None, uses constant value of 1e-3.
                             Only used if SAE doesn't use TopK activation.
            optimizer_class: Optimizer class to use (default: Adam)
            optimizer_kwargs: Additional kwargs for optimizer
        """
        self.sae = sae
        self.lr = lr

        # Default sparsity penalty if none provided
        if sparsity_penalty is None:
            sparsity_penalty = ConstantSchedule(1e-3)
        self.sparsity_penalty = sparsity_penalty

        # Create optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = optimizer_class(sae.parameters(), lr=lr, **optimizer_kwargs)

        self.step_count = 0

    def train_step(
        self,
        batch: torch.Tensor,
        use_l1_penalty: bool = True
    ) -> TrainMetrics:
        """
        Perform one training step on a batch.

        Args:
            batch: Batch of activations, shape (batch_size, input_dim)
            use_l1_penalty: Whether to use L1 sparsity penalty.
                           Set to False if using TopK activation.

        Returns:
            TrainMetrics object with loss and sparsity information
        """
        # Forward pass
        reconstructed, sparse_features = self.sae(batch)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, batch)

        # Sparsity loss (L1 penalty on features)
        if use_l1_penalty:
            sparsity_loss = torch.abs(sparse_features).mean()
            penalty_weight = self.sparsity_penalty(self.step_count)
            total_loss = recon_loss + penalty_weight * sparsity_loss
        else:
            sparsity_loss = torch.tensor(0.0)
            total_loss = recon_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Compute sparsity metrics
        num_active = (sparse_features > 0.01).float().sum(dim=1).mean().item()

        pct_active = (num_active / sparse_features.shape[1]) * 100

        self.step_count += 1

        return TrainMetrics(
            loss=total_loss.item(),
            recon_loss=recon_loss.item(),
            sparsity_loss=sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss,
            num_active=num_active,
            pct_active=pct_active,
        )
