"""
Training loop for SAEs.

This is the thin "pipeline" layer that coordinates training.
"""
import time
from contextlib import ExitStack

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union

from src.sae.training.schedules import Schedule, ConstantSchedule
from src.sae.models.base import BaseSAE
from src.sae.training.train_metrics import TrainMetrics


class TrainPipeline:
    def __init__(
        self,
        sae: BaseSAE,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        activation_mean: Optional[torch.Tensor] = None,
        lr_schedule: Optional[Union[Schedule, torch.optim.lr_scheduler.LRScheduler]] = None,
        sparsity_schedule: Optional[Schedule] = None,
        use_amp: bool = False,
    ):
        """
        Train a Sparse Autoencoder on collected activations.

        This is the main training loop that:
        1. Centers the activations (subtracts mean) - if using raw tensor
        2. Runs training epochs with batching
        3. Tracks metrics
        4. Returns results

        Two modes of operation:
        1. Legacy mode (activations tensor): Pass raw activations, pipeline handles centering
        2. DataLoader mode (train_loader): Pass pre-centered DataLoader from ActivationDataset

        Args:
            sae: The SAE model to train
            optimizer: The optimizer to use for training
            train_loader: [PREFERRED] DataLoader from ActivationDataset (already centered).
            activation_mean: Mean tensor for centering. Required if train_loader is used.
                            Should be computed from TRAIN data only to avoid data leakage.
            lr_schedule: Learning rate schedule. Three options:
                - None: Use the optimizer's initial learning rate (no scheduling).
                - Schedule: Custom epoch-based schedule (calls schedule(epoch)).
                - torch.optim.lr_scheduler.LRScheduler: PyTorch scheduler (calls scheduler.step()).
            sparsity_schedule: Schedule for sparsity penalty coefficient (epoch-based).
                If None, uses constant coefficient of 1.0.
            use_amp: Use automatic mixed precision training with bfloat16.
                Speeds up training on modern GPUs with minimal accuracy impact.

        Example (DataLoader mode - preferred):
            >>> train_raw, test_raw = split_activations(activations, train_frac=0.9)
            >>> train_mean = train_raw.mean(dim=0, keepdim=True)
            >>> train_dataset = ActivationDataset(train_raw, mean=train_mean)
            >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            >>> pipeline = TrainPipeline(
            ...     sae=sae,
            ...     optimizer=optimizer,
            ...     train_loader=train_loader,
            ...     activation_mean=train_mean,
            ... )
        """
        self.sae = sae
        self.optimizer = optimizer
        self._device = next(sae.parameters()).device

        self._train_loader = train_loader
        self._centered_activations = None
        self._num_samples = len(train_loader.dataset)
        self._use_dataloader = True

        # Require activation mean for DataLoader mode
        if activation_mean is None:
            raise ValueError(
                "activation_mean is required when using DataLoader mode. "
                "Compute it from your training data before creating datasets."
            )
        self.activation_mean = activation_mean

        # Schedules
        self._lr_schedule = lr_schedule
        if sparsity_schedule is None:
            sparsity_schedule = ConstantSchedule(1.0)
        self._sparsity_schedule = sparsity_schedule

        # AMP setup (always use bfloat16 when enabled)
        self._amp_dtype = torch.bfloat16 if use_amp else None

    def train_step(
            self,
            batch: torch.Tensor,
            sparsity_penalty: float = 1.0,
            current_lr: float | None = None,
    ) -> TrainMetrics:
        """
        Perform one training step on a batch.

        Args:
            batch: Batch of activations, shape (batch_size, input_dim)
            sparsity_penalty: Multiplier for sparsity penalty
            current_lr: If provided, use this value for the learning rate.

        Returns:
            TrainMetrics object with loss and sparsity information
        """
        if current_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

        # Forward pass (with AMP if enabled)
        with ExitStack() as exit_stack:
            if self._amp_dtype is not None:
                exit_stack.enter_context(torch.amp.autocast(enabled=True, dtype=self._amp_dtype, device_type='cuda'))
            reconstructed, sparse_features, pre_activation = self.sae(batch)

            # Reconstruction loss (MSE)
            recon_loss = torch.nn.functional.mse_loss(reconstructed, batch)

            # Sparsity penalty (computed by sparsity mechanism)
            sparsity_loss = sparsity_penalty * self.sae.sparsity.compute_penalty(pre_activation)

            # Global scale regularization (for DeepSAE with spectral norm)
            # Penalize absurdly large global_scale to prevent the "tiny latents, huge scale" cheat.
            # This keeps latents in a healthy range (compatible with JumpReLU thresholds ~0.001-5.0).
            scale_penalty = torch.tensor(0.0, device=batch.device)
            if hasattr(self.sae, 'global_scale'):
                scale_penalty = 1e-2 * (self.sae.global_scale ** 2)

            # Total loss
            total_loss = recon_loss + sparsity_loss + scale_penalty

        # Backward pass (no gradient scaling needed for bfloat16)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # CRITICAL POST-PROCESSING STEPS
        with torch.no_grad():
            # A. Clamp JumpReLU Thresholds
            # Keeps features alive and ensures valid bounds
            if hasattr(self.sae.sparsity, 'clamp_thresholds'):
                self.sae.sparsity.clamp_thresholds(min_val=0.001, max_val=5.0)

            self.sae.anti_cheat()

        # Advance the LR schedule.
        if self._lr_schedule is not None and isinstance(self._lr_schedule, torch.optim.lr_scheduler.LRScheduler):
            self._lr_schedule.step()

        # Compute sparsity metrics
        num_active = (sparse_features > 0.01).float().sum(dim=1).mean().item()

        pct_active = (num_active / sparse_features.shape[1]) * 100

        return TrainMetrics(
            loss=total_loss.item(),
            recon_loss=recon_loss.item(),
            sparsity_loss=sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else 0.0,
            num_active=num_active,
            pct_active=pct_active,
        )

    def train_sae(
        self,
        num_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a training loop for num_epochs epochs.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training (ignored if using DataLoader mode)
            verbose: Whether to print progress

        Returns:
            Dictionary containing:
                - 'activation_mean': Mean of input activations (needed for inference)
                - 'loss_history': List of average losses per epoch
                - 'recon_loss_history': List of reconstruction losses
                - 'sparsity_history': List of sparsity percentages
                - 'final_metrics': TrainMetrics object with final epoch metrics

        Note:
            Sparsity is controlled by the SAE's sparsity mechanism (sae.sparsity).
            Configure it when creating the SAE (TopKSparsity, L1Sparsity, etc.).
        """
        # Determine effective batch size
        if self._use_dataloader:
            effective_batch_size = self._train_loader.batch_size
        else:
            effective_batch_size = batch_size

        if verbose:
            mode_str = "DataLoader" if self._use_dataloader else "tensor"
            print(f"\nTraining SAE on {self._num_samples:,} activation vectors ({mode_str} mode)")
            print(f"Input dim: {self.sae.input_dim}, Latent dim: {self.sae.probe_dim}")
            print(f"Batch size: {effective_batch_size}, Epochs: {num_epochs}")
            print("=" * 60)

        # Track metrics over time
        loss_history = []
        recon_loss_history = []
        sparsity_history = []

        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Initialize epoch metrics
            epoch_metrics = TrainMetrics()
            num_batches = 0

            # Sparsity penalty and learning rate multiplier
            sparsity_penalty = self._sparsity_schedule(epoch)
            if self._lr_schedule is not None and isinstance(self._lr_schedule, Schedule):
                current_lr = self._lr_schedule(epoch)
            else:
                current_lr = None

            # Batch iteration - different for DataLoader vs tensor mode
            if self._use_dataloader:
                # DataLoader mode: iterate over DataLoader
                for batch in self._train_loader:
                    batch = batch.to(self._device)

                    batch_metrics = self.train_step(
                        batch,
                        sparsity_penalty=sparsity_penalty,
                        current_lr=current_lr,
                    )

                    epoch_metrics.update(batch_metrics)
                    num_batches += 1
            else:
                # Legacy tensor mode: manual batching with shuffle
                perm = torch.randperm(self._num_samples)

                for i in range(0, self._num_samples, batch_size):
                    batch_indices = perm[i:i + batch_size]
                    batch = self._centered_activations[batch_indices]

                    batch_metrics = self.train_step(
                        batch,
                        sparsity_penalty=sparsity_penalty,
                        current_lr=current_lr,
                    )

                    epoch_metrics.update(batch_metrics)
                    num_batches += 1

            # Average metrics over batches
            epoch_metrics.scale(1.0 / num_batches)

            # Store history
            loss_history.append(epoch_metrics.loss)
            recon_loss_history.append(epoch_metrics.recon_loss)
            sparsity_history.append(epoch_metrics.pct_active)

            epoch_time = time.time() - epoch_start_time
            # Print progress
            if verbose:
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Loss: {epoch_metrics.loss:.4f} | "
                    f"Recon: {epoch_metrics.recon_loss:.4f} | "
                    f"Active: {epoch_metrics.num_active:.0f}/{self.sae.probe_dim} "
                    f"({epoch_metrics.pct_active:.1f}%)"
                    f" | Time: {epoch_time:.2f}s"
                )

        if verbose:
            print("=" * 60)
            print("Training complete!\n")

        return {
            'activation_mean': self.activation_mean,
            'loss_history': loss_history,
            'recon_loss_history': recon_loss_history,
            'sparsity_history': sparsity_history,
            'final_metrics': epoch_metrics,
        }
