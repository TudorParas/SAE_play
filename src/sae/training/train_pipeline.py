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
from src.sae.configs.training import AuxKConfig
from src.sae.training.dead_latent_tracker import DeadLatentTracker


class TrainPipeline:
    def __init__(
        self,
        sae: BaseSAE,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        lr_schedule: Optional[Union[Schedule, torch.optim.lr_scheduler.LRScheduler]] = None,
        sparsity_schedule: Optional[Schedule] = None,
        use_amp: bool = False,
        auxk_config: Optional[AuxKConfig] = None,
        max_grad_norm: Optional[float] = None,
    ):
        """
        Train a Sparse Autoencoder on collected activations.

        This is the main training loop that:
        1. Runs training epochs with batching
        2. Tracks metrics
        3. Returns results

        Args:
            sae: The SAE model to train
            optimizer: The optimizer to use for training
            train_loader: DataLoader from ActivationDataset (already centered).
            lr_schedule: Learning rate schedule. Three options:
                - None: Use the optimizer's initial learning rate (no scheduling).
                - Schedule: Custom epoch-based schedule (calls schedule(epoch)).
                - torch.optim.lr_scheduler.LRScheduler: PyTorch scheduler (calls scheduler.step()).
            sparsity_schedule: Schedule for sparsity penalty coefficient (epoch-based).
                If None, uses constant coefficient of 1.0.
            use_amp: Use automatic mixed precision training with bfloat16.
                Speeds up training on modern GPUs with minimal accuracy impact.
            auxk_config: Optional AuxK configuration for combating dead latents.
                If provided, enables auxiliary loss on dead latents.
            max_grad_norm: Maximum gradient norm for clipping. If None, no clipping.
                Recommended: 1.0 for stability, especially with JumpReLU.
        """
        self.sae = sae
        self.optimizer = optimizer
        self._device = next(sae.parameters()).device
        self._max_grad_norm = max_grad_norm

        self._train_loader = train_loader
        self._num_samples = len(train_loader.dataset)

        # Schedules
        self._lr_schedule = lr_schedule
        if sparsity_schedule is None:
            sparsity_schedule = ConstantSchedule(1.0)
        self._sparsity_schedule = sparsity_schedule

        # AMP setup (always use bfloat16 when enabled)
        self._amp_dtype = torch.bfloat16 if use_amp else None

        # AuxK setup for dead latent resurrection
        self._auxk_config = auxk_config
        if auxk_config is not None:
            # Validate sparsity type (AuxK only works with TopK)
            from src.sae.sparsity import TopKSparsity
            from src.sae.models.simple import SimpleSAE

            if not isinstance(sae.sparsity, TopKSparsity):
                raise ValueError(
                    f"AuxK only supports TopKSparsity, got {type(sae.sparsity).__name__}. "
                    f"Set auxk=None in training config or change sparsity type."
                )

            # Validate SAE architecture (SimpleSAE only for now)
            if not isinstance(sae, SimpleSAE):
                raise NotImplementedError(
                    "AuxK currently only supports SimpleSAE. "
                    "See TODO in train_pipeline.py for DeepSAE implementation."
                )

            # Instantiate dead latent tracker
            self._dead_tracker = DeadLatentTracker(
                num_latents=sae.probe_dim,
                dead_threshold_tokens=auxk_config.dead_threshold_tokens,
                device=self._device,
            )
            print(
                f"AuxK enabled: k={auxk_config.k}, α={auxk_config.coefficient}, "
                f"threshold={auxk_config.dead_threshold_tokens:,} tokens"
            )
        else:
            self._dead_tracker = None

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

            # AuxK auxiliary loss for dead latent resurrection
            auxk_loss = torch.tensor(0.0, device=batch.device)
            if self._dead_tracker is not None:
                # 1. Compute and detach reconstruction error
                # CRITICAL: Detach to prevent gradients flowing back to main model
                error = (batch - reconstructed).detach()

                # 2. Get dead latent mask
                dead_mask = self._dead_tracker.get_dead_mask()
                num_dead = dead_mask.sum().item()

                if num_dead > 0:
                    # 3. Extract pre-activations for dead latents only
                    # Shape: (batch_size, num_dead)
                    dead_pre_activation = pre_activation[:, dead_mask]

                    # 4. Select top k_aux dead latents (per sample)
                    k_current = min(self._auxk_config.k, num_dead)

                    # TopK selection per sample
                    # Shape: vals=(batch, k_current), inds=(batch, k_current)
                    topk_vals, topk_inds_local = torch.topk(
                        dead_pre_activation, k=k_current, dim=-1
                    )

                    # 5. Map local indices back to global latent indices
                    dead_global_indices = torch.where(dead_mask)[0]

                    topk_inds_global = dead_global_indices[topk_inds_local]

                    error_reconstruction = self.sae.decode_sparse(
                        indices=topk_inds_global, values=topk_vals, apply_bias=False
                    )

                    # 7. Compute auxiliary loss
                    aux_loss_raw = torch.nn.functional.mse_loss(error, error_reconstruction)

                    # 8. NaN guard (critical for stability)
                    if torch.isnan(aux_loss_raw) or torch.isinf(aux_loss_raw):
                        print("WARNING: AuxK loss is NaN/Inf, zeroing it out")
                        auxk_loss = torch.tensor(0.0, device=batch.device, requires_grad=True)
                    else:
                        auxk_loss = self._auxk_config.coefficient * aux_loss_raw

            # Total loss
            total_loss = recon_loss + sparsity_loss + scale_penalty + auxk_loss

        # Backward pass (no gradient scaling needed for bfloat16)
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        if self._max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self._max_grad_norm)

        self.optimizer.step()

        # CRITICAL POST-PROCESSING STEPS
        with torch.no_grad():
            # A. Clamp JumpReLU Thresholds
            # Keeps features alive and ensures valid bounds
            if hasattr(self.sae.sparsity, 'clamp_thresholds'):
                self.sae.sparsity.clamp_thresholds(min_val=0.001, max_val=5.0)

            self.sae.anti_cheat()

            # B. Update dead latent tracker (for AuxK)
            # Track which latents are actually firing
            if self._auxk_config is not None and self._dead_tracker is not None:
                # IMPORTANT: Use sparse_features (post-TopK) not pre_activation
                # For TopK sparsity, we care about which features actually fire
                # (survive TopK), not which ones had high pre-activation values
                self._dead_tracker.update(sparse_features.detach())

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
            batch_size: Unused (batch size comes from DataLoader)
            verbose: Whether to print progress

        Returns:
            Dictionary containing:
                - 'loss_history': List of average losses per epoch
                - 'recon_loss_history': List of reconstruction losses
                - 'sparsity_history': List of sparsity percentages
                - 'final_metrics': TrainMetrics object with final epoch metrics

        Note:
            Sparsity is controlled by the SAE's sparsity mechanism (sae.sparsity).
            Configure it when creating the SAE (TopKSparsity, L1Sparsity, etc.).
        """
        effective_batch_size = self._train_loader.batch_size

        if verbose:
            print(f"\nTraining SAE on {self._num_samples:,} activation vectors")
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

            for batch in self._train_loader:
                batch = batch.to(self._device)

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
            # ToDo: maybe add the number of dead parameters here as well? If we use
            if verbose:
                if self._dead_tracker is not None:
                    # 2. Get dead latent mask
                    dead_mask = self._dead_tracker.get_dead_mask()
                    num_dead = str(dead_mask.sum().item())
                else:
                    num_dead = "Unknown"
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Loss: {epoch_metrics.loss:.4f} | "
                    f"Recon: {epoch_metrics.recon_loss:.4f} | "
                    f"Active: {epoch_metrics.num_active:.0f}/{self.sae.probe_dim} "
                    f"({epoch_metrics.pct_active:.1f}%)"
                    f" | Dead: {num_dead}"
                    f" | Time: {epoch_time:.2f}s"
                )

        if verbose:
            print("=" * 60)
            print("Training complete!\n")

        return {
            'loss_history': loss_history,
            'recon_loss_history': recon_loss_history,
            'sparsity_history': sparsity_history,
            'final_metrics': epoch_metrics,
        }
