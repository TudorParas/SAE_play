"""
Unified evaluation for trained SAEs.

The Evaluator class computes all evaluation metrics and generates reports.
It handles device management properly by using DataLoaders.

Usage:
    evaluator = Evaluator(
        sae=sae,
        activation_mean=activation_mean,
        config=eval_config,
    )

    # Run evaluation on test set
    eval_results = evaluator.evaluate(test_loader)

    # Analyze specific texts (separate from evaluation)
    analysis = evaluator.analyze_texts(model, tokenizer, texts, layer_idx)

    # Generate full report
    evaluator.generate_report(
        experiment_id="my_exp",
        training_results=train_results,
        eval_results=eval_results,
        save_path="reports/my_exp",
    )
"""

import torch
from torch.utils.data import DataLoader
from typing import Any
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.sae.models.base import BaseSAE
from src.sae.activations import extract_activations
from src.sae.configs.evaluation import EvalConfig
from src.sae.evaluation.report import ExperimentReport
from src.sae.evaluation.metrics import compute_reconstruction_loss, compute_sparsity, get_spectral_stats


@dataclass
class EvalResults:
    """Container for evaluation results (metrics on test set)."""
    # Core metrics
    reconstruction_loss: float = 0.0
    sparsity_metrics: dict[str, float] = field(default_factory=dict)
    dead_features: dict[str, Any] = field(default_factory=dict)
    spectral_stats: dict[str, float] = field(default_factory=dict)

    # Metadata
    num_eval_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "reconstruction_loss": self.reconstruction_loss,
            "sparsity_metrics": self.sparsity_metrics,
            "dead_features": self.dead_features,
            "spectral_stats": self.spectral_stats,
            "num_eval_samples": self.num_eval_samples,
        }


@dataclass
class AnalysisResults:
    """Container for text analysis results."""
    texts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {"texts": self.texts}


class Evaluator:
    """
    Unified evaluation for trained SAEs.

    Handles device management properly by using DataLoaders and moving
    tensors to the correct device before operations.

    Args:
        sae: Trained SAE model
        activation_mean: Mean tensor used for centering (needed for text analysis)
        config: Evaluation configuration
    """

    def __init__(
        self,
        sae: BaseSAE,
        activation_mean: torch.Tensor,
        config: EvalConfig,
    ):
        self.sae = sae
        self.activation_mean = activation_mean
        self.config = config

        # Get device from SAE
        self._device = next(sae.parameters()).device

        # Move activation mean to device
        self._activation_mean_device = activation_mean.to(self._device)

    def evaluate(self, test_loader: DataLoader) -> EvalResults:
        """
        Run evaluation on test set.

        Computes reconstruction loss, sparsity metrics, dead features,
        and spectral statistics.

        Args:
            test_loader: DataLoader yielding centered activations

        Returns:
            EvalResults containing all computed metrics
        """
        results = EvalResults()
        results.num_eval_samples = len(test_loader.dataset)

        # Put SAE in eval mode
        self.sae.eval()

        # Compute core metrics (iterate through test loader once)
        recon_loss, sparsity_metrics, dead_features = self._compute_core_metrics(test_loader)
        results.reconstruction_loss = recon_loss
        results.sparsity_metrics = sparsity_metrics
        results.dead_features = dead_features

        # Compute spectral stats
        results.spectral_stats = self._compute_spectral_stats(test_loader)

        return results

    def analyze_texts(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        texts: list[str],
        layer_idx: int,
    ) -> AnalysisResults:
        """
        Analyze which features activate for specific texts.

        This is separate from evaluate() - it's for understanding
        what the SAE has learned on specific examples.

        Args:
            model: Language model
            tokenizer: Tokenizer
            texts: List of texts to analyze
            layer_idx: Layer to extract activations from

        Returns:
            AnalysisResults containing per-text feature analysis
        """
        results = AnalysisResults()
        threshold = self.config.dead_feature_threshold
        top_k = self.config.feature_analysis_top_k

        self.sae.eval()

        for text in texts:
            # Extract activations
            activations = extract_activations(
                model=model,
                tokenizer=tokenizer,
                texts=[text],
                layer_idx=layer_idx,
            )

            # Center and move to device
            centered = activations.to(self._device) - self._activation_mean_device

            # Pass through SAE
            with torch.no_grad():
                _, sparse_features, _ = self.sae(centered)

            # Average features across tokens
            avg_features = sparse_features.mean(dim=0)

            # Get top-k features
            top_values, top_indices = torch.topk(
                avg_features,
                k=min(top_k, len(avg_features))
            )

            top_features = [
                (idx.item(), val.item())
                for idx, val in zip(top_indices, top_values)
            ]

            # Compute active count
            num_active = (avg_features > threshold).sum().item()
            total_features = len(avg_features)

            results.texts.append({
                "text": text,
                "top_features": top_features,
                "num_active": num_active,
                "total_features": total_features,
                "pct_active": (num_active / total_features) * 100,
            })

        return results

    def _compute_core_metrics(self, test_loader: DataLoader) -> tuple:
        """
        Compute reconstruction loss, sparsity metrics, and dead features in one pass.

        Args:
            test_loader: DataLoader yielding centered activations

        Returns:
            Tuple of (recon_loss, sparsity_metrics, dead_features)
        """
        total_recon_loss = 0.0
        total_sparsity_metrics = {
            "num_active": 0.0,
            "l0_norm": 0.0,
            "l1_norm": 0.0,
        }
        num_batches = 0

        # Track which features have ever been active (for dead feature computation)
        num_features = self.sae.probe_dim
        ever_active = torch.zeros(num_features, dtype=torch.bool, device=self._device)
        threshold = self.config.dead_feature_threshold

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self._device)

                # Forward pass
                reconstructed, sparse_features, _ = self.sae(batch)

                # Reconstruction loss (using metrics.py)
                total_recon_loss += compute_reconstruction_loss(batch, reconstructed)

                # Sparsity metrics (using metrics.py)
                batch_sparsity = compute_sparsity(sparse_features, threshold=threshold)
                total_sparsity_metrics["num_active"] += batch_sparsity["num_active"]
                total_sparsity_metrics["l0_norm"] += batch_sparsity["l0_norm"]
                total_sparsity_metrics["l1_norm"] += batch_sparsity["l1_norm"]

                # Dead feature tracking
                is_active = sparse_features > threshold
                batch_active = is_active.any(dim=0)
                ever_active = ever_active | batch_active

                num_batches += 1

        # Average metrics
        avg_recon_loss = total_recon_loss / num_batches
        sparsity_metrics = {
            "num_active": total_sparsity_metrics["num_active"] / num_batches,
            "pct_active": (total_sparsity_metrics["num_active"] / num_batches / num_features) * 100,
            "l0_norm": total_sparsity_metrics["l0_norm"] / num_batches,
            "l1_norm": total_sparsity_metrics["l1_norm"] / num_batches,
        }

        # Dead features
        alive_count = ever_active.sum().item()
        dead_count = num_features - alive_count

        dead_features = {
            "count": dead_count,
            "fraction": dead_count / num_features,
            "threshold": threshold,
            "total": num_features,
            "alive_count": alive_count,
        }

        return avg_recon_loss, sparsity_metrics, dead_features

    def _compute_spectral_stats(self, test_loader: DataLoader) -> dict[str, float]:
        """
        Computes the Effective Latent Dimension (ELD) based on the spectral properties
        of the SAE feature activations.

        Args:
            test_loader: DataLoader yielding centered activations

        Returns:
            dict: Contains 'ELD' and 'Top_Component_Explained_Var'.
        """
        # Collect activations (up to max_spectral_samples)
        all_features = []
        samples_collected = 0
        max_samples = self.config.max_spectral_samples

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self._device)

                # Encode to get sparse features
                feature_acts = self.sae.encode(batch)
                all_features.append(feature_acts)

                samples_collected += batch.shape[0]
                if samples_collected >= max_samples:
                    break

        # Concatenate and truncate to max samples
        feature_acts = torch.cat(all_features, dim=0)[:max_samples]

        # Use metrics.py function for spectral analysis
        return get_spectral_stats(feature_acts)

    def generate_report(
        self,
        experiment_id: str,
        timestamp: str,
        description: str = "",
        training_results: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        log_text: str = "",
        checkpoint_path: str = "",
        save_path: str | None = None,
        eval_results: EvalResults | None = None,
        analysis_results: AnalysisResults | None = None,
    ) -> ExperimentReport:
        """
        Generate a complete experiment report.

        Combines pre-computed evaluation and analysis results into a report.

        Args:
            experiment_id: Unique experiment identifier
            timestamp: Timestamp string in format YYYYMMDD_HHMMSS
            description: Human-readable description
            training_results: Results from training (loss_history, final_metrics, etc.)
            metadata: Experiment metadata (device, runtime, etc.)
            log_text: Captured stdout log
            checkpoint_path: Path to saved checkpoint
            save_path: If provided, saves report to this directory
            eval_results: Pre-computed EvalResults from evaluate()
            analysis_results: Pre-computed AnalysisResults from analyze_texts()

        Returns:
            ExperimentReport instance
        """

        # Create report
        report = ExperimentReport(
            experiment_id=experiment_id,
            timestamp=timestamp,
            description=description,
        )

        # Set metadata
        if metadata:
            report.set_metadata(**metadata)

        # Set training results
        if training_results:
            final_metrics = training_results.get("final_metrics")
            if hasattr(final_metrics, "loss"):
                # TrainMetrics object
                report.set_training_results(
                    final_loss=final_metrics.loss,
                    final_recon_loss=final_metrics.recon_loss,
                    final_sparsity_loss=final_metrics.sparsity_loss,
                    final_num_active=final_metrics.num_active,
                    final_pct_active=final_metrics.pct_active,
                    loss_history=training_results.get("loss_history", []),
                    recon_loss_history=training_results.get("recon_loss_history", []),
                    sparsity_history=training_results.get("sparsity_history", []),
                )
            else:
                # Dict-based
                report.set_training_results(**training_results)

        # Set eval results
        if eval_results:
            report.set_eval_results(
                dead_features=eval_results.dead_features,
                spectral_stats=eval_results.spectral_stats,
                num_eval_tokens=eval_results.num_eval_samples,
                reconstruction_loss=eval_results.reconstruction_loss,
                sparsity_metrics=eval_results.sparsity_metrics,
                # Include analysis if provided
                feature_analysis=analysis_results.texts if analysis_results else [],
                num_eval_samples=len(analysis_results.texts) if analysis_results else 0,
            )

        # Set log and artifacts
        report.set_log(log_text)
        report.set_artifacts(checkpoint_path=checkpoint_path)

        # Save if path provided
        if save_path:
            report.save(save_path)

        return report