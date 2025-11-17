"""
Training metrics dataclass.

Type-safe container for training metrics instead of passing dicts with string keys.
"""

from dataclasses import dataclass, field


@dataclass
class TrainMetrics:
    """
    Metrics from SAE training.

    Can be used for both batch-level and epoch-level metrics.
    """
    loss: float = 0.0
    recon_loss: float = 0.0
    sparsity_loss: float = 0.0
    num_active: float = 0.0
    pct_active: float = 0.0

    def update(self, other: 'TrainMetrics', weight: float = 1.0):
        """
        Update this metrics object by adding weighted values from another.

        Useful for accumulating batch metrics into epoch metrics.

        Args:
            other: Another TrainMetrics object
            weight: Weight to apply to the other metrics (default: 1.0)
        """
        self.loss += other.loss * weight
        self.recon_loss += other.recon_loss * weight
        self.sparsity_loss += other.sparsity_loss * weight
        self.num_active += other.num_active * weight
        self.pct_active += other.pct_active * weight

    def scale(self, factor: float):
        """
        Scale all metrics by a factor (e.g., to compute averages).

        Args:
            factor: Scaling factor
        """
        self.loss *= factor
        self.recon_loss *= factor
        self.sparsity_loss *= factor
        self.num_active *= factor
        self.pct_active *= factor

    def __repr__(self) -> str:
        """String representation for logging."""
        return (
            f"TrainMetrics(loss={self.loss:.4f}, "
            f"recon={self.recon_loss:.4f}, "
            f"active={self.num_active:.0f}, "
            f"pct={self.pct_active:.1f}%)"
        )
