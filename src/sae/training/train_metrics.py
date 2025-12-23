"""
Training metrics dataclass.

Type-safe container for training metrics instead of passing dicts with string keys.
"""

from dataclasses import dataclass, fields


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
        """Add weighted values from another TrainMetrics object."""
        for f in fields(self):
            current = getattr(self, f.name)
            other_val = getattr(other, f.name)
            setattr(self, f.name, current + other_val * weight)

    def scale(self, factor: float):
        """Scale all metrics by a factor (e.g., to compute averages)."""
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) * factor)

    def __repr__(self) -> str:
        """String representation for logging."""
        return (
            f"TrainMetrics(loss={self.loss:.4f}, "
            f"recon={self.recon_loss:.4f}, "
            f"active={self.num_active:.0f}, "
            f"pct={self.pct_active:.1f}%)"
        )
