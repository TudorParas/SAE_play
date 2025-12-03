"""
Unit tests for DeadLatentTracker.

Tests the dead latent tracking mechanism used in AuxK auxiliary loss.
"""

import torch
import pytest

from src.sae.training.dead_latent_tracker import DeadLatentTracker


class TestDeadLatentTracker:
    """Test suite for DeadLatentTracker."""

    def test_initialization(self):
        """Test tracker initialization with default and custom parameters."""
        # Default initialization
        tracker = DeadLatentTracker(num_latents=100)
        assert tracker.num_latents == 100
        assert tracker.dead_threshold_tokens == 10_000_000
        assert tracker.activation_threshold == 0.0
        assert tracker.total_tokens == 0
        assert tracker.tokens_since_activation.shape == (100,)
        assert torch.all(tracker.tokens_since_activation == 0)

        # Custom initialization
        tracker = DeadLatentTracker(
            num_latents=50,
            dead_threshold_tokens=1000,
            activation_threshold=0.5,
            device="cpu",
        )
        assert tracker.num_latents == 50
        assert tracker.dead_threshold_tokens == 1000
        assert tracker.activation_threshold == 0.5

    def test_update_with_activations(self):
        """Test that update correctly tracks activated latents."""
        tracker = DeadLatentTracker(num_latents=10, dead_threshold_tokens=100)

        # Create batch where only latents 0, 1, 2 activate
        batch_size = 5
        pre_activation = torch.zeros(batch_size, 10)
        pre_activation[:, 0] = 1.0  # Latent 0 activates
        pre_activation[:, 1] = 0.5  # Latent 1 activates
        pre_activation[0, 2] = 0.1  # Latent 2 activates (only first sample)

        tracker.update(pre_activation)

        # Check token counts
        assert tracker.total_tokens == batch_size
        assert tracker.tokens_since_activation[0].item() == 0  # Activated
        assert tracker.tokens_since_activation[1].item() == 0  # Activated
        assert tracker.tokens_since_activation[2].item() == 0  # Activated
        assert tracker.tokens_since_activation[3].item() == batch_size  # Not activated

    def test_update_accumulation(self):
        """Test that tokens_since_activation accumulates correctly over multiple updates."""
        tracker = DeadLatentTracker(num_latents=5, dead_threshold_tokens=100)

        # First batch: only latent 0 activates
        batch1 = torch.zeros(10, 5)
        batch1[:, 0] = 1.0
        tracker.update(batch1)

        assert tracker.tokens_since_activation[0].item() == 0
        assert tracker.tokens_since_activation[1].item() == 10
        assert tracker.total_tokens == 10

        # Second batch: only latent 1 activates
        batch2 = torch.zeros(5, 5)
        batch2[:, 1] = 1.0
        tracker.update(batch2)

        assert tracker.tokens_since_activation[0].item() == 5  # Didn't activate, incremented
        assert tracker.tokens_since_activation[1].item() == 0  # Activated, reset to 0
        assert tracker.tokens_since_activation[2].item() == 15  # Never activated
        assert tracker.total_tokens == 15

    def test_get_dead_mask(self):
        """Test that get_dead_mask correctly identifies dead latents."""
        tracker = DeadLatentTracker(num_latents=5, dead_threshold_tokens=100)

        # Manually set tokens_since_activation
        tracker.tokens_since_activation = torch.tensor([50, 100, 150, 0, 99])

        dead_mask = tracker.get_dead_mask()

        # Expected: [False, True, True, False, False]
        # (>= threshold is dead)
        expected = torch.tensor([False, True, True, False, False])
        assert torch.all(dead_mask == expected)

    def test_get_dead_latents(self):
        """Test that get_dead_latents returns correct indices."""
        tracker = DeadLatentTracker(num_latents=5, dead_threshold_tokens=100)

        # Manually set tokens_since_activation
        tracker.tokens_since_activation = torch.tensor([50, 100, 150, 0, 99])

        dead_latents = tracker.get_dead_latents()

        # Expected: indices [1, 2]
        expected = torch.tensor([1, 2])
        assert torch.all(dead_latents == expected)

    def test_get_top_k_dead_latents(self):
        """Test that get_top_k_dead_latents returns k most dead latents."""
        tracker = DeadLatentTracker(num_latents=6, dead_threshold_tokens=100)

        # Manually set tokens_since_activation
        # Dead latents: 1 (tokens=100), 2 (tokens=150), 4 (tokens=200)
        # Alive: 0, 3, 5
        tracker.tokens_since_activation = torch.tensor([50, 100, 150, 0, 200, 99])

        # Get top 2 dead latents
        top_k = tracker.get_top_k_dead_latents(k=2)

        # Expected: [4, 2] (most dead first)
        # Latent 4 has 200 tokens, latent 2 has 150 tokens
        assert len(top_k) == 2
        assert top_k[0].item() == 4  # Most dead
        assert top_k[1].item() == 2  # Second most dead

    def test_get_top_k_dead_latents_fewer_than_k(self):
        """Test get_top_k_dead_latents when fewer dead latents than k exist."""
        tracker = DeadLatentTracker(num_latents=5, dead_threshold_tokens=100)

        # Only 2 dead latents
        tracker.tokens_since_activation = torch.tensor([50, 100, 150, 0, 99])

        # Request 5, should return only 2
        top_k = tracker.get_top_k_dead_latents(k=5)

        assert len(top_k) == 2
        assert set(top_k.tolist()) == {1, 2}

    def test_get_top_k_dead_latents_none(self):
        """Test get_top_k_dead_latents returns None when no dead latents."""
        tracker = DeadLatentTracker(num_latents=3, dead_threshold_tokens=100)

        # All alive
        tracker.tokens_since_activation = torch.tensor([0, 50, 99])

        top_k = tracker.get_top_k_dead_latents(k=2)

        assert top_k is None

    def test_get_stats(self):
        """Test that get_stats returns correct statistics."""
        tracker = DeadLatentTracker(num_latents=5, dead_threshold_tokens=100)

        # 2 dead latents (indices 1, 2)
        tracker.tokens_since_activation = torch.tensor([50, 100, 150, 0, 99])
        tracker.total_tokens = 150

        stats = tracker.get_stats()

        assert stats["num_dead"] == 2
        assert stats["frac_dead"] == 2 / 5
        assert stats["total_tokens"] == 150

    def test_activation_threshold(self):
        """Test that activation_threshold is respected."""
        tracker = DeadLatentTracker(
            num_latents=3,
            dead_threshold_tokens=100,
            activation_threshold=0.5,  # Custom threshold
        )

        # Batch with values below and above threshold
        batch = torch.zeros(1, 3)
        batch[0, 0] = 0.4  # Below threshold, should not activate
        batch[0, 1] = 0.5  # Equal to threshold, should not activate (> not >=)
        batch[0, 2] = 0.6  # Above threshold, should activate

        tracker.update(batch)

        # Only latent 2 should have reset counter
        assert tracker.tokens_since_activation[0].item() == 1
        assert tracker.tokens_since_activation[1].item() == 1
        assert tracker.tokens_since_activation[2].item() == 0

    def test_device_placement(self):
        """Test that tracker tensors are on correct device."""
        # CPU test
        tracker_cpu = DeadLatentTracker(num_latents=10, device="cpu")
        assert tracker_cpu.tokens_since_activation.device.type == "cpu"

        # GPU test (if available)
        if torch.cuda.is_available():
            tracker_gpu = DeadLatentTracker(num_latents=10, device="cuda")
            assert tracker_gpu.tokens_since_activation.device.type == "cuda"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
