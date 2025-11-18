"""
Checkpoint management for saving and loading trained SAEs.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from .models.base import BaseSAE
from .models.simple import SimpleSAE
from .models.deep import DeepSAE


def save_checkpoint(
    path: str,
    sae: BaseSAE,
    activation_mean: torch.Tensor,
    metadata: Dict[str, Any] = None
):
    """
    Save a trained SAE to disk.

    Args:
        path: Directory path to save checkpoint (will be created if doesn't exist)
        sae: Trained SAE model
        activation_mean: Mean of training activations (needed for inference)
        metadata: Optional metadata to save (e.g., training info, model name, layer)
    """
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save SAE weights
    weights_path = save_path / "sae_weights.pt"
    torch.save(sae.state_dict(), weights_path)

    # Save activation mean
    mean_path = save_path / "activation_mean.pt"
    torch.save(activation_mean, mean_path)

    # Save configuration and metadata
    config = sae.get_config()
    if metadata is not None:
        config.update(metadata)

    config_path = save_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Checkpoint saved to: {save_path.absolute()}")
    print(f"  - Weights: {weights_path.name}")
    print(f"  - Activation mean: {mean_path.name}")
    print(f"  - Config: {config_path.name}")


def load_checkpoint(
    path: str,
    device: str = "cpu"
) -> Tuple[BaseSAE, torch.Tensor, Dict[str, Any]]:
    """
    Load a trained SAE from disk.

    Args:
        path: Directory path where checkpoint was saved
        device: Device to load the SAE onto ("cpu" or "cuda")

    Returns:
        Tuple of (sae, activation_mean, metadata)

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist
        ValueError: If SAE type is not recognized
    """
    save_path = Path(path)

    if not save_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load config
    config_path = save_path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Instantiate the correct SAE type
    sae_type = config.get('type', 'SimpleSAE')

    if sae_type == 'SimpleSAE':
        sae = SimpleSAE(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            top_k=config.get('top_k', 0)
        )
    elif sae_type == 'DeepSAE':
        sae = DeepSAE(
            input_dim=config['input_dim'],
            encoder_hidden_dims=config['encoder_hidden_dims'],
            decoder_hidden_dims=config.get('decoder_hidden_dims'),
            top_k=config.get('top_k', 0)
        )
    else:
        raise ValueError(f"Unknown SAE type: {sae_type}")

    # Load weights
    weights_path = save_path / "sae_weights.pt"
    sae.load_state_dict(torch.load(weights_path, map_location=device))
    sae.to(device)
    sae.eval()

    # Load activation mean
    mean_path = save_path / "activation_mean.pt"
    activation_mean = torch.load(mean_path, map_location=device)

    print(f"✓ Checkpoint loaded from: {save_path.absolute()}")
    print(f"  - Type: {sae_type}")
    print(f"  - Architecture: {sae.input_dim} → {sae.probe_dim}")

    return sae, activation_mean, config
