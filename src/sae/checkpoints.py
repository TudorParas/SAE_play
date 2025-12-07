"""
Checkpoint management for saving and loading trained SAEs.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import asdict
from src.sae.models.base import BaseSAE
from src.sae.models.simple import SimpleSAE
from src.sae.models.deep import DeepSAE


def save_checkpoint(
    path: str,
    sae: BaseSAE,
    activation_mean: torch.Tensor,
    experiment_config: Any = None,
    final_loss: Optional[float] = None,
):
    """
    Save a trained SAE to disk.

    Args:
        path: Directory path to save checkpoint (will be created if doesn't exist)
        sae: Trained SAE model
        activation_mean: Mean of training activations (needed for inference)
        experiment_config: Full experiment configuration (SAEExperimentConfig)
        final_loss: Final training loss
    """
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save SAE weights
    weights_path = save_path / "sae_weights.pt"
    torch.save(sae.state_dict(), weights_path)

    # Save activation mean
    mean_path = save_path / "activation_mean.pt"
    torch.save(activation_mean, mean_path)

    # Save full experiment config
    if experiment_config is not None:
        config_dict = asdict(experiment_config)
        # Add runtime info
        config_dict['final_loss'] = final_loss
        config_dict['input_dim'] = sae.input_dim
        config_dict['probe_dim'] = sae.probe_dim
    else:
        # Fallback if no experiment config provided
        config_dict = sae.get_config()

    config_path = save_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

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

    # Load from nested config format
    input_dim = config['input_dim']
    sae_config = config['sae']

    # Determine SAE type from nested config
    if 'hidden_dim_multiplier' in sae_config:
        # SimpleSAE
        hidden_dim = input_dim * sae_config['hidden_dim_multiplier']
        from src.sae.sparsity import TopKSparsity, L1Sparsity, JumpReLUSparsity

        # Create sparsity mechanism
        sparsity_type = sae_config.get('sparsity_type', 'topk')
        if sparsity_type == 'topk':
            sparsity = TopKSparsity(k=sae_config.get('sparsity_k', 128))
        elif sparsity_type == 'l1':
            sparsity = L1Sparsity()
        else:
            sparsity = JumpReLUSparsity(num_features=hidden_dim)

        sae = SimpleSAE(input_dim=input_dim, hidden_dim=hidden_dim, sparsity=sparsity)
        sae_type = "SimpleSAE"
    else:
        # DeepSAE
        encoder_hidden_dims = [input_dim * m for m in sae_config['encoder_hidden_dims']]
        decoder_hidden_dims = [input_dim * m for m in sae_config['decoder_hidden_dims']]

        from src.sae.sparsity import TopKSparsity, L1Sparsity, JumpReLUSparsity
        sparsity_type = sae_config.get('sparsity_type', 'l1')
        if sparsity_type == 'topk':
            sparsity = TopKSparsity(k=sae_config.get('sparsity_k', 64))
        elif sparsity_type == 'l1':
            sparsity = L1Sparsity()
        else:
            sparsity = JumpReLUSparsity(num_features=encoder_hidden_dims[-1])

        sae = DeepSAE(
            input_dim=input_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            sparsity=sparsity
        )
        sae_type = "DeepSAE"

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
