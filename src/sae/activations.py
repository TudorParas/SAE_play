"""
Extract activations from transformer language models.

This module provides utilities to extract per-token activations from specific layers
of transformer models for use in training SAEs.
"""

import torch
from typing import List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


def extract_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    layer_idx: Optional[int] = None,
    batch_size: int = 8,
    max_length: int = 128,
) -> torch.Tensor:
    """
    Extract per-token activations from a specific transformer layer.

    This processes texts in an autoregressive manner - each token position produces
    one activation vector based on the context of all previous tokens.

    IMPORTANT: Padding tokens are automatically filtered out to ensure clean activations.

    Args:
        model: HuggingFace transformer model (must support output_hidden_states=True)
        tokenizer: Tokenizer corresponding to the model
        texts: List of text strings to process
        layer_idx: Which transformer layer to extract from (0-indexed).
                   If None, uses the middle layer.
        batch_size: How many texts to process at once (for GPU efficiency)
        max_length: Maximum sequence length (longer sequences are truncated)

    Returns:
        Tensor of activations with shape (total_tokens, hidden_dim)
        where total_tokens is the sum of all real (non-padding) tokens across all texts
    """
    # Auto-detect number of layers and choose middle layer if not specified
    num_layers = model.config.num_hidden_layers
    if layer_idx is None:
        layer_idx = num_layers // 2  # Use middle layer

    # Validate layer index
    if not 0 <= layer_idx < num_layers:
        raise ValueError(
            f"layer_idx {layer_idx} is out of range. "
            f"Model has {num_layers} layers (0-{num_layers-1})"
        )

    device = next(model.parameters()).device
    all_activations = []

    # Process in batches for GPU efficiency
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize with padding
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Move to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run forward pass with hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract activations from target layer
        # outputs.hidden_states[0] is embeddings, [1] is layer 0, etc.
        # Shape: (batch_size, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states[layer_idx + 1]

        # Filter out padding tokens using attention mask
        attention_mask = inputs['attention_mask']

        # Collect activations for real tokens only
        for b in range(hidden_states.shape[0]):
            # Count real tokens (where mask = 1)
            num_real_tokens = attention_mask[b].sum().item()

            # Extract only real token activations
            text_activations = hidden_states[b, :num_real_tokens, :]

            all_activations.append(text_activations)

    # Concatenate all token activations
    # Final shape: (total_num_tokens, hidden_dim)
    all_activations = torch.cat(all_activations, dim=0)

    return all_activations
