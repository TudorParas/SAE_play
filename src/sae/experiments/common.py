"""
Common utilities for SAE experiments.

These are library functions, not a framework - experiments call them explicitly
and remain in control of the flow. Use what you need, skip what you don't.

Philosophy:
- Explicit over implicit: You see what you're using
- Composition over configuration: Pass objects, not configs
- Easy to customize: Just don't call the helper, write your own code
"""

import torch
from typing import Tuple, List, Optional
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.sae.data.loader import load_pile_samples
from src.sae.data.datasets import split_activations, ActivationDataset, create_dataloader
from src.sae.activations import extract_activations
from src.sae.models.base import BaseSAE
from src.sae.evaluation.evaluator import Evaluator, EvalConfig, EvalResults, AnalysisResults


def load_model(
    model_name: str = "EleutherAI/pythia-70m",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, str]:
    """
    Load a HuggingFace model and tokenizer, move to best available device.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Tuple of (model, tokenizer, device_string)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT models need this

    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def prepare_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    layer_idx: int,
    train_frac: float = 0.9,
    batch_size: int = 8,
    seed: int = 42,
) -> Tuple[ActivationDataset, ActivationDataset, torch.Tensor]:
    """
    Extract activations, split into train/test, and center with train mean.

    This handles the full pipeline:
    1. Extract activations from the specified layer
    2. Split into train/test (before centering to avoid data leakage)
    3. Compute mean on train data only
    4. Create centered datasets

    Args:
        model: The language model to extract activations from
        tokenizer: Tokenizer for the model
        texts: List of text strings to process
        layer_idx: Which layer to extract activations from
        train_frac: Fraction of data for training (default 0.9)
        batch_size: Batch size for activation extraction
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_dataset, test_dataset, activation_mean)
    """
    # Extract activations
    activations = extract_activations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_idx=layer_idx,
        batch_size=batch_size
    )
    # Move to CPU - DataLoader will handle GPU transfer (pin_memory requires CPU tensors)
    activations = activations.cpu()

    # Split FIRST, then center (to avoid data leakage)
    train_raw, test_raw = split_activations(activations, train_frac=train_frac, seed=seed)

    # Compute mean on TRAIN only
    train_mean = train_raw.mean(dim=0, keepdim=True)

    # Create datasets (both centered with train mean)
    train_dataset = ActivationDataset(train_raw, mean=train_mean)
    test_dataset = ActivationDataset(test_raw, mean=train_mean)

    return train_dataset, test_dataset, train_mean


def run_evaluation(
    sae: BaseSAE,
    test_dataset: ActivationDataset,
    activation_mean: torch.Tensor,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_idx: int,
    sample_texts: Optional[List[str]] = None,
    eval_batch_size: int = 256,
) -> Tuple[Evaluator, EvalResults, Optional[AnalysisResults]]:
    """
    Run full evaluation: compute metrics on test set and optionally analyze sample texts.

    Args:
        sae: The trained SAE model
        test_dataset: Test dataset (centered with train mean)
        activation_mean: Mean used for centering (from training data)
        model: The language model (needed for text analysis)
        tokenizer: Tokenizer (needed for text analysis)
        layer_idx: Layer index (needed for text analysis)
        sample_texts: Optional list of texts to analyze features on
        eval_batch_size: Batch size for evaluation

    Returns:
        Tuple of (evaluator, eval_results, analysis_results)
        analysis_results is None if sample_texts is None
    """
    # Create test DataLoader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Create evaluator
    evaluator = Evaluator(
        sae=sae,
        activation_mean=activation_mean,
        config=EvalConfig(dead_feature_threshold=0.01),
    )

    # Run evaluation on test set
    eval_results = evaluator.evaluate(test_loader)

    # Analyze sample texts if provided
    analysis_results = None
    if sample_texts is not None:
        analysis_results = evaluator.analyze_texts(model, tokenizer, sample_texts, layer_idx)

    return evaluator, eval_results, analysis_results
