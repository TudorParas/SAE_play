"""
SAE Library - Sparse Autoencoder experimentation toolkit.

A library (not a framework) for training and analyzing Sparse Autoencoders
on language model activations.

Philosophy: More control, less magic, easier to understand and modify.

Example usage:
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from sae.data.loader import load_pile_samples
    >>> from sae.activations import extract_activations
    >>> from sae.models.simple import SimpleSAE
    >>> from sae.training.trainer import SAETrainer
    >>> from sae.training.loop import train_sae
    >>>
    >>> # Load your data (automatically finds data file)
    >>> texts = load_pile_samples(num_samples=1000)
    >>>
    >>> # Load your model
    >>> model = AutoModelForCausalLM.from_pretrained("pythia-70m")
    >>> tokenizer = AutoTokenizer.from_pretrained("pythia-70m")
    >>>
    >>> # Extract activations
    >>> activations = extract_activations(model, tokenizer, texts, layer_idx=3)
    >>>
    >>> # Create SAE
    >>> sae = SimpleSAE(input_dim=512, hidden_dim=2048)
    >>>
    >>> # Train
    >>> trainer = SAETrainer(sae, lr=1e-3)
    >>> results = train_sae(sae, trainer, activations, num_epochs=100)
"""

# Version
__version__ = "0.1.0"

# Convenient imports
from .data.loader import load_pile_samples
from .models.simple import SimpleSAE
from .models.deep import DeepSAE
from .sparsity import SparsityMechanism, TopKSparsity, L1Sparsity, NoSparsity, BatchTopKSparsity, JumpReLUSparsity
from .activations import extract_activations
from .training.train_pipeline import TrainPipeline
from .checkpoints import save_checkpoint, load_checkpoint
from .evaluation.report import ExperimentReport
from .evaluation.metrics import compute_dead_features
from .util.logging import TeeLogger

__all__ = [
    "load_pile_samples",
    "SimpleSAE",
    "DeepSAE",
    "SparsityMechanism",
    "TopKSparsity",
    "L1Sparsity",
    "NoSparsity",
    "BatchTopKSparsity",
    "JumpReLUSparsity",
    "extract_activations",
    "TrainPipeline",
    "save_checkpoint",
    "load_checkpoint",
    "ExperimentReport",
    "compute_dead_features",
    "TeeLogger",
]
