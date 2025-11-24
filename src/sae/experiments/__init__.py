"""
SAE experiment utilities and example experiments.

This module provides two workflows:

1. Config-based experiments (90% usecase):
   - run_sae_experiment(config) - One function to run complete experiments
   - See example_config_experiment.py for usage

2. Library primitives (10% custom cases):
   - load_model, prepare_activations, run_evaluation - Compose manually
   - See simple_sae_experiment.py for usage

Both approaches use the same underlying library - choose based on your needs.
"""

from .common import (
    load_model,
    prepare_activations,
    run_evaluation,
)
from .runner import (
    run_sae_experiment,
    get_project_name,
    get_experiment_name,
)

__all__ = [
    # Config-based workflow
    "run_sae_experiment",
    "get_project_name",
    "get_experiment_name",
    # Library primitives
    "load_model",
    "prepare_activations",
    "run_evaluation",
]
