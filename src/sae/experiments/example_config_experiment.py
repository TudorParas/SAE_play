"""
Example: Config-based SAE experiment.

This demonstrates the config-based experiment workflow:
1. Start with a baseline config
2. Use dataclasses.replace() to modify specific parameters
3. Use helper functions for easy copy-paste
4. Run the experiment with a single function call

For custom experiments that don't fit this pattern, you can still import
and use the library primitives directly (see simple_sae_experiment.py).

Run from project root:
    python -m src.sae.experiments.example_config_experiment
"""

from dataclasses import replace
from src.sae.configs import SIMPLE_SAE
from src.sae.experiments.runner import (
    run_sae_experiment,
    get_project_name,
    get_experiment_name,
)


def simple_sae_more_epochs():
    """Example experiment with modified config."""

    # Use helpers for easy copy-paste
    # Just change the function name and these auto-update
    PROJECT_NAME = get_project_name(__file__)
    EXPERIMENT_NAME = get_experiment_name(simple_sae_more_epochs)

    # Modify baseline config
    config = replace(
        SIMPLE_SAE,
        project_name=PROJECT_NAME,
        experiment_name=EXPERIMENT_NAME,
        # Modify nested configs
        training=replace(SIMPLE_SAE.training, num_epochs=50),
        model=replace(SIMPLE_SAE.model, layer_idx=5),
    )

    # Run the experiment
    results = run_sae_experiment(config)

    # Results contain training history, evaluation, and output path
    print(f"\nExperiment outputs saved to: {results['output_dir']}")


if __name__ == "__main__":
    simple_sae_more_epochs()
