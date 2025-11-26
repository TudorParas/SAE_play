from dataclasses import replace
from src.sae.configs import SIMPLE_SAE
from src.sae.experiments.runner import run_sae_experiment, get_experiment_name, get_project_name


def simple_sae_fast():
    """Fast experiment experiment."""
    # Modify for this experiment - using dataclasses.replace
    # Changes are visible and explicit
    config = replace(
        SIMPLE_SAE,
        experiment_name=get_experiment_name(), project_name=get_project_name(__file__),
        data=replace(SIMPLE_SAE.data, num_samples=100),
        training=replace(SIMPLE_SAE.training, num_epochs=3),
    )

    # Run the experiment - that's it!
    results = run_sae_experiment(config)

    # Results contain training history, evaluation, and output path
    print(f"\nExperiment outputs saved to: {results['output_dir']}")

def simple_sae_fast_perf():
    """Add performance such as compile, autocast, and num_workers to the dataloader."""
    # Modify for this experiment - using dataclasses.replace
    # Changes are visible and explicit
    config = replace(
        SIMPLE_SAE,
        experiment_name=get_experiment_name(), project_name=get_project_name(__file__),
        data=replace(SIMPLE_SAE.data, num_samples=100),
        training=replace(SIMPLE_SAE.training, num_epochs=3),
    )

    # Run the experiment - that's it!
    results = run_sae_experiment(config)

    # Results contain training history, evaluation, and output path
    print(f"\nExperiment outputs saved to: {results['output_dir']}")


def simple_sae_base():
    """Fast experiment experiment."""
    # Modify for this experiment - using dataclasses.replace
    # Changes are visible and explicit
    config = replace(
        SIMPLE_SAE,
        experiment_name=get_experiment_name(), project_name=get_project_name(__file__),
        training=replace(SIMPLE_SAE.training),
        data=replace(SIMPLE_SAE.data, num_workers=0),
    )

    # Run the experiment - that's it!
    results = run_sae_experiment(config)

    # Results contain training history, evaluation, and output path
    print(f"\nExperiment outputs saved to: {results['output_dir']}")


if __name__ == "__main__":
    simple_sae_base()