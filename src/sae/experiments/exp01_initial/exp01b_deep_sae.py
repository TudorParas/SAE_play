from dataclasses import replace

from sae.configs.data import SourceConfig
from src.sae.configs.baselines import DEEP_SAE
from src.sae.configs.training import AuxKConfig
from src.sae.experiments.runner import run_sae_experiment, get_experiment_name, get_project_name


def deep_sae_fast():
    """Fast experiment experiment."""
    # Modify for this experiment - using dataclasses.replace
    # Changes are visible and explicit
    config = replace(
        DEEP_SAE,
        experiment_name=get_experiment_name(), project_name=get_project_name(__file__),
        data=replace(DEEP_SAE.data, num_samples=1000),
        training=replace(DEEP_SAE.training, num_epochs=5),
    )

    # Run the experiment - that's it!
    results = run_sae_experiment(config)

    # Results contain training history, evaluation, and output path
    print(f"\nExperiment outputs saved to: {results['output_dir']}")


def deep_sae_base():
    """Fast experiment experiment."""
    # Modify for this experiment - using dataclasses.replace
    # Changes are visible and explicit
    # auxk = AuxKConfig(dead_threshold_tokens=100_000)
    config = replace(
        DEEP_SAE,
        experiment_name=get_experiment_name(), project_name=get_project_name(__file__),
        training=replace(DEEP_SAE.training),
    )

    # Run the experiment - that's it!
    results = run_sae_experiment(config)

    # Results contain training history, evaluation, and output path
    print(f"\nExperiment outputs saved to: {results['output_dir']}")


def deep_sae_fast_auxk():
    """Fast experiment experiment."""
    # Modify for this experiment - using dataclasses.replace
    # Changes are visible and explicit
    auxk = AuxKConfig(dead_threshold_tokens=100_000)

    config = replace(
        DEEP_SAE,
        experiment_name=get_experiment_name(), project_name=get_project_name(__file__),
        data=replace(DEEP_SAE.data, num_samples=1000, training_batch_size=128),
        training=replace(DEEP_SAE.training, auxk=auxk, num_epochs=5),
    )

    # Run the experiment - that's it!
    results = run_sae_experiment(config)

    # Results contain training history, evaluation, and output path
    print(f"\nExperiment outputs saved to: {results['output_dir']}")

def deep_sae_auxk():
    """SimpleSAE with AuxK auxiliary loss for combating dead latents."""
    # Enable AuxK with default parameters from paper:
    # - coefficient: 1/32 (α in paper)
    # - k: 512 (number of dead latents to use)
    # - dead_threshold_tokens: 10M (tokens without activation before flagged dead)
    auxk = AuxKConfig(dead_threshold_tokens=1_000_000)


    config = replace(
        DEEP_SAE,
        experiment_name=get_experiment_name(), project_name=get_project_name(__file__),
        training=replace(DEEP_SAE.training, auxk=auxk),
    )

    # Run the experiment - that's it!
    results = run_sae_experiment(config)

    # Results contain training history, evaluation, and output path
    print(f"\nExperiment outputs saved to: {results['output_dir']}")



if __name__ == "__main__":
    deep_sae_fast_auxk()