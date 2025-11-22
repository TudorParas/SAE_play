"""
Experiment report generation and management.

Creates comprehensive reports for each SAE training experiment.
Reports are saved as:
- report.json: Machine-readable structured data
- report.md: Human-readable summary
- log.txt: Captured stdout from experiment run

Philosophy: Simple dict-based structure, human-readable first.

TODO: Add programmatic config capture via get_config() methods
      - Add get_config() to SparsityMechanism classes
      - Add get_config() to TrainPipeline
      - Add get_config() to Schedule classes
      - Store structured configs alongside log.txt
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class ExperimentReport:
    """
    Build and save experiment reports.

    Usage:
        report = ExperimentReport(
            experiment_id="simple_topk32_20241119",
            description="SimpleSAE with TopK-32 on layer 3"
        )

        # Add data as you go
        report.set_metadata(runtime_seconds=245.3, device="cuda")
        report.set_training_results(
            final_loss=0.0498,
            loss_history=[0.094, 0.056, ...],
            ...
        )
        report.set_eval_results(feature_analysis=[...], dead_features={...})
        report.set_log(captured_stdout)

        # Save everything
        report.save("reports/exp_simple_topk32")

    Creates:
        reports/exp_simple_topk32/
        ├── report.json     # Structured data (machine readable)
        ├── report.md       # Human readable summary
        └── log.txt         # Captured stdout
    """

    def __init__(
        self,
        experiment_id: str,
        description: str = "",
    ):
        """
        Initialize a new experiment report.

        Args:
            experiment_id: Unique identifier for this experiment
            description: Human-readable description of what this experiment tests
        """
        self.experiment_id = experiment_id
        self.description = description
        self.timestamp = datetime.now().isoformat()

        # Initialize data sections
        self._metadata: Dict[str, Any] = {}
        self._training_results: Dict[str, Any] = {}
        self._eval_results: Dict[str, Any] = {}
        self._log: str = ""
        self._artifacts: Dict[str, str] = {}

    def set_metadata(
        self,
        runtime_seconds: float = 0.0,
        device: str = "",
        git_commit: str = "",
        **kwargs
    ):
        """
        Set experiment metadata.

        Args:
            runtime_seconds: Total experiment runtime
            device: Device used (e.g., "cuda", "cpu")
            git_commit: Git commit hash (optional)
            **kwargs: Any additional metadata
        """
        self._metadata = {
            "runtime_seconds": runtime_seconds,
            "device": device,
            "git_commit": git_commit,
            **kwargs
        }

    def set_training_results(
        self,
        final_loss: float,
        final_recon_loss: float,
        final_sparsity_loss: float = 0.0,
        final_num_active: float = 0.0,
        final_pct_active: float = 0.0,
        loss_history: Optional[List[float]] = None,
        recon_loss_history: Optional[List[float]] = None,
        sparsity_history: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Set training results.

        Args:
            final_loss: Final total loss
            final_recon_loss: Final reconstruction loss
            final_sparsity_loss: Final sparsity loss
            final_num_active: Final average number of active features
            final_pct_active: Final percentage of active features
            loss_history: Per-epoch loss values (full array)
            recon_loss_history: Per-epoch reconstruction loss values
            sparsity_history: Per-epoch sparsity percentages
            **kwargs: Any additional training metrics
        """
        self._training_results = {
            "final_metrics": {
                "loss": final_loss,
                "recon_loss": final_recon_loss,
                "sparsity_loss": final_sparsity_loss,
                "num_active": final_num_active,
                "pct_active": final_pct_active,
            },
            "loss_history": loss_history or [],
            "recon_loss_history": recon_loss_history or [],
            "sparsity_history": sparsity_history or [],
            **kwargs
        }

    def set_eval_results(
        self,
        feature_analysis: Optional[List[Dict[str, Any]]] = None,
        dead_features: Optional[Dict[str, Any]] = None,
        spectral_stats: Optional[Dict[str, Any]] = None,
        num_eval_samples: int = 0,
        num_eval_tokens: int = 0,
        **kwargs
    ):
        """
        Set evaluation results.

        Args:
            feature_analysis: List of per-text feature analysis results
                Each item: {"text": str, "top_features": [[idx, val], ...],
                           "num_active": int, "pct_active": float}
            dead_features: Dead feature statistics
                {"count": int, "fraction": float, "threshold": float, "total": int}
            spectral_stats: Spectral analysis of latent space (if computed)
            num_eval_samples: Number of evaluation text samples
            num_eval_tokens: Number of evaluation tokens
            **kwargs: Any additional evaluation metrics
        """
        self._eval_results = {
            "feature_analysis": feature_analysis or [],
            "dead_features": dead_features or {},
            "spectral_stats": spectral_stats or {},
            "num_eval_samples": num_eval_samples,
            "num_eval_tokens": num_eval_tokens,
            **kwargs
        }

    def set_log(self, log_text: str):
        """
        Set the captured stdout log.

        Args:
            log_text: Captured stdout from experiment run
        """
        self._log = log_text

    def set_artifacts(
        self,
        checkpoint_path: str = "",
        **kwargs
    ):
        """
        Set artifact paths.

        Args:
            checkpoint_path: Path to saved checkpoint
            **kwargs: Any additional artifact paths (e.g., plot paths)
        """
        self._artifacts = {
            "checkpoint_path": checkpoint_path,
            **kwargs
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "metadata": {
                "experiment_id": self.experiment_id,
                "description": self.description,
                "timestamp": self.timestamp,
                **self._metadata
            },
            "training_results": self._training_results,
            "eval_results": self._eval_results,
            "artifacts": self._artifacts,
        }

    def save(self, path: str):
        """
        Save report to disk.

        Creates:
            {path}/
            ├── report.json     # Structured data
            ├── report.md       # Human readable
            └── log.txt         # Captured stdout

        Args:
            path: Directory to save report files
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = save_path / "report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save Markdown
        md_path = save_path / "report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self._to_markdown())

        # Save log
        log_path = save_path / "log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(self._log)

        print(f"\n{'='*60}")
        print(f"REPORT SAVED: {self.experiment_id}")
        print(f"{'='*60}")
        print(f"  Location: {save_path.absolute()}")
        print(f"  - report.json (machine readable)")
        print(f"  - report.md (human readable)")
        print(f"  - log.txt (stdout capture)")
        print(f"{'='*60}\n")

    def _to_markdown(self) -> str:
        """Generate human-readable Markdown report."""
        md = []

        # Header
        md.append(f"# Experiment Report: {self.experiment_id}")
        md.append("")
        md.append(f"**Date:** {self.timestamp}")
        md.append(f"**Description:** {self.description}")
        md.append("")

        # Metadata
        if self._metadata:
            md.append("## Metadata")
            md.append("")
            if self._metadata.get("device"):
                md.append(f"- **Device:** {self._metadata['device']}")
            if self._metadata.get("runtime_seconds"):
                runtime = self._metadata['runtime_seconds']
                md.append(f"- **Runtime:** {runtime:.1f}s ({runtime/60:.1f} min)")
            if self._metadata.get("git_commit"):
                md.append(f"- **Git Commit:** {self._metadata['git_commit']}")
            # Any other metadata
            for k, v in self._metadata.items():
                if k not in ["device", "runtime_seconds", "git_commit"]:
                    md.append(f"- **{k}:** {v}")
            md.append("")

        # Training Results
        if self._training_results:
            md.append("## Training Results")
            md.append("")

            final = self._training_results.get("final_metrics", {})
            md.append("### Final Metrics")
            md.append("")
            md.append(f"| Metric | Value |")
            md.append(f"|--------|-------|")
            md.append(f"| Loss | {final.get('loss', 0):.6f} |")
            md.append(f"| Reconstruction Loss | {final.get('recon_loss', 0):.6f} |")
            md.append(f"| Sparsity Loss | {final.get('sparsity_loss', 0):.6f} |")
            md.append(f"| Active Features | {final.get('num_active', 0):.1f} ({final.get('pct_active', 0):.2f}%) |")
            md.append("")

            # Training curve summary
            loss_history = self._training_results.get("loss_history", [])
            if loss_history:
                md.append("### Training Curve")
                md.append("")
                md.append(f"- **Epochs:** {len(loss_history)}")
                md.append(f"- **Initial Loss:** {loss_history[0]:.6f}")
                md.append(f"- **Final Loss:** {loss_history[-1]:.6f}")
                md.append(f"- **Best Loss:** {min(loss_history):.6f}")
                md.append("")

        # Evaluation Results
        if self._eval_results:
            md.append("## Evaluation Results")
            md.append("")

            # Dead features
            dead = self._eval_results.get("dead_features", {})
            if dead:
                md.append("### Dead Features")
                md.append("")
                md.append(f"- **Dead Count:** {dead.get('count', 0)} / {dead.get('total', 0)}")
                md.append(f"- **Dead Fraction:** {dead.get('fraction', 0):.2%}")
                md.append(f"- **Threshold:** {dead.get('threshold', 0.01)}")
                md.append("")

            # Spectral stats
            spectral = self._eval_results.get("spectral_stats", {})
            if spectral:
                md.append("### Spectral Statistics")
                md.append("")
                for k, v in spectral.items():
                    if isinstance(v, float):
                        md.append(f"- **{k}:** {v:.4f}")
                    else:
                        md.append(f"- **{k}:** {v}")
                md.append("")

            # Feature analysis
            feature_analysis = self._eval_results.get("feature_analysis", [])
            if feature_analysis:
                md.append("### Feature Analysis")
                md.append("")
                for i, analysis in enumerate(feature_analysis):
                    text = analysis.get("text", "")
                    # Truncate long texts
                    display_text = text[:50] + "..." if len(text) > 50 else text
                    md.append(f"**Text {i+1}:** \"{display_text}\"")
                    md.append("")
                    md.append(f"- Active features: {analysis.get('num_active', 0)} ({analysis.get('pct_active', 0):.1f}%)")
                    md.append(f"- Top features:")

                    top_features = analysis.get("top_features", [])
                    for feat_idx, feat_val in top_features[:5]:  # Show top 5
                        md.append(f"  - Feature {feat_idx}: {feat_val:.3f}")
                    md.append("")

        # Artifacts
        if self._artifacts and any(self._artifacts.values()):
            md.append("## Artifacts")
            md.append("")
            for name, path in self._artifacts.items():
                if path:
                    md.append(f"- **{name}:** `{path}`")
            md.append("")

        return "\n".join(md)

    @classmethod
    def load(cls, path: str) -> 'ExperimentReport':
        """
        Load report from JSON file.

        Args:
            path: Path to report directory or report.json file

        Returns:
            ExperimentReport instance
        """
        path_obj = Path(path)

        # Find JSON file
        if path_obj.is_dir():
            json_path = path_obj / "report.json"
        else:
            json_path = path_obj

        if not json_path.exists():
            raise FileNotFoundError(f"Report not found: {json_path}")

        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct report
        metadata = data.get("metadata", {})
        report = cls(
            experiment_id=metadata.get("experiment_id", "unknown"),
            description=metadata.get("description", ""),
        )
        report.timestamp = metadata.get("timestamp", "")

        # Extract non-standard metadata fields
        standard_fields = {"experiment_id", "description", "timestamp"}
        extra_metadata = {k: v for k, v in metadata.items() if k not in standard_fields}
        report._metadata = extra_metadata

        report._training_results = data.get("training_results", {})
        report._eval_results = data.get("eval_results", {})
        report._artifacts = data.get("artifacts", {})

        # Try to load log
        if path_obj.is_dir():
            log_path = path_obj / "log.txt"
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8') as f:
                    report._log = f.read()

        return report


def create_experiment_id(prefix: str = "exp") -> str:
    """
    Generate a unique experiment ID.

    Format: {prefix}_{YYYYMMDD}_{HHMMSS}
    Example: exp_20241119_143025

    Args:
        prefix: Prefix for experiment ID

    Returns:
        Unique experiment ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"
