"""Configuration for GraphCoarsening experiments."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SpectralConfig:
    """Spectral coarsening hyperparameters."""

    k: int = 100
    alpha: float = 0.75


@dataclass
class ModelConfig:
    """GCN / link-prediction model hyperparameters."""

    hidden_channels: int = 128
    num_layers: int = 3
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 100
    neg_ratio: float = 1.0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42
    device: str = "cuda"

    explanation_datasets: List[str] = field(
        default_factory=lambda: [
            "Cora",
            "Citeseer",
            "PubMed",
            "ogbl-ppa",
            "ogbl-collab",
            "ogbl-ddi",
        ]
    )
    scalability_datasets: List[str] = field(
        default_factory=lambda: [
            "Cora",
            "Citeseer",
            "PubMed",
            "Coauthor-CS",
            "Coauthor-Physics",
            "Amazon-Computers",
            "ogbl-ppa",
            "ogbl-collab",
            "ogbl-ddi",
        ]
    )

    ablation_ratios: List[float] = field(
        default_factory=lambda: [0.3, 0.6, 0.9, 0.95, 0.99]
    )
