"""
Steering Vector Library

A unified library for computing and analyzing CAA (Contrastive Activation Addition)
steering vectors for collusion detection in language models.

Modules:
    data_loader: Load and process collusion-labeled datasets
    activation_extractor: Extract model activations from all layers
    caa_trainer: Compute CAA vectors and performance metrics
"""

__version__ = "2.0.0"

from .data_loader import DataLoader
from .activation_extractor import ActivationExtractor
from .caa_trainer import CAATrainer
from .visualization import Visualizer

__all__ = [
    "DataLoader",
    "ActivationExtractor",
    "CAATrainer",
    "Visualizer",
]
