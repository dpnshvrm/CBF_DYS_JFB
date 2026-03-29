"""Training module for CBF-based control."""

from .config import TrainingConfig, single_integrator_two_obstacles, double_integrator_three_obstacles
from .trainer import CBFTrainer

__all__ = [
    'TrainingConfig',
    'single_integrator_two_obstacles',
    'double_integrator_three_obstacles',
    'CBFTrainer',
]
