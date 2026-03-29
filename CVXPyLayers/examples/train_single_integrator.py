"""
Example: Train CBF controller for single integrator with 2 obstacles.

This demonstrates:
    - Single integrator dynamics (ẋ = u)
    - Standard CBF (relative degree 1)
    - Multiple obstacles (2)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training import TrainingConfig, CBFTrainer
from visualization import plot_trajectories


def main():
    # Configuration
    config = TrainingConfig(
        # Dynamics
        dynamics_type='single_integrator',
        position_dim=2,

        # Problem setup
        initial_state=[0.0, 0.0],
        target_state=[4.0, 4.0],

        # Obstacles
        obstacles=[
            {'center': [1.0, 1.0], 'radius': 0.55, 'epsilon': 0.1},
            {'center': [2.5, 2.5], 'radius': 0.55, 'epsilon': 0.1},
        ],

        # Time parameters
        T=6.0,
        dt=0.05,

        # CBF parameters
        cbf_alpha=10.0,  # Standard CBF class-K parameter

        # Cost weights
        control_penalty=1.0,
        terminal_cost_weight=500.0,

        # Training
        num_epochs=300,
        batch_size=32,
        learning_rate=1e-3,

        # Saving
        save_path='./models/single_int_two_obs.pth',
        use_double_precision=True
    )

    # Create trainer
    trainer = CBFTrainer(config)

    # Train
    print("\n" + "="*70)
    print("TRAINING: Single Integrator + 2 Obstacles (Standard CBF)")
    print("="*70)
    policy = trainer.train(verbose=True)

    # Visualize results
    print("\n" + "="*70)
    print("VISUALIZING RESULTS")
    print("="*70)
    plot_trajectories(
        trainer,
        num_trajectories=10,
        save_path='../models/single_int_trajectories.png'
    )

    print("\nTraining complete!")
    print(f"Model saved to: {config.save_path}")


if __name__ == "__main__":
    main()
