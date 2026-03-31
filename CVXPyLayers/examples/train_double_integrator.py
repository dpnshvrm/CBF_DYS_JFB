"""
Example: Train CBF controller for double integrator with 3 obstacles.

This demonstrates:
    - Double integrator dynamics (ẍ = a)
    - Higher-Order CBF (relative degree 2, HOCBF)
    - Multiple obstacles (3)

This matches your naive_soft_for_double_int.py setup but with HARD constraints!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training import TrainingConfig, CBFTrainer
from visualization import plot_trajectories


def main():
    # Configuration (matching your naive_soft_for_double_int.py setup)
    config = TrainingConfig(
        # Dynamics
        dynamics_type='double_integrator',
        position_dim=2,

        # Problem setup
        initial_state=[0.0, 0.0, 0.0, 0.0],  # [px, py, vx, vy]
        target_state=[3.0, 3.0, 0.0, 0.0],   # [px_target, py_target, vx_target, vy_target]

        # Obstacles (same as your naive_soft_for_double_int.py)
        obstacles=[
            {'center': [0.4, 1.0], 'radius': 0.4, 'epsilon': 0.1},
            {'center': [2.2, 2.2], 'radius': 0.4, 'epsilon': 0.1},
            {'center': [2.4, 0.6], 'radius': 0.4, 'epsilon': 0.1},
        ],

        # Time parameters
        T=10.0,
        dt=0.2,

        # HOCBF parameters (relative degree 2)
        cbf_alpha=(5.0, 5.0),  # (alpha1, alpha2) for HOCBF

        # Cost weights
        control_penalty=1.0,
        terminal_cost_weight=100.0,

        # Training
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-3,

        # Saving
        save_path='./models/double_int_three_obs.pth',
        use_double_precision=True
    )

    # Create trainer
    trainer = CBFTrainer(config)

    # Train
    print("\n" + "="*70)
    print("TRAINING: Double Integrator + 3 Obstacles (HOCBF)")
    print("="*70)
    print("This uses HARD constraints (HOCBF) instead of soft penalties!")
    print("="*70)
    policy = trainer.train(verbose=True)

    # Visualize results
    print("\n" + "="*70)
    print("VISUALIZING RESULTS")
    print("="*70)
    plot_trajectories(
        trainer,
        num_trajectories=10,
        show_velocity=False,
        save_path='../models/double_int_trajectories.png'
    )

    print("\nTraining complete!")
    print(f"Model saved to: {config.save_path}")


if __name__ == "__main__":
    main()
