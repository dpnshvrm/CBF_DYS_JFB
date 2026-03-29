"""
Neural network policy for control.

Simple MLP that maps state to control action.
"""

import torch
import torch.nn as nn
import os


class PolicyNetwork(nn.Module):
    """
    Multi-layer perceptron for control policy.

    Architecture: state → hidden → hidden → hidden → control
    Activation: ReLU (or SiLU for smoother gradients)
    """

    def __init__(self, state_dim, control_dim, hidden_dim=64, num_hidden_layers=3, activation='relu'):
        """
        Initialize policy network.

        Args:
            state_dim: Input dimension (state space)
            control_dim: Output dimension (control space)
            hidden_dim: Hidden layer size
            num_hidden_layers: Number of hidden layers
            activation: 'relu' or 'silu' (Swish)
        """
        super().__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        # Choose activation
        if activation.lower() == 'relu':
            act_fn = nn.ReLU
        elif activation.lower() == 'silu':
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(act_fn())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())

        # Output layer
        layers.append(nn.Linear(hidden_dim, control_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            control: Control tensor (batch_size, control_dim)
        """
        return self.network(state)

    def save(self, filepath, metadata=None):
        """
        Save model checkpoint with metadata.

        Args:
            filepath: Path to save checkpoint
            metadata: Optional dictionary with training info
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'architecture': {
                'state_dim': self.state_dim,
                'control_dim': self.control_dim,
                'hidden_dim': self.hidden_dim,
                'num_hidden_layers': self.num_hidden_layers
            }
        }

        if metadata is not None:
            checkpoint['metadata'] = metadata

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Model saved: {filepath}")

    @staticmethod
    def load(filepath, device=None, dtype=torch.float32):
        """
        Load model from checkpoint.

        Args:
            filepath: Path to checkpoint
            device: Target device (auto-detected if None)
            dtype: Target dtype

        Returns:
            model: Loaded policy network
            metadata: Training metadata (if available)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(filepath, map_location=device)
        arch = checkpoint['architecture']

        model = PolicyNetwork(
            state_dim=arch['state_dim'],
            control_dim=arch['control_dim'],
            hidden_dim=arch['hidden_dim'],
            num_hidden_layers=arch.get('num_hidden_layers', 3)
        ).to(device).to(dtype)

        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = checkpoint.get('metadata', None)

        print(f"Model loaded: {filepath}")
        if metadata:
            print(f"  Epoch {metadata.get('epoch', 'N/A')}, Loss {metadata.get('loss', 'N/A'):.4f}")

        return model, metadata

    def __repr__(self):
        return (f"PolicyNetwork(state_dim={self.state_dim}, "
                f"control_dim={self.control_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"layers={self.num_hidden_layers})")
