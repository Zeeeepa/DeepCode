"""
Neural network architectures for RICE implementation.
This module contains the core network architectures used in the RICE paper:
1. Policy and Value networks (using MLPPolicy from stable-baselines3)
2. MaskNetwork for identifying critical states
3. RNDNetwork for exploration bonuses
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from stable_baselines3.common.policies import ActorCriticPolicy

@dataclass
class NetworkConfig:
    """Configuration for network architectures"""
    state_dim: int
    action_dim: int 
    hidden_sizes: List[int] = None
    activation: nn.Module = nn.ReLU
    use_layernorm: bool = False

class MaskNetwork(nn.Module):
    """
    Mask Network that identifies critical states in trajectories.
    Outputs importance scores in [0,1] for each state.
    """
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        if config.hidden_sizes is None:
            config.hidden_sizes = [128, 128]  # Default architecture
            
        layers = []
        prev_dim = config.state_dim
        
        # Build MLP layers
        for hidden_dim in config.hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                config.activation()
            ])
            if config.use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
            
        # Output layer with sigmoid to get scores in [0,1]
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute importance scores
        Args:
            states: Tensor of shape [batch_size, state_dim]
        Returns:
            scores: Tensor of shape [batch_size, 1] with values in [0,1]
        """
        return self.net(states)

class RNDNetwork(nn.Module):
    """
    Random Network Distillation implementation for exploration bonuses.
    Contains a fixed random target network and trainable predictor network.
    """
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        if config.hidden_sizes is None:
            config.hidden_sizes = [128, 128]
            
        # Target network (fixed random weights)
        self.target = self._build_network(config)
        # Predictor network (trainable)
        self.predictor = self._build_network(config)
        
        # Initialize target network with random weights
        for param in self.target.parameters():
            param.requires_grad = False
            
    def _build_network(self, config: NetworkConfig) -> nn.Sequential:
        """Helper to build network architecture"""
        layers = []
        prev_dim = config.state_dim
        
        for hidden_dim in config.hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                config.activation()
            ])
            if config.use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, config.hidden_sizes[-1]))
        
        return nn.Sequential(*layers)
        
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute RND prediction error
        Args:
            states: Tensor of shape [batch_size, state_dim]
        Returns:
            target_features: Target network output
            predicted_features: Predictor network output
        """
        with torch.no_grad():
            target_features = self.target(states)
        predicted_features = self.predictor(states)
        return target_features, predicted_features
        
    def get_exploration_bonus(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute exploration bonus based on prediction error
        Args:
            states: Tensor of shape [batch_size, state_dim]
        Returns:
            bonus: Tensor of shape [batch_size] with exploration bonuses
        """
        target, pred = self.forward(states)
        error = torch.norm(target - pred, dim=-1, p=2)
        return error

class MLPActorCritic(ActorCriticPolicy):
    """
    MLP-based Actor-Critic policy network.
    Inherits from stable-baselines3 ActorCriticPolicy for compatibility.
    Implements core policy and value networks with shared features.
    """
    def __init__(self, 
                 observation_space,
                 action_space,
                 lr_schedule,
                 hidden_sizes: List[int] = None,
                 activation_fn = nn.ReLU,
                 *args, 
                 **kwargs):
                 
        if hidden_sizes is None:
            hidden_sizes = [64, 64]  # Default architecture from SB3
            
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[dict(pi=hidden_sizes, vf=hidden_sizes)],
            activation_fn=activation_fn,
            *args,
            **kwargs
        )