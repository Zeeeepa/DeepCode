"""
Neural network architectures for RICE implementation.
This module contains the core network architectures used in the RICE paper:
1. Policy and Value networks (using MLPPolicy from stable-baselines3)
2. MaskNetwork for identifying critical states
3. RNDNetwork for exploration bonuses

Paper Architecture Requirements:
- MuJoCo tasks: 2 hidden layers with 64 units each, Tanh activation
- Selfish Mining: 4 hidden layers with 128 units each, Tanh activation
- All networks use Tanh activation as specified in the paper
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from stable_baselines3.common.policies import ActorCriticPolicy

@dataclass
class NetworkConfig:
    """
    Configuration for network architectures following RICE paper specifications.
    
    Paper Requirements:
    - MuJoCo environments: 2 layers × 64 units, Tanh activation
    - Selfish Mining environment: 4 layers × 128 units, Tanh activation
    - All networks use Tanh activation for consistency with paper
    """
    state_dim: int
    action_dim: int 
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])  # Default for MuJoCo
    activation: nn.Module = nn.Tanh  # Paper uses Tanh activation
    use_layernorm: bool = False
    task_type: str = "mujoco"  # "mujoco" or "selfish_mining"
    
    def __post_init__(self):
        """Set task-specific network architectures according to paper specifications"""
        if self.task_type == "mujoco":
            # MuJoCo tasks: 2 hidden layers with 64 units each (Paper Section 4.1)
            self.hidden_sizes = [64, 64]
        elif self.task_type == "selfish_mining":
            # Selfish Mining: 4 hidden layers with 128 units each (Paper Section 4.2)
            self.hidden_sizes = [128, 128, 128, 128]
        
        # Ensure Tanh activation as specified in paper
        self.activation = nn.Tanh

class MaskNetwork(nn.Module):
    """
    Mask Network that identifies critical states in trajectories.
    
    Paper Implementation Details:
    - Outputs importance scores in [0,1] for each state
    - Uses same architecture as policy/value networks
    - Final layer uses sigmoid activation for probability output
    - Critical for identifying important state transitions in RICE algorithm
    """
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        layers = []
        prev_dim = config.state_dim
        
        # Build MLP layers following paper architecture
        for hidden_dim in config.hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                config.activation()  # Tanh activation as per paper
            ])
            if config.use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
            
        # Output layer with sigmoid to get importance scores in [0,1]
        # This is crucial for the RICE algorithm's state importance weighting
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization for Tanh networks
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights appropriately for Tanh activation networks"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute importance scores for RICE algorithm.
        
        Args:
            states: Tensor of shape [batch_size, state_dim]
        Returns:
            scores: Tensor of shape [batch_size, 1] with values in [0,1]
                   representing state importance for trajectory selection
        """
        return self.net(states)

class RNDNetwork(nn.Module):
    """
    Random Network Distillation implementation for exploration bonuses.
    
    Paper Implementation Details:
    - Contains a fixed random target network and trainable predictor network
    - Both networks use same architecture as policy networks
    - Exploration bonus = ||target(s) - predictor(s)||_2
    - Used to encourage exploration in sparse reward environments
    """
    def __init__(self, config: NetworkConfig):
        super().__init__()
        
        # Target network (fixed random weights) - never updated
        self.target = self._build_network(config)
        # Predictor network (trainable) - learns to predict target outputs
        self.predictor = self._build_network(config)
        
        # Freeze target network parameters as per RND paper
        for param in self.target.parameters():
            param.requires_grad = False
            
        # Initialize predictor network weights
        self._initialize_predictor_weights()
            
    def _build_network(self, config: NetworkConfig) -> nn.Sequential:
        """
        Build network architecture following paper specifications.
        Both target and predictor use identical architectures.
        """
        layers = []
        prev_dim = config.state_dim
        
        for hidden_dim in config.hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                config.activation()  # Tanh activation as per paper
            ])
            if config.use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
            
        # Output layer - feature dimension same as last hidden layer
        layers.append(nn.Linear(prev_dim, config.hidden_sizes[-1]))
        
        return nn.Sequential(*layers)
    
    def _initialize_predictor_weights(self):
        """Initialize predictor network weights for Tanh activation"""
        for module in self.predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute RND prediction error for exploration bonus.
        
        Args:
            states: Tensor of shape [batch_size, state_dim]
        Returns:
            target_features: Target network output [batch_size, feature_dim]
            predicted_features: Predictor network output [batch_size, feature_dim]
        """
        with torch.no_grad():
            target_features = self.target(states)
        predicted_features = self.predictor(states)
        return target_features, predicted_features
        
    def get_exploration_bonus(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute exploration bonus based on prediction error.
        
        Paper Formula: bonus = ||f_target(s) - f_predictor(s)||_2
        
        Args:
            states: Tensor of shape [batch_size, state_dim]
        Returns:
            bonus: Tensor of shape [batch_size] with exploration bonuses
        """
        target, pred = self.forward(states)
        # L2 norm of prediction error as exploration bonus
        error = torch.norm(target - pred, dim=-1, p=2)
        return error

class MLPActorCritic(ActorCriticPolicy):
    """
    MLP-based Actor-Critic policy network following RICE paper specifications.
    
    Paper Architecture:
    - MuJoCo: 2 hidden layers × 64 units, Tanh activation
    - Selfish Mining: 4 hidden layers × 128 units, Tanh activation
    - Shared feature extraction with separate policy and value heads
    - Compatible with stable-baselines3 for easy integration
    """
    def __init__(self, 
                 observation_space,
                 action_space,
                 lr_schedule,
                 hidden_sizes: List[int] = None,
                 activation_fn = nn.Tanh,  # Paper uses Tanh activation
                 task_type: str = "mujoco",
                 *args, 
                 **kwargs):
        
        # Set architecture based on task type as per paper
        if hidden_sizes is None:
            if task_type == "mujoco":
                hidden_sizes = [64, 64]  # MuJoCo specification
            elif task_type == "selfish_mining":
                hidden_sizes = [128, 128, 128, 128]  # Selfish Mining specification
            else:
                hidden_sizes = [64, 64]  # Default to MuJoCo
                
        # Use Tanh activation as specified in paper
        activation_fn = nn.Tanh
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[dict(pi=hidden_sizes, vf=hidden_sizes)],
            activation_fn=activation_fn,
            *args,
            **kwargs
        )

def create_network_config(env_name: str, obs_dim: int, action_dim: int) -> NetworkConfig:
    """
    Factory function to create appropriate NetworkConfig based on environment.
    
    Args:
        env_name: Name of the environment
        obs_dim: Observation space dimension
        action_dim: Action space dimension
        
    Returns:
        NetworkConfig with appropriate settings for the environment
    """
    # Determine task type based on environment name
    if "selfish" in env_name.lower() or "mining" in env_name.lower():
        task_type = "selfish_mining"
    else:
        task_type = "mujoco"  # Default for continuous control tasks
    
    return NetworkConfig(
        state_dim=obs_dim,
        action_dim=action_dim,
        task_type=task_type
    )