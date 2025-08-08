"""
RICE (Reinforcement Learning with Explanation) Core Module

This module implements the core components of the RICE algorithm as described in the paper.
The RICE algorithm combines reinforcement learning with explanation methods to improve
policy learning through critical state identification and mixed state sampling.

Components:
- MaskNetwork: Neural network for identifying critical states
- RNDModule: Random Network Distillation for exploration bonus computation
- PolicyRefiner: Main refinement algorithm implementation
- RefinementConfig: Configuration dataclass for algorithm parameters

Version: 1.0.0
Author: RICE Implementation Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

# Version information
__version__ = "1.0.0"
__author__ = "RICE Implementation Team"

@dataclass
class RefinementConfig:
    """
    Configuration class for RICE refinement algorithm parameters.
    
    This dataclass contains all hyperparameters used in the RICE algorithm,
    following the specifications from the original paper.
    
    Attributes:
        p (float): Probability of sampling from critical states vs initial states.
                  Paper suggests 0.25 as optimal value.
        lambda_ (float): Exploration bonus coefficient for RND module.
                        Controls the strength of exploration incentive.
        alpha (float): Mask network blinding coefficient.
                      Encourages sparsity in critical state identification.
        batch_size (int): Batch size for training neural networks.
        hidden_dim (int): Hidden dimension for neural network layers.
        trajectory_length (int): Maximum length of trajectories for evaluation.
        learning_rate (float): Learning rate for neural network optimization.
        device (str): Device for computation ('cpu' or 'cuda').
    """
    p: float = 0.25  # Critical state sampling probability
    lambda_: float = 0.01  # Exploration bonus coefficient
    alpha: float = 0.0001  # Mask network blinding coefficient
    batch_size: int = 64
    hidden_dim: int = 128
    trajectory_length: int = 1000
    learning_rate: float = 1e-3
    device: str = 'cpu'

class MaskNetwork(nn.Module):
    """
    Neural network implementation for identifying critical states in trajectories.
    
    This network implements an improved version of StateMask as described in the paper.
    It learns to assign importance scores to states based on their contribution to
    the final reward, with a blinding incentive to encourage sparsity.
    
    The network architecture consists of:
    - Input layer: state_dim -> hidden_dim
    - Hidden layers: hidden_dim -> hidden_dim (with ReLU activation)
    - Output layer: hidden_dim -> 1 (with Sigmoid activation)
    
    Args:
        state_dim (int): Dimension of the input state space
        hidden_dim (int): Dimension of hidden layers (default: 128)
        device (str): Device for computation (default: 'cpu')
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128, device: str = 'cpu'):
        super(MaskNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Neural network architecture following paper specifications
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output importance scores in [0, 1]
        )
        
        # Move to specified device
        self.to(device)
        
        # Initialize weights using Xavier initialization for better convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate importance mask for given states.
        
        Args:
            state: Input state tensor of shape [batch_size, state_dim]
            
        Returns:
            mask: Importance scores tensor of shape [batch_size, 1]
                 Values are in range [0, 1] where 1 indicates high importance
        """
        # Ensure input is on correct device
        if state.device != self.device:
            state = state.to(self.device)
            
        return self.net(state)

    def compute_loss(self, 
                    states: torch.Tensor, 
                    target_rewards: torch.Tensor, 
                    alpha: float) -> torch.Tensor:
        """
        Compute mask network loss with blinding incentive.
        
        The loss function combines:
        1. Reward prediction loss: MSE between mask values and normalized rewards
        2. Blinding incentive: L1 penalty to encourage sparsity
        
        Args:
            states: Input states tensor [batch_size, state_dim]
            target_rewards: Target rewards tensor [batch_size]
            alpha: Coefficient for blinding incentive (sparsity regularization)
            
        Returns:
            loss: Scalar loss value combining both components
        """
        # Ensure inputs are on correct device
        states = states.to(self.device)
        target_rewards = target_rewards.to(self.device)
        
        # Generate importance masks
        masks = self.forward(states).squeeze()
        
        # Normalize target rewards to [0, 1] range for better training stability
        if target_rewards.numel() > 1:
            reward_min = target_rewards.min()
            reward_max = target_rewards.max()
            if reward_max > reward_min:
                normalized_rewards = (target_rewards - reward_min) / (reward_max - reward_min)
            else:
                normalized_rewards = torch.zeros_like(target_rewards)
        else:
            normalized_rewards = target_rewards
        
        # Main loss: match mask values to normalized rewards
        reward_loss = F.mse_loss(masks, normalized_rewards)
        
        # Blinding incentive: encourage sparsity (L1 regularization)
        blinding_loss = alpha * masks.mean()
        
        total_loss = reward_loss + blinding_loss
        
        return total_loss

class RNDModule(nn.Module):
    """
    Random Network Distillation module for computing exploration bonuses.
    
    This module implements the RND algorithm which uses prediction error of a
    randomly initialized target network as an exploration bonus. The intuition
    is that states that are harder to predict (higher error) are less visited
    and thus should receive higher exploration bonuses.
    
    Architecture:
    - Target network: Fixed random weights, not trainable
    - Predictor network: Trainable, learns to predict target network outputs
    
    Args:
        state_dim (int): Dimension of the input state space
        hidden_dim (int): Dimension of hidden layers (default: 128)
        device (str): Device for computation (default: 'cpu')
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128, device: str = 'cpu'):
        super(RNDModule, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Fixed random target network (not trainable)
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Trainable predictor network
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Move to specified device
        self.to(device)
        
        # Fix target network weights (no gradient computation)
        for param in self.target.parameters():
            param.requires_grad = False
            
        # Initialize predictor weights
        self._initialize_predictor_weights()
            
    def _initialize_predictor_weights(self):
        """Initialize predictor network weights using Xavier initialization."""
        for module in self.predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute RND prediction error as exploration bonus.
        
        The exploration bonus is computed as the mean squared error between
        the target network output and predictor network output.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            bonus: Exploration bonus based on prediction error [batch_size, 1]
                  Higher values indicate less familiar states
        """
        # Ensure input is on correct device
        if state.device != self.device:
            state = state.to(self.device)
            
        # Compute target and predicted features
        with torch.no_grad():
            target_feat = self.target(state)
        pred_feat = self.predictor(state)
        
        # Compute prediction error (exploration bonus)
        error = ((target_feat - pred_feat) ** 2).mean(dim=-1, keepdim=True)
        
        return error
    
    def update_predictor(self, states: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Update predictor network to minimize prediction error.
        
        Args:
            states: Batch of states for training [batch_size, state_dim]
            optimizer: Optimizer for predictor network
            
        Returns:
            loss: Training loss value
        """
        states = states.to(self.device)
        
        # Compute prediction loss
        with torch.no_grad():
            target_feat = self.target(states)
        pred_feat = self.predictor(states)
        
        loss = F.mse_loss(pred_feat, target_feat)
        
        # Update predictor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

class PolicyRefiner:
    """
    Main RICE policy refinement implementation.
    
    This class orchestrates the RICE algorithm by:
    1. Identifying critical states using the mask network
    2. Sampling from mixed initial state distribution
    3. Computing exploration bonuses using RND
    4. Managing the overall refinement process
    
    Args:
        state_dim (int): Dimension of the state space
        config (RefinementConfig): Configuration object with algorithm parameters
    """
    
    def __init__(self, 
                 state_dim: int,
                 config: Optional[RefinementConfig] = None):
        
        # Use default config if none provided
        if config is None:
            config = RefinementConfig()
        
        self.config = config
        self.state_dim = state_dim
        
        # Initialize neural network components
        self.mask_net = MaskNetwork(
            state_dim=state_dim, 
            hidden_dim=config.hidden_dim,
            device=config.device
        )
        
        self.rnd = RNDModule(
            state_dim=state_dim,
            hidden_dim=config.hidden_dim,
            device=config.device
        )
        
        # Initialize optimizers
        self.mask_optimizer = torch.optim.Adam(
            self.mask_net.parameters(), 
            lr=config.learning_rate
        )
        
        self.rnd_optimizer = torch.optim.Adam(
            self.rnd.predictor.parameters(),
            lr=config.learning_rate
        )
        
        # Storage for critical states
        self.critical_states = []
        
    def get_initial_state(self, env, critical_states: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Sample initial state from mixed distribution.
        
        With probability p, sample from critical states; otherwise sample from
        the environment's default initial state distribution.
        
        Args:
            env: Gym environment with reset() method
            critical_states: List of identified critical states (optional)
            
        Returns:
            init_state: Initial state for policy refinement
        """
        # Use stored critical states if none provided
        if critical_states is None:
            critical_states = self.critical_states
            
        # Sample from mixed distribution
        if not critical_states or np.random.random() > self.config.p:
            # Sample from default initial distribution
            state = env.reset()
            # Handle both old and new gym API
            if isinstance(state, tuple):
                state = state[0]
            return state
        else:
            # Sample from critical states
            idx = np.random.randint(len(critical_states))
            return critical_states[idx].copy()
            
    def compute_exploration_bonus(self, state: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute RND exploration bonus for a state.
        
        Args:
            state: Input state (numpy array or torch tensor)
            
        Returns:
            bonus: Exploration bonus value scaled by lambda_
        """
        # Convert to tensor if necessary
        if isinstance(state, np.ndarray):
            state_t = torch.FloatTensor(state).unsqueeze(0)
        else:
            state_t = state.unsqueeze(0) if state.dim() == 1 else state
            
        with torch.no_grad():
            bonus = self.rnd(state_t).item()
            
        return self.config.lambda_ * bonus

    def identify_critical_states(self, 
                               states: List[np.ndarray],
                               rewards: List[float],
                               k: int = 10) -> List[np.ndarray]:
        """
        Identify top-k critical states using the mask network.
        
        Critical states are those with highest importance scores as determined
        by the mask network. These states are considered most influential for
        the final outcome.
        
        Args:
            states: List of states from trajectory
            rewards: List of corresponding rewards
            k: Number of critical states to identify
            
        Returns:
            critical_states: List of k most critical states
        """
        if not states:
            return []
            
        # Convert to tensors
        states_array = np.stack(states)
        states_t = torch.FloatTensor(states_array)
        
        # Compute importance scores
        with torch.no_grad():
            importance = self.mask_net(states_t).squeeze().cpu().numpy()
        
        # Handle single state case
        if importance.ndim == 0:
            importance = np.array([importance])
            
        # Get indices of top-k important states
        k = min(k, len(states))  # Ensure k doesn't exceed available states
        top_k_idx = np.argsort(importance)[-k:]
        
        critical_states = [states[i] for i in top_k_idx]
        
        # Update stored critical states
        self.critical_states = critical_states
        
        return critical_states
    
    def update_mask_network(self, 
                          states: List[np.ndarray], 
                          rewards: List[float]) -> float:
        """
        Update mask network using trajectory data.
        
        Args:
            states: List of states from trajectory
            rewards: List of corresponding rewards
            
        Returns:
            loss: Training loss value
        """
        if not states or not rewards:
            return 0.0
            
        # Convert to tensors
        states_t = torch.FloatTensor(np.stack(states))
        rewards_t = torch.FloatTensor(rewards)
        
        # Compute loss and update
        loss = self.mask_net.compute_loss(states_t, rewards_t, self.config.alpha)
        
        self.mask_optimizer.zero_grad()
        loss.backward()
        self.mask_optimizer.step()
        
        return loss.item()
    
    def update_rnd_network(self, states: List[np.ndarray]) -> float:
        """
        Update RND predictor network.
        
        Args:
            states: List of states for training
            
        Returns:
            loss: Training loss value
        """
        if not states:
            return 0.0
            
        states_t = torch.FloatTensor(np.stack(states))
        return self.rnd.update_predictor(states_t, self.rnd_optimizer)
    
    def save_networks(self, filepath: str):
        """Save trained networks to file."""
        torch.save({
            'mask_net_state_dict': self.mask_net.state_dict(),
            'rnd_predictor_state_dict': self.rnd.predictor.state_dict(),
            'config': self.config,
            'critical_states': self.critical_states
        }, filepath)
    
    def load_networks(self, filepath: str):
        """Load trained networks from file."""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        self.mask_net.load_state_dict(checkpoint['mask_net_state_dict'])
        self.rnd.predictor.load_state_dict(checkpoint['rnd_predictor_state_dict'])
        if 'critical_states' in checkpoint:
            self.critical_states = checkpoint['critical_states']

# Export all public classes and functions
__all__ = [
    'RefinementConfig',
    'MaskNetwork', 
    'RNDModule',
    'PolicyRefiner',
    '__version__',
    '__author__'
]