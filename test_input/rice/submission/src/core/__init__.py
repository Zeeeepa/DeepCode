"""
Core implementation of RICE (Reinforcement Learning with Explanation)
Components:
- MaskNetwork: Implementation of the explanation method
- PolicyRefiner: Implementation of policy refinement using mixed states
- RNDModule: Random Network Distillation for exploration bonus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class RefinementConfig:
    """Configuration for RICE refinement"""
    p: float = 0.25  # Probability of sampling from critical states vs initial states
    lambda_: float = 0.01  # Exploration bonus coefficient
    alpha: float = 0.0001  # Mask network blinding coefficient
    batch_size: int = 64
    hidden_dim: int = 128
    trajectory_length: int = 1000

class MaskNetwork(nn.Module):
    """
    Implementation of mask network to identify critical states
    Based on the paper's improved version of StateMask
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate importance mask for a given state
        Args:
            state: Input state tensor [batch_size, state_dim]
        Returns:
            mask: Importance scores [batch_size, 1]
        """
        return self.net(state)

    def compute_loss(self, states: torch.Tensor, target_rewards: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Compute mask network loss with blinding incentive
        Args:
            states: Input states [batch_size, state_dim]
            target_rewards: Target rewards [batch_size]
            alpha: Coefficient for blinding incentive
        Returns:
            loss: Scalar loss value
        """
        masks = self.forward(states)
        # Main loss: match mask values to normalized rewards
        reward_loss = F.mse_loss(masks.squeeze(), target_rewards)
        # Blinding incentive: encourage mask to be close to 0 when possible
        blinding_loss = alpha * masks.mean()
        return reward_loss + blinding_loss

class RNDModule(nn.Module):
    """
    Random Network Distillation module for exploration bonus
    Uses a fixed random target network and trainable predictor
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Fixed random target network
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
        
        # Fix target network weights
        for param in self.target.parameters():
            param.requires_grad = False
            
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute RND prediction error (exploration bonus)
        Args:
            state: Input state tensor [batch_size, state_dim]
        Returns:
            bonus: Exploration bonus based on prediction error [batch_size, 1]
        """
        target_feat = self.target(state)
        pred_feat = self.predictor(state)
        error = ((target_feat - pred_feat) ** 2).mean(dim=-1, keepdim=True)
        return error

class PolicyRefiner:
    """
    Main RICE policy refinement class
    Handles mixed state sampling and exploration bonuses
    """
    def __init__(self, 
                 state_dim: int,
                 config: RefinementConfig = RefinementConfig()):
        self.config = config
        self.mask_net = MaskNetwork(state_dim)
        self.rnd = RNDModule(state_dim)
        
    def get_initial_state(self, env, critical_states: List[np.ndarray]) -> np.ndarray:
        """
        Sample initial state from mixed distribution
        Args:
            env: Gym environment
            critical_states: List of identified critical states
        Returns:
            init_state: Initial state for refinement
        """
        if not critical_states or np.random.random() > self.config.p:
            # Sample from default initial distribution
            return env.reset()
        else:
            # Sample from critical states
            idx = np.random.randint(len(critical_states))
            return critical_states[idx].copy()
            
    def compute_exploration_bonus(self, state: np.ndarray) -> float:
        """
        Compute RND exploration bonus for a state
        Args:
            state: Input state array
        Returns:
            bonus: Exploration bonus value
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state)
            bonus = self.rnd(state_t).item()
        return self.config.lambda_ * bonus

    def identify_critical_states(self, 
                               states: List[np.ndarray],
                               rewards: List[float],
                               k: int = 10) -> List[np.ndarray]:
        """
        Identify top-k critical states using mask network
        Args:
            states: List of states from trajectory
            rewards: List of corresponding rewards
            k: Number of critical states to identify
        Returns:
            critical_states: List of k most critical states
        """
        states_t = torch.FloatTensor(np.stack(states))
        with torch.no_grad():
            importance = self.mask_net(states_t).squeeze().numpy()
        
        # Get indices of top-k important states
        top_k_idx = np.argsort(importance)[-k:]
        return [states[i] for i in top_k_idx]