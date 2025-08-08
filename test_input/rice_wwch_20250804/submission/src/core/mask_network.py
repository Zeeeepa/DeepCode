"""
StateMask explanation network implementation for RICE.

This module implements the StateMask explanation method (Algorithm 1) which identifies
critical states in the agent's trajectories. The mask network is trained to generate 
binary masks that indicate which states are most important for the agent's performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any
import random

@dataclass
class MaskNetworkConfig:
    """Configuration for the mask network."""
    observation_dim: int
    hidden_dims: List[int] = None  # Hidden layer dimensions
    alpha: float = 0.0001  # Coefficient for blinding reward bonus (intrinsic reward)
    learning_rate: float = 3e-4
    device: str = "cpu"
    
    # PPO-specific parameters for mask network training
    ppo_epochs: int = 10
    ppo_clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]  # Default architecture

class MaskNetwork(nn.Module):
    """
    Neural network that generates binary masks to identify critical states.
    Implements StateMask with PPO training as described in Algorithm 1.
    
    The network outputs probabilities for masking each state, and is trained
    using PPO with intrinsic rewards to encourage identifying critical states.
    """
    
    def __init__(self, config: MaskNetworkConfig):
        super().__init__()
        self.config = config
        
        # Build policy network layers (outputs mask probabilities)
        policy_layers = []
        in_dim = config.observation_dim
        for hidden_dim in config.hidden_dims:
            policy_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh()  # Use Tanh as in original paper
            ])
            in_dim = hidden_dim
            
        # Output layer produces mask probability (0 = no mask, 1 = mask)
        policy_layers.append(nn.Linear(in_dim, 1))
        policy_layers.append(nn.Sigmoid())  # Output probability of masking
        self.policy_network = nn.Sequential(*policy_layers)
        
        # Build value network for PPO training
        value_layers = []
        in_dim = config.observation_dim
        for hidden_dim in config.hidden_dims:
            value_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh()
            ])
            in_dim = hidden_dim
        value_layers.append(nn.Linear(in_dim, 1))
        self.value_network = nn.Sequential(*value_layers)
        
        # Move to specified device
        self.device = torch.device(config.device)
        self.to(self.device)
        
        # Setup optimizer for both networks
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate
        )
        
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate mask probabilities and state values.
        
        Args:
            states: Tensor of shape (batch_size, observation_dim)
            
        Returns:
            mask_probs: Tensor of shape (batch_size, 1) - probability of masking
            state_values: Tensor of shape (batch_size, 1) - estimated state values
        """
        mask_probs = self.policy_network(states)
        state_values = self.value_network(states)
        return mask_probs, state_values
        
    def get_mask_action(self, states: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample mask actions from the policy.
        
        Args:
            states: Tensor of shape (batch_size, observation_dim)
            deterministic: If True, use deterministic policy (prob > 0.5)
            
        Returns:
            actions: Binary tensor of shape (batch_size, 1) - mask actions
            log_probs: Log probabilities of the actions
        """
        with torch.no_grad() if deterministic else torch.enable_grad():
            mask_probs, _ = self.forward(states)
            
            if deterministic:
                actions = (mask_probs > 0.5).float()
                # For deterministic case, compute log prob of the chosen action
                log_probs = torch.log(torch.where(actions == 1, mask_probs, 1 - mask_probs))
            else:
                # Sample from Bernoulli distribution
                dist = torch.distributions.Bernoulli(mask_probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                
        return actions, log_probs
        
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training.
        
        Args:
            states: Tensor of shape (batch_size, observation_dim)
            actions: Tensor of shape (batch_size, 1) - mask actions
            
        Returns:
            log_probs: Log probabilities of actions
            state_values: Estimated state values
            entropy: Policy entropy
        """
        mask_probs, state_values = self.forward(states)
        
        # Compute log probabilities
        dist = torch.distributions.Bernoulli(mask_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, state_values.squeeze(-1), entropy
        
    def train_mask_network(
        self,
        trajectories: List[Dict[str, Any]],
        target_policy: Any,
        env: Any,
        num_episodes: int = 100
    ) -> Dict[str, List[float]]:
        """
        Train the mask network using Algorithm 1 from the paper.
        
        This implements the core StateMask training procedure:
        1. Collect trajectories using pre-trained policy π and mask network
        2. Compute intrinsic rewards R'_t = R_t + α * a_t^m
        3. Update mask network parameters using PPO
        
        Args:
            trajectories: List of trajectory dictionaries with 'states', 'actions', 'rewards'
            target_policy: Pre-trained policy π for action sampling
            env: Environment for trajectory collection
            num_episodes: Number of training episodes
            
        Returns:
            training_metrics: Dictionary containing training statistics
        """
        training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'mask_probability': [],
            'intrinsic_reward': []
        }
        
        for episode in range(num_episodes):
            # Step 1: Collect trajectory data
            episode_data = self._collect_episode_data(target_policy, env)
            
            if not episode_data:
                continue
                
            # Step 2: Compute intrinsic rewards (Equation from Algorithm 1)
            # R'_t = R_t + α * a_t^m where a_t^m is the mask action
            intrinsic_rewards = self._compute_intrinsic_rewards(episode_data)
            
            # Step 3: Update mask network using PPO
            ppo_metrics = self._update_with_ppo(episode_data, intrinsic_rewards)
            
            # Record metrics
            for key, value in ppo_metrics.items():
                if key in training_metrics:
                    training_metrics[key].append(value)
                    
            # Log progress
            if episode % 10 == 0:
                avg_mask_prob = np.mean(training_metrics['mask_probability'][-10:]) if training_metrics['mask_probability'] else 0
                avg_intrinsic = np.mean(training_metrics['intrinsic_reward'][-10:]) if training_metrics['intrinsic_reward'] else 0
                print(f"Episode {episode}: Avg Mask Prob = {avg_mask_prob:.4f}, Avg Intrinsic Reward = {avg_intrinsic:.4f}")
                
        return training_metrics
        
    def _collect_episode_data(self, target_policy: Any, env: Any) -> Dict[str, torch.Tensor]:
        """
        Collect one episode of trajectory data using target policy and mask network.
        
        Args:
            target_policy: Pre-trained policy π
            env: Environment
            
        Returns:
            episode_data: Dictionary with states, actions, rewards, mask_actions, log_probs
        """
        states, actions, rewards, mask_actions, log_probs, values = [], [], [], [], [], []
        
        state = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from target policy π
            with torch.no_grad():
                if hasattr(target_policy, 'get_action'):
                    action = target_policy.get_action(state)
                else:
                    # Fallback for different policy interfaces
                    action = target_policy(state_tensor).cpu().numpy()
                    
            # Get mask action from mask network
            mask_action, log_prob = self.get_mask_action(state_tensor)
            _, value = self.forward(state_tensor)
            
            # Execute action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store trajectory data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            mask_actions.append(mask_action.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
            values.append(value.cpu().numpy())
            
            state = next_state
            
        # Convert to tensors
        episode_data = {
            'states': torch.FloatTensor(np.array(states)).to(self.device),
            'actions': torch.FloatTensor(np.array(actions)).to(self.device),
            'rewards': torch.FloatTensor(np.array(rewards)).to(self.device),
            'mask_actions': torch.FloatTensor(np.array(mask_actions)).to(self.device),
            'old_log_probs': torch.FloatTensor(np.array(log_probs)).to(self.device),
            'values': torch.FloatTensor(np.array(values)).to(self.device)
        }
        
        return episode_data
        
    def _compute_intrinsic_rewards(self, episode_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute intrinsic rewards according to Algorithm 1.
        
        Formula: R'_t = R_t + α * a_t^m
        where a_t^m is the mask action (1 for mask, 0 for no mask)
        
        Args:
            episode_data: Episode trajectory data
            
        Returns:
            intrinsic_rewards: Modified rewards with intrinsic bonus
        """
        original_rewards = episode_data['rewards']
        mask_actions = episode_data['mask_actions'].squeeze(-1)
        
        # Apply intrinsic reward formula from Algorithm 1
        # α * a_t^m gives bonus for masking (blinding) states
        intrinsic_bonus = self.config.alpha * mask_actions
        intrinsic_rewards = original_rewards + intrinsic_bonus
        
        return intrinsic_rewards
        
    def _update_with_ppo(self, episode_data: Dict[str, torch.Tensor], intrinsic_rewards: torch.Tensor) -> Dict[str, float]:
        """
        Update mask network parameters using PPO algorithm.
        
        Args:
            episode_data: Episode trajectory data
            intrinsic_rewards: Rewards with intrinsic bonus
            
        Returns:
            metrics: Training metrics for this update
        """
        states = episode_data['states']
        mask_actions = episode_data['mask_actions']
        old_log_probs = episode_data['old_log_probs'].squeeze(-1)
        old_values = episode_data['values'].squeeze(-1)
        
        # Compute returns and advantages
        returns = self._compute_returns(intrinsic_rewards)
        advantages = returns - old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # PPO update epochs
        for _ in range(self.config.ppo_epochs):
            # Evaluate current policy
            log_probs, values, entropy = self.evaluate_actions(states, mask_actions)
            
            # Compute ratio for PPO clipping
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Compute policy loss with clipping
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values, returns)
            
            # Compute entropy loss (for exploration)
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.config.value_loss_coef * value_loss + 
                         self.config.entropy_coef * entropy_loss)
            
            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            
        # Compute average metrics
        num_epochs = self.config.ppo_epochs
        metrics = {
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs,
            'entropy_loss': total_entropy_loss / num_epochs,
            'total_loss': (total_policy_loss + total_value_loss + total_entropy_loss) / num_epochs,
            'mask_probability': mask_actions.mean().item(),
            'intrinsic_reward': intrinsic_rewards.mean().item()
        }
        
        return metrics
        
    def _compute_returns(self, rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """
        Compute discounted returns for the episode.
        
        Args:
            rewards: Episode rewards
            gamma: Discount factor
            
        Returns:
            returns: Discounted returns
        """
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
            
        return returns
        
    def get_mask(self, states: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Generate binary masks for given states.
        
        Args:
            states: Tensor of shape (batch_size, observation_dim)
            deterministic: Whether to use deterministic policy
            
        Returns:
            masks: Binary tensor of shape (batch_size, 1)
        """
        masks, _ = self.get_mask_action(states, deterministic=deterministic)
        return masks
        
    def save(self, path: str):
        """Save mask network state."""
        torch.save({
            "network_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config
        }, path)
        
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "MaskNetwork":
        """Load mask network from saved state."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        if device is not None:
            config.device = device
            
        network = cls(config)
        network.load_state_dict(checkpoint["network_state"])
        network.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return network