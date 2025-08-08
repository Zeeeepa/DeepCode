"""Random Network Distillation (RND) module for exploration.

This module implements the RND exploration mechanism used in RICE to help break through
training bottlenecks. It consists of two networks:
1. A fixed random target network that generates random features
2. A predictor network that tries to predict the target network's output

The prediction error is used as an exploration bonus reward.

Implementation follows the RND paper (Burda et al., 2018) with optimizations for
numerical stability and reward normalization to prevent exploration bonuses from
dominating task rewards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class RNDNetwork(nn.Module):
    """Base network architecture for both target and predictor networks.
    
    Uses orthogonal initialization as recommended in the RND paper for better
    feature diversity and training stability.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [512, 512, 512]):
        """Initialize RND network with deeper architecture for better representation.
        
        Args:
            input_dim: Dimension of input state
            hidden_dims: List of hidden layer dimensions (increased for better capacity)
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with ReLU activation
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
            
        # Output layer produces feature embedding
        # Using 512-dim output for richer feature representation
        layers.append(nn.Linear(prev_dim, 512))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize with orthogonal weights for better feature diversity
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights using orthogonal initialization.
        
        Orthogonal initialization helps maintain gradient flow and ensures
        diverse feature representations in the target network.
        """
        if isinstance(module, nn.Linear):
            # Orthogonal initialization with gain=sqrt(2) for ReLU networks
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            # Zero bias initialization
            if module.bias is not None:
                module.bias.data.zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Feature embedding tensor of shape (batch_size, 512)
        """
        return self.network(x)

class RNDModule:
    """RND module that provides exploration bonuses based on prediction error.
    
    Implements the core RND algorithm:
    1. Fixed target network f(s) generates random features
    2. Trainable predictor network f_hat(s) learns to predict target features
    3. Prediction error ||f(s) - f_hat(s)||^2 serves as exploration bonus
    
    Key optimizations for numerical stability:
    - State normalization using running statistics
    - Reward normalization to prevent exploration dominance
    - Gradient clipping and proper learning rate scheduling
    """
    
    def __init__(
        self, 
        input_dim: int,
        device: str = "cpu",
        hidden_dims: list = [512, 512, 512],
        learning_rate: float = 1e-4,  # Reduced for stability
        update_proportion: float = 0.25,
        reward_scale: float = 1.0,
        reward_clip: float = 5.0,
        obs_norm_clip: float = 5.0,  # Clip normalized observations
        gamma: float = 0.99  # Discount factor for reward normalization
    ):
        """Initialize RND module with enhanced numerical stability.
        
        Args:
            input_dim: Dimension of input state
            device: Device to run networks on ("cpu" or "cuda")
            hidden_dims: Hidden layer dimensions (deeper for better representation)
            learning_rate: Learning rate for predictor network (reduced for stability)
            update_proportion: Proportion of states to use for predictor update
            reward_scale: Scale factor for intrinsic rewards
            reward_clip: Maximum value for clipping intrinsic rewards
            obs_norm_clip: Clipping range for normalized observations
            gamma: Discount factor for reward normalization
        """
        self.device = device
        self.update_proportion = update_proportion
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.obs_norm_clip = obs_norm_clip
        self.gamma = gamma
        
        # Create target and predictor networks with deeper architecture
        self.target_network = RNDNetwork(input_dim, hidden_dims).to(device)
        self.predictor_network = RNDNetwork(input_dim, hidden_dims).to(device)
        
        # Freeze target network weights (f remains fixed throughout training)
        for param in self.target_network.parameters():
            param.requires_grad = False
            
        # Setup optimizer for predictor network with gradient clipping
        self.optimizer = torch.optim.Adam(
            self.predictor_network.parameters(),
            lr=learning_rate,
            eps=1e-4  # Increased epsilon for numerical stability
        )
        
        # Running statistics for observation normalization
        # Using Welford's online algorithm for numerical stability
        self.obs_rms = RunningMeanStd(shape=(input_dim,), device=device)
        
        # Running statistics for reward normalization
        # Prevents exploration rewards from dominating task rewards
        self.reward_rms = RunningMeanStd(shape=(), device=device)
        
        # Training step counter
        self.update_count = 0
        
    def normalize_obs(self, obs: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """Normalize observations using running mean and standard deviation.
        
        Observation normalization is crucial for RND stability as it ensures
        the networks receive inputs in a consistent range.
        
        Args:
            obs: Observations to normalize
            update_stats: Whether to update running statistics
            
        Returns:
            Normalized observations clipped to [-obs_norm_clip, obs_norm_clip]
        """
        if update_stats:
            self.obs_rms.update(obs)
            
        # Normalize: (obs - mean) / sqrt(var + eps)
        normalized_obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
        
        # Clip to prevent extreme values
        normalized_obs = torch.clamp(normalized_obs, -self.obs_norm_clip, self.obs_norm_clip)
        
        return normalized_obs
        
    def compute_intrinsic_reward(self, next_obs, update_stats: bool = True) -> np.ndarray:
        """Compute RND intrinsic reward: R_RND = ||f(s_{t+1}) - f_hat(s_{t+1})||^2.
        
        This is the core RND exploration mechanism. High prediction error indicates
        novel states that should be explored more.
        
        Args:
            next_obs: Next observations to compute intrinsic reward for
            update_stats: Whether to update normalization statistics
            
        Returns:
            Array of normalized intrinsic rewards
        """
        # Convert to tensor if needed
        if isinstance(next_obs, np.ndarray):
            next_obs = torch.FloatTensor(next_obs).to(self.device)
            
        # Ensure proper batch dimension
        if next_obs.dim() == 1:
            next_obs = next_obs.unsqueeze(0)
            
        # Normalize observations
        obs_normalized = self.normalize_obs(next_obs, update_stats=update_stats)
        
        with torch.no_grad():
            # Get target features f(s_{t+1}) - fixed random network
            target_features = self.target_network(obs_normalized)
            
            # Get predicted features f_hat(s_{t+1}) - trainable predictor
            predicted_features = self.predictor_network(obs_normalized)
            
            # Compute MSE prediction error: ||f(s) - f_hat(s)||^2
            # This is the RND exploration bonus formula
            prediction_error = torch.mean((target_features - predicted_features) ** 2, dim=1)
            
        # Convert to numpy
        intrinsic_reward = prediction_error.cpu().numpy()
        
        # Normalize rewards to prevent them from dominating task rewards
        if update_stats:
            self.reward_rms.update(torch.FloatTensor(intrinsic_reward).to(self.device))
            
        # Normalize: reward / sqrt(var + eps)
        normalized_reward = intrinsic_reward / np.sqrt(self.reward_rms.var.cpu().numpy() + 1e-8)
        
        # Apply reward scaling and clipping
        normalized_reward = np.clip(normalized_reward * self.reward_scale, 
                                  -self.reward_clip, self.reward_clip)
        
        return normalized_reward
        
    def update(self, obs_batch: torch.Tensor) -> Dict[str, float]:
        """Update predictor network using batch of observations.
        
        Trains the predictor network f_hat to minimize prediction error against
        the fixed target network f. Uses only a subset of the batch for efficiency.
        
        Args:
            obs_batch: Batch of observations for training
            
        Returns:
            Dictionary containing training metrics
        """
        batch_size = obs_batch.shape[0]
        
        # Sample subset of observations for computational efficiency
        num_updates = max(1, int(self.update_proportion * batch_size))
        indices = torch.randperm(batch_size)[:num_updates]
        update_obs = obs_batch[indices]
        
        # Normalize observations (update stats during training)
        obs_normalized = self.normalize_obs(update_obs, update_stats=True)
        
        # Get target features (fixed network f)
        with torch.no_grad():
            target_features = self.target_network(obs_normalized)
            
        # Get predicted features (trainable network f_hat)
        predicted_features = self.predictor_network(obs_normalized)
        
        # Calculate MSE loss between target and predicted features
        loss = F.mse_loss(predicted_features, target_features)
        
        # Update predictor network parameters
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(self.predictor_network.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        self.update_count += 1
        
        # Calculate metrics for monitoring
        with torch.no_grad():
            # Mean prediction error across full batch
            full_obs_norm = self.normalize_obs(obs_batch, update_stats=False)
            target_full = self.target_network(full_obs_norm)
            pred_full = self.predictor_network(full_obs_norm)
            mean_error = F.mse_loss(pred_full, target_full, reduction='mean').item()
            
            # Intrinsic reward statistics
            intrinsic_rewards = self.compute_intrinsic_reward(obs_batch, update_stats=False)
            
        return {
            "rnd_loss": loss.item(),
            "rnd_mean_error": mean_error,
            "rnd_reward_mean": np.mean(intrinsic_rewards),
            "rnd_reward_std": np.std(intrinsic_rewards),
            "rnd_obs_mean": self.obs_rms.mean.mean().item(),
            "rnd_obs_std": torch.sqrt(self.obs_rms.var.mean()).item(),
            "rnd_update_count": self.update_count
        }
        
    def get_exploration_bonus(self, obs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Calculate exploration bonus for given observations.
        
        Args:
            obs: Observations to calculate bonus for
            training: Whether in training mode (updates running stats)
            
        Returns:
            Tensor of exploration bonuses
        """
        # Normalize observations
        obs_normalized = self.normalize_obs(obs, update_stats=training)
        
        with torch.no_grad():
            # Get target and predicted features
            target_features = self.target_network(obs_normalized)
            predicted_features = self.predictor_network(obs_normalized)
            
            # Calculate MSE prediction error
            prediction_error = F.mse_loss(
                predicted_features,
                target_features,
                reduction='none'
            ).mean(dim=-1)
            
        return prediction_error
        
    def save(self, path: str):
        """Save RND module state including all normalization statistics.
        
        Args:
            path: Path to save state to
        """
        torch.save({
            "target_network": self.target_network.state_dict(),
            "predictor_network": self.predictor_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
            "reward_rms_mean": self.reward_rms.mean,
            "reward_rms_var": self.reward_rms.var,
            "reward_rms_count": self.reward_rms.count,
            "update_count": self.update_count,
            "reward_scale": self.reward_scale,
            "reward_clip": self.reward_clip,
            "obs_norm_clip": self.obs_norm_clip
        }, path)
        
    def load(self, path: str):
        """Load RND module state including all normalization statistics.
        
        Args:
            path: Path to load state from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.predictor_network.load_state_dict(checkpoint["predictor_network"]) 
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Load observation normalization statistics
        self.obs_rms.mean = checkpoint["obs_rms_mean"]
        self.obs_rms.var = checkpoint["obs_rms_var"]
        self.obs_rms.count = checkpoint["obs_rms_count"]
        
        # Load reward normalization statistics
        self.reward_rms.mean = checkpoint["reward_rms_mean"]
        self.reward_rms.var = checkpoint["reward_rms_var"]
        self.reward_rms.count = checkpoint["reward_rms_count"]
        
        # Load training state
        self.update_count = checkpoint.get("update_count", 0)
        
        # Load hyperparameters if available
        if "reward_scale" in checkpoint:
            self.reward_scale = checkpoint["reward_scale"]
            self.reward_clip = checkpoint["reward_clip"]
            self.obs_norm_clip = checkpoint["obs_norm_clip"]


class RunningMeanStd:
    """Tracks running mean and standard deviation using Welford's online algorithm.
    
    This provides numerically stable computation of running statistics, which is
    crucial for proper normalization in RND. Welford's algorithm avoids numerical
    issues that can occur with naive variance computation.
    """
    
    def __init__(self, shape, device="cpu", epsilon=1e-4):
        """Initialize running mean and std tracker.
        
        Args:
            shape: Shape of the data to track statistics for
            device: Device to store tensors on
            epsilon: Small constant for numerical stability
        """
        self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var = torch.ones(shape, device=device, dtype=torch.float32)
        self.count = epsilon
        self.device = device
        
    def update(self, x):
        """Update running statistics with new data using Welford's algorithm.
        
        Welford's algorithm provides numerically stable online computation of
        variance without storing all previous values.
        
        Args:
            x: New data tensor to incorporate into statistics
        """
        if x.dim() > len(self.mean.shape):
            # Handle batch dimension by computing batch statistics first
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, unbiased=False)
            batch_count = x.shape[0]
        else:
            batch_mean = x
            batch_var = torch.zeros_like(x)
            batch_count = 1
            
        # Welford's online algorithm for combining statistics
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # Update mean
        new_mean = self.mean + delta * batch_count / total_count
        
        # Update variance using the parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        
        # Store updated statistics
        self.mean = new_mean
        self.var = M2 / total_count
        self.count = total_count