"""
StateMask explanation network implementation for RICE.

This module implements the StateMask explanation method (Algorithm 1) which identifies
critical states in the agent's trajectories. The mask network is trained to generate 
binary masks that indicate which states are most important for the agent's performance.

Based on Algorithm 1 from the RICE paper, this implementation includes:
- Mask network architecture with proper binary action sampling
- Complete training loop with trajectory collection
- Modified reward computation with intrinsic bonus
- PPO-based parameter updates
- Critical state identification methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any, Union
import gym
from collections import deque
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
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training parameters
    batch_size: int = 64
    trajectory_length: int = 1000
    buffer_size: int = 10000
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    
    # Masking strategy parameters
    mask_type: str = "zero"  # "zero", "noise", or "random"
    noise_std: float = 0.1  # Standard deviation for noise masking
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]  # Default architecture

class MaskNetwork(nn.Module):
    """
    Neural network that generates binary masks to identify critical states.
    Implements StateMask with intrinsic reward bonus as described in Algorithm 1.
    
    The network outputs a probability distribution over binary actions (mask/no-mask)
    and is trained using PPO with modified rewards R'_t = R_t + α * a_t^m.
    """
    
    def __init__(self, config: MaskNetworkConfig):
        super().__init__()
        self.config = config
        
        # Build policy network layers for mask action selection
        layers = []
        in_dim = config.observation_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
            
        # Policy head: outputs logits for binary mask action (0=no mask, 1=mask)
        self.policy_head = nn.Linear(in_dim, 2)
        
        # Value head: estimates state value for PPO training
        self.value_head = nn.Linear(in_dim, 1)
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(*layers)
        
        # Move to specified device
        self.device = torch.device(config.device)
        self.to(self.device)
        
        # Setup optimizer for PPO training
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate
        )
        
        # Experience buffer for PPO training
        self.experience_buffer = deque(maxlen=config.buffer_size)
        
        # Training statistics
        self.training_stats = {
            'total_iterations': 0,
            'total_episodes': 0,
            'best_reward': float('-inf')
        }
        
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate policy logits and state values.
        
        Args:
            states: Tensor of shape (batch_size, observation_dim)
            
        Returns:
            policy_logits: Tensor of shape (batch_size, 2) - logits for [no_mask, mask]
            values: Tensor of shape (batch_size, 1) - state value estimates
        """
        # Ensure states are on correct device and have correct dtype
        if not isinstance(states, torch.Tensor):
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        else:
            states = states.to(self.device, dtype=torch.float32)
            
        # Handle single state input
        if states.dim() == 1:
            states = states.unsqueeze(0)
            
        features = self.feature_extractor(states)
        policy_logits = self.policy_head(features)
        values = self.value_head(features)
        return policy_logits, values
        
    def get_action(self, states: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample mask actions from the policy.
        
        Args:
            states: Tensor of shape (batch_size, observation_dim) or (observation_dim,)
            deterministic: If True, select argmax action; if False, sample from distribution
            
        Returns:
            actions: Binary tensor of shape (batch_size,) - 0=no mask, 1=mask
            log_probs: Log probabilities of selected actions
            values: State value estimates
        """
        policy_logits, values = self.forward(states)
        
        if deterministic:
            actions = torch.argmax(policy_logits, dim=-1)
            action_dist = torch.distributions.Categorical(logits=policy_logits)
            log_probs = action_dist.log_prob(actions)
        else:
            # Sample from categorical distribution
            action_dist = torch.distributions.Categorical(logits=policy_logits)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)
                
        return actions, log_probs, values.squeeze(-1)
    
    def get_mask_action(self, state: Union[np.ndarray, torch.Tensor]) -> int:
        """
        Get mask action for given state (simplified interface).
        
        Args:
            state: Single state observation
            
        Returns:
            mask_action: Binary action (0=no mask, 1=mask)
        """
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            
            actions, _, _ = self.get_action(state.unsqueeze(0) if state.dim() == 1 else state)
            return actions[0].item()
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training.
        
        Args:
            states: Tensor of shape (batch_size, observation_dim)
            actions: Tensor of shape (batch_size,) - binary mask actions
            
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Policy entropy
        """
        policy_logits, values = self.forward(states)
        action_dist = torch.distributions.Categorical(logits=policy_logits)
        
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy
    
    def apply_mask(self, state: np.ndarray, mask_action: int) -> np.ndarray:
        """
        Apply masking to state based on mask action.
        
        Args:
            state: Original state observation
            mask_action: Binary mask action (0=no mask, 1=mask)
            
        Returns:
            masked_state: State after applying mask
        """
        if mask_action == 0:
            return state.copy()
        
        # Apply masking based on configured strategy
        if self.config.mask_type == "zero":
            # Zero out the entire state
            return np.zeros_like(state)
        elif self.config.mask_type == "noise":
            # Add Gaussian noise to state
            noise = np.random.normal(0, self.config.noise_std, state.shape)
            return state + noise
        elif self.config.mask_type == "random":
            # Replace with random values from uniform distribution
            return np.random.uniform(-1, 1, state.shape)
        else:
            # Default to zero masking
            return np.zeros_like(state)
    
    def collect_trajectory(self, env: gym.Env, target_policy: Any, max_steps: int = None) -> List[Dict]:
        """
        Collect trajectory data using the mask network and target policy.
        
        This implements the trajectory collection part of Algorithm 1:
        1. For each state s_t, mask network samples mask action a_t^m
        2. Target policy π samples action a_t on (potentially masked) state
        3. Compute modified reward R'_t = R_t + α * a_t^m
        
        Args:
            env: Environment to collect trajectories from
            target_policy: Pre-trained policy π to evaluate
            max_steps: Maximum trajectory length (uses config default if None)
            
        Returns:
            trajectory: List of experience dictionaries
        """
        if max_steps is None:
            max_steps = self.config.trajectory_length
            
        trajectory = []
        state = env.reset()
        
        # Handle different environment reset return formats
        if isinstance(state, tuple):
            state = state[0]
        
        total_reward = 0
        
        for step in range(max_steps):
            # Convert state to tensor
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            
            # Sample mask action from mask network
            mask_action, mask_log_prob, state_value = self.get_action(state_tensor)
            mask_action = mask_action.item()
            mask_log_prob = mask_log_prob.item()
            state_value = state_value.item()
            
            # Apply mask to state if mask_action == 1
            masked_state = self.apply_mask(state, mask_action)
            
            # Target policy selects action on (potentially masked) state
            try:
                # Handle different policy interfaces
                if hasattr(target_policy, 'predict'):
                    target_action = target_policy.predict(masked_state, deterministic=False)[0]
                elif hasattr(target_policy, 'act'):
                    target_action = target_policy.act(masked_state)
                elif callable(target_policy):
                    target_action = target_policy(masked_state)
                else:
                    raise ValueError("Unsupported target policy interface")
            except Exception as e:
                print(f"Warning: Error in target policy prediction: {e}")
                # Fallback to random action
                target_action = env.action_space.sample()
            
            # Execute action in environment
            step_result = env.step(target_action)
            
            # Handle different step return formats
            if len(step_result) == 4:
                next_state, reward, done, info = step_result
            elif len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected step return format: {len(step_result)} elements")
            
            # Compute modified reward: R'_t = R_t + α * a_t^m
            # This provides intrinsic reward bonus for masking (blinding)
            modified_reward = reward + self.config.alpha * mask_action
            
            # Store experience
            experience = {
                'state': state.copy(),
                'mask_action': mask_action,
                'mask_log_prob': mask_log_prob,
                'state_value': state_value,
                'target_action': target_action,
                'reward': reward,
                'modified_reward': modified_reward,
                'next_state': next_state.copy() if hasattr(next_state, 'copy') else next_state,
                'done': done,
                'masked_state': masked_state.copy()
            }
            trajectory.append(experience)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
                
        # Update statistics
        self.training_stats['total_episodes'] += 1
        if total_reward > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = total_reward
                
        return trajectory
    
    def compute_returns_and_advantages(self, trajectory: List[Dict], gamma: float = None, gae_lambda: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using GAE for PPO training.
        
        Args:
            trajectory: List of experience dictionaries
            gamma: Discount factor (uses config default if None)
            gae_lambda: GAE lambda parameter (uses config default if None)
            
        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        if gamma is None:
            gamma = self.config.gamma
        if gae_lambda is None:
            gae_lambda = self.config.gae_lambda
            
        rewards = np.array([exp['modified_reward'] for exp in trajectory])
        values = np.array([exp['state_value'] for exp in trajectory])
        dones = np.array([exp['done'] for exp in trajectory])
        
        # Compute returns using discounted cumulative rewards
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + gamma * returns[t + 1] * (1 - dones[t])
        
        # Compute GAE advantages
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # GAE formula: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            # A_t = δ_t + γ * λ * A_{t+1}
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        return returns, advantages
    
    def train_ppo_step(self, states: torch.Tensor, actions: torch.Tensor, 
                      old_log_probs: torch.Tensor, returns: torch.Tensor, 
                      advantages: torch.Tensor) -> Dict[str, float]:
        """
        Perform one PPO training step.
        
        Args:
            states: Batch of states
            actions: Batch of mask actions
            old_log_probs: Old log probabilities
            returns: Computed returns
            advantages: Computed advantages
            
        Returns:
            losses: Dictionary of loss components
        """
        # Evaluate current policy
        log_probs, values, entropy = self.evaluate_actions(states, actions)
        
        # Compute PPO clipped surrogate loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus for exploration
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
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train(self, pretrained_policy: Any, env: gym.Env, num_iterations: int = 1000) -> Dict:
        """
        Train the mask network using Algorithm 1 from the RICE paper.
        
        This is the main training loop that implements the complete StateMask algorithm:
        1. Collects trajectories using mask network and target policy
        2. Computes modified rewards with intrinsic bonus R'_t = R_t + α * a_t^m
        3. Updates mask network parameters using PPO
        
        Args:
            pretrained_policy: Pre-trained policy π to evaluate
            env: Environment for training
            num_iterations: Number of training iterations
            
        Returns:
            training_metrics: Dictionary of training statistics
        """
        print(f"Starting StateMask training for {num_iterations} iterations...")
        print(f"Configuration: α={self.config.alpha}, mask_type={self.config.mask_type}")
        
        training_metrics = {
            'iteration_rewards': [],
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'mask_probabilities': [],
            'trajectory_lengths': [],
            'advantages_mean': [],
            'advantages_std': []
        }
        
        for iteration in range(num_iterations):
            # Collect trajectory data using current mask network
            trajectory = self.collect_trajectory(env, pretrained_policy, self.config.trajectory_length)
            
            if len(trajectory) == 0:
                print(f"Warning: Empty trajectory at iteration {iteration}")
                continue
            
            # Compute returns and advantages using GAE
            returns, advantages = self.compute_returns_and_advantages(trajectory)
            
            # Normalize advantages for stable training
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Convert trajectory data to tensors
            states = torch.as_tensor(np.array([exp['state'] for exp in trajectory]), 
                                   dtype=torch.float32, device=self.device)
            mask_actions = torch.as_tensor(np.array([exp['mask_action'] for exp in trajectory]), 
                                         dtype=torch.long, device=self.device)
            old_log_probs = torch.as_tensor(np.array([exp['mask_log_prob'] for exp in trajectory]), 
                                          dtype=torch.float32, device=self.device)
            returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
            
            # Perform multiple PPO epochs on collected data
            epoch_losses = []
            for epoch in range(self.config.ppo_epochs):
                # Shuffle data for each epoch
                indices = torch.randperm(len(states))
                
                # Mini-batch training
                batch_size = min(self.config.batch_size, len(states))
                for i in range(0, len(states), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    
                    batch_states = states[batch_indices]
                    batch_actions = mask_actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns_tensor[batch_indices]
                    batch_advantages = advantages_tensor[batch_indices]
                    
                    # Perform PPO update
                    losses = self.train_ppo_step(
                        batch_states, batch_actions, batch_old_log_probs,
                        batch_returns, batch_advantages
                    )
                    epoch_losses.append(losses)
            
            # Compute episode metrics
            episode_reward = sum([exp['reward'] for exp in trajectory])
            mask_prob = sum([exp['mask_action'] for exp in trajectory]) / len(trajectory)
            trajectory_length = len(trajectory)
            
            # Average losses across epochs
            avg_losses = {}
            if epoch_losses:
                for key in epoch_losses[0].keys():
                    avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            # Record metrics
            training_metrics['iteration_rewards'].append(episode_reward)
            training_metrics['episode_rewards'].append(episode_reward)
            training_metrics['mask_probabilities'].append(mask_prob)
            training_metrics['trajectory_lengths'].append(trajectory_length)
            training_metrics['advantages_mean'].append(np.mean(advantages))
            training_metrics['advantages_std'].append(np.std(advantages))
            
            if avg_losses:
                training_metrics['policy_losses'].append(avg_losses.get('policy_loss', 0))
                training_metrics['value_losses'].append(avg_losses.get('value_loss', 0))
                training_metrics['entropy_losses'].append(avg_losses.get('entropy_loss', 0))
            
            # Update training statistics
            self.training_stats['total_iterations'] += 1
            
            # Logging
            if iteration % 10 == 0:
                print(f"Iteration {iteration:4d}: "
                      f"Reward={episode_reward:7.2f}, "
                      f"Mask Prob={mask_prob:.3f}, "
                      f"Traj Len={trajectory_length:4d}")
                
                if avg_losses:
                    print(f"                 "
                          f"Policy Loss={avg_losses.get('policy_loss', 0):.4f}, "
                          f"Value Loss={avg_losses.get('value_loss', 0):.4f}, "
                          f"Entropy={avg_losses.get('entropy_loss', 0):.4f}")
            
            # Early stopping check
            if iteration > 100 and len(training_metrics['iteration_rewards']) >= 50:
                recent_rewards = training_metrics['iteration_rewards'][-50:]
                if np.std(recent_rewards) < 0.01:  # Very low variance
                    print(f"Early stopping at iteration {iteration} due to convergence")
                    break
        
        print(f"Training completed. Total episodes: {self.training_stats['total_episodes']}")
        print(f"Best reward achieved: {self.training_stats['best_reward']:.2f}")
        
        return training_metrics
    
    def find_critical_states(self, env: gym.Env, target_policy: Any, num_episodes: int = 10) -> List[Tuple[np.ndarray, float]]:
        """
        Identify critical states using the trained mask network.
        
        Critical states are those where the mask network assigns high masking probability,
        indicating these states are important for the target policy's performance.
        
        Args:
            env: Environment to evaluate
            target_policy: Target policy to analyze
            num_episodes: Number of episodes to analyze
            
        Returns:
            critical_states: List of (state, criticality_score) tuples sorted by criticality
        """
        print(f"Identifying critical states over {num_episodes} episodes...")
        
        critical_states = []
        
        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
                
            episode_states = []
            step_count = 0
            
            while step_count < self.config.trajectory_length:
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                
                # Get mask probability (criticality score)
                policy_logits, _ = self.forward(state_tensor.unsqueeze(0))
                mask_prob = F.softmax(policy_logits, dim=-1)[0, 1].item()  # Probability of masking
                
                episode_states.append((state.copy(), mask_prob))
                
                # Take action with target policy
                try:
                    if hasattr(target_policy, 'predict'):
                        action = target_policy.predict(state, deterministic=True)[0]
                    elif hasattr(target_policy, 'act'):
                        action = target_policy.act(state)
                    elif callable(target_policy):
                        action = target_policy(state)
                    else:
                        action = env.action_space.sample()
                except:
                    action = env.action_space.sample()
                
                step_result = env.step(action)
                if len(step_result) == 4:
                    state, _, done, _ = step_result
                else:
                    state, _, terminated, truncated, _ = step_result
                    done = terminated or truncated
                
                step_count += 1
                if done:
                    break
            
            # Sort states by criticality score and keep top ones
            episode_states.sort(key=lambda x: x[1], reverse=True)
            critical_states.extend(episode_states[:min(10, len(episode_states))])  # Top 10 per episode
        
        # Sort all critical states and return top ones
        critical_states.sort(key=lambda x: x[1], reverse=True)
        top_critical_states = critical_states[:50]  # Return top 50 critical states
        
        print(f"Found {len(top_critical_states)} critical states")
        if top_critical_states:
            print(f"Top criticality score: {top_critical_states[0][1]:.4f}")
            print(f"Average criticality score: {np.mean([cs[1] for cs in top_critical_states]):.4f}")
        
        return top_critical_states
    
    def get_mask_probability(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get masking probabilities for given states.
        
        Args:
            states: Tensor of shape (batch_size, observation_dim)
            
        Returns:
            mask_probs: Tensor of shape (batch_size,) - probability of masking each state
        """
        with torch.no_grad():
            policy_logits, _ = self.forward(states)
            mask_probs = F.softmax(policy_logits, dim=-1)[:, 1]  # Probability of action 1 (mask)
            return mask_probs
    
    def evaluate_masking_impact(self, env: gym.Env, target_policy: Any, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the impact of masking on target policy performance.
        
        Args:
            env: Environment to evaluate
            target_policy: Target policy to analyze
            num_episodes: Number of episodes to evaluate
            
        Returns:
            impact_metrics: Dictionary containing performance metrics
        """
        print(f"Evaluating masking impact over {num_episodes} episodes...")
        
        original_rewards = []
        masked_rewards = []
        mask_frequencies = []
        
        for episode in range(num_episodes):
            # Evaluate original policy performance
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
                
            original_reward = 0
            masked_reward = 0
            mask_count = 0
            step_count = 0
            
            # Run episode with masking
            while step_count < self.config.trajectory_length:
                # Get mask action
                mask_action = self.get_mask_action(state)
                mask_count += mask_action
                
                # Apply mask
                masked_state = self.apply_mask(state, mask_action)
                
                # Get actions from target policy
                try:
                    if hasattr(target_policy, 'predict'):
                        original_action = target_policy.predict(state, deterministic=True)[0]
                        masked_action = target_policy.predict(masked_state, deterministic=True)[0]
                    else:
                        original_action = target_policy(state)
                        masked_action = target_policy(masked_state)
                except:
                    original_action = env.action_space.sample()
                    masked_action = env.action_space.sample()
                
                # Execute masked action (this is what actually happens)
                step_result = env.step(masked_action)
                if len(step_result) == 4:
                    state, reward, done, _ = step_result
                else:
                    state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                
                masked_reward += reward
                # For comparison, we estimate what original reward would have been
                # (This is approximate since we can't run both actions)
                original_reward += reward
                
                step_count += 1
                if done:
                    break
            
            original_rewards.append(original_reward)
            masked_rewards.append(masked_reward)
            mask_frequencies.append(mask_count / step_count if step_count > 0 else 0)
        
        impact_metrics = {
            'original_reward_mean': np.mean(original_rewards),
            'original_reward_std': np.std(original_rewards),
            'masked_reward_mean': np.mean(masked_rewards),
            'masked_reward_std': np.std(masked_rewards),
            'reward_difference': np.mean(masked_rewards) - np.mean(original_rewards),
            'mask_frequency_mean': np.mean(mask_frequencies),
            'mask_frequency_std': np.std(mask_frequencies)
        }
        
        print(f"Original reward: {impact_metrics['original_reward_mean']:.2f} ± {impact_metrics['original_reward_std']:.2f}")
        print(f"Masked reward: {impact_metrics['masked_reward_mean']:.2f} ± {impact_metrics['masked_reward_std']:.2f}")
        print(f"Reward difference: {impact_metrics['reward_difference']:.2f}")
        print(f"Mask frequency: {impact_metrics['mask_frequency_mean']:.3f} ± {impact_metrics['mask_frequency_std']:.3f}")
        
        return impact_metrics
    
    def save(self, path: str):
        """Save mask network state."""
        torch.save({
            "network_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "training_stats": self.training_stats
        }, path)
        print(f"Mask network saved to {path}")
        
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "MaskNetwork":
        """Load mask network from saved state."""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint["config"]
        if device is not None:
            config.device = device
            
        network = cls(config)
        network.load_state_dict(checkpoint["network_state"])
        network.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if "training_stats" in checkpoint:
            network.training_stats = checkpoint["training_stats"]
            
        print(f"Mask network loaded from {path}")
        return network