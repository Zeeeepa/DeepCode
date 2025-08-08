"""
PPO (Proximal Policy Optimization) implementation with RICE refinement support.
This implementation allows for both regular PPO training and refinement from mixed initial states.
Enhanced to support intrinsic rewards and exploration bonuses as required by RICE algorithm.

Key algorithmic components following PPO paper:
- Clipped surrogate objective: L_CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
- Value function loss: L_VF(θ) = E[(V_θ(s_t) - V_target)²]
- Entropy bonus: S[π_θ](s) = E[H(π_θ(·|s_t))]
- Combined loss: L_PPO = L_CLIP(θ) - c1*L_VF(θ) + c2*S[π_θ](s)

RICE enhancements:
- Enhanced rewards: R_enhanced = R_task + λ * R_intrinsic
- Critical state initialization for refinement
- Exploration bonuses via RND networks
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import logging

# Set up logging for training monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PPOHyperParams:
    """
    Hyperparameters for PPO training and refinement
    Following standard PPO paper recommendations with RICE extensions
    """
    learning_rate: float = 3e-4
    n_epochs: int = 10
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2  # ε in clipped surrogate objective
    clip_range_vf: Optional[float] = None  # Optional value function clipping
    ent_coef: float = 0.0  # c2 in PPO loss (entropy coefficient)
    vf_coef: float = 0.5  # c1 in PPO loss (value function coefficient)
    max_grad_norm: float = 0.5
    target_kl: float = 0.015  # Early stopping threshold
    # RICE-specific hyperparameters
    lambda_coeff: float = 0.01  # λ coefficient for intrinsic rewards
    reward_normalization: bool = True  # Whether to normalize rewards
    normalize_advantages: bool = True  # Whether to normalize advantages

class RolloutBuffer:
    """
    Storage for collected trajectories during training
    Enhanced to support RICE intrinsic rewards and exploration bonuses
    """
    def __init__(self, size: int, observation_dim: int, action_dim: int):
        self.observations = np.zeros((size, observation_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.intrinsic_rewards = np.zeros(size, dtype=np.float32)  # For exploration bonuses
        self.enhanced_rewards = np.zeros(size, dtype=np.float32)  # Combined rewards
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        
        self.pos = 0
        self.full = False
        self.size = size
        
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            done: bool, value: float, log_prob: float, intrinsic_reward: float = 0.0) -> None:
        """Add a new transition to the buffer"""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.intrinsic_rewards[self.pos] = intrinsic_reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0
    
    def compute_enhanced_rewards(self, lambda_coeff: float = 0.01) -> None:
        """
        Compute enhanced rewards combining task rewards and intrinsic rewards
        Following RICE algorithm: R_enhanced = R_task + λ * R_intrinsic
        
        Args:
            lambda_coeff: Coefficient λ for combining rewards
        """
        self.enhanced_rewards = self.rewards + lambda_coeff * self.intrinsic_rewards
        logger.debug(f"Enhanced rewards computed with λ={lambda_coeff}")
            
    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float, 
                                     use_enhanced_rewards: bool = False) -> None:
        """
        Compute returns and GAE (Generalized Advantage Estimation) advantages
        
        GAE formula: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        
        Args:
            last_value: Value estimate for the last state
            gamma: Discount factor γ
            gae_lambda: GAE parameter λ
            use_enhanced_rewards: If True, use enhanced rewards for advantage computation
        """
        # Choose which rewards to use for advantage computation
        rewards_to_use = self.enhanced_rewards if use_enhanced_rewards else self.rewards
        
        last_gae_lam = 0
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            # GAE computation: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards_to_use[step] + gamma * next_value * next_non_terminal - self.values[step]
            # A_t = δ_t + (γλ)(1-done_{t+1})A_{t+1}
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            
        # Returns are advantages + values: R_t = A_t + V(s_t)
        self.returns = self.advantages + self.values
        
        logger.debug(f"Computed returns and advantages using {'enhanced' if use_enhanced_rewards else 'task'} rewards")
        
    def get_batches(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Get randomized mini-batches from the buffer for training"""
        indices = np.random.permutation(self.size)
        start_idx = 0
        
        while start_idx < self.size:
            end_idx = min(start_idx + batch_size, self.size)
            batch_indices = indices[start_idx:end_idx]
            
            batch = {
                "observations": torch.as_tensor(self.observations[batch_indices]),
                "actions": torch.as_tensor(self.actions[batch_indices]),
                "values": torch.as_tensor(self.values[batch_indices]),
                "log_probs": torch.as_tensor(self.log_probs[batch_indices]),
                "advantages": torch.as_tensor(self.advantages[batch_indices]),
                "returns": torch.as_tensor(self.returns[batch_indices]),
                "rewards": torch.as_tensor(self.rewards[batch_indices]),
                "intrinsic_rewards": torch.as_tensor(self.intrinsic_rewards[batch_indices]),
                "enhanced_rewards": torch.as_tensor(self.enhanced_rewards[batch_indices])
            }
            
            yield batch
            start_idx = end_idx

class PPO:
    """
    PPO agent implementation with RICE refinement capabilities
    
    Implements the PPO algorithm with clipped surrogate objective:
    L_PPO = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)] - c1*L_VF(θ) + c2*S[π_θ](s)
    
    Enhanced to support:
    - Intrinsic rewards and exploration bonuses (RICE)
    - Critical state initialization for refinement
    - Reward normalization and advantage normalization
    """
    def __init__(
        self,
        policy_network: nn.Module,
        value_network: nn.Module,
        observation_dim: int,
        action_dim: int,
        hyperparams: PPOHyperParams = PPOHyperParams(),
        device: str = "cpu"
    ):
        self.policy = policy_network
        self.value_net = value_network
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hp = hyperparams
        self.device = torch.device(device)
        
        self.policy.to(self.device)
        self.value_net.to(self.device)
        
        # Optimizers for policy and value networks
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.hp.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.hp.learning_rate)
        
        self.rollout_buffer = RolloutBuffer(self.hp.n_steps, observation_dim, action_dim)
        
        # Training statistics for monitoring and analysis
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
            "kl_divergence": [],
            "mean_reward": [],
            "mean_intrinsic_reward": [],
            "mean_enhanced_reward": [],
            "explained_variance": [],
            "clip_fraction": []
        }
        
        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        
    def select_action(self, observation: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Select action using current policy
        
        Args:
            observation: Current state observation
            
        Returns:
            Tuple of (action, value_estimate, log_probability)
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation).float().to(self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
                
            action_mean, action_logstd = self.policy(obs_tensor)
            value = self.value_net(obs_tensor)
            
            action_std = torch.exp(action_logstd)
            action_dist = Normal(action_mean, action_std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            
        return action.cpu().numpy().squeeze(), value.cpu().numpy().squeeze(), log_prob.cpu().numpy().squeeze()
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss components following the paper:
        L_PPO = L_CLIP(θ) - c1*L_VF(θ) + c2*S[π_θ](s)
        
        Args:
            batch: Mini-batch of training data
            
        Returns:
            Dictionary containing loss components
        """
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_values = batch["values"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        
        # Normalize advantages (critical for PPO performance)
        if self.hp.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass through policy network
        action_mean, action_logstd = self.policy(obs)
        action_std = torch.exp(action_logstd)
        action_dist = Normal(action_mean, action_std)
        
        # Compute new log probabilities
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        
        # Compute probability ratio: r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate objective: L_CLIP(θ)
        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * torch.clamp(
            ratio, 
            1 - self.hp.clip_range, 
            1 + self.hp.clip_range
        )
        policy_loss = torch.mean(torch.max(policy_loss1, policy_loss2))
        
        # Value function loss: L_VF(θ) = (V_θ(s_t) - V_target)²
        values = self.value_net(obs).squeeze()
        if self.hp.clip_range_vf is None:
            # Unclipped value loss
            value_loss = torch.mean((returns - values) ** 2)
        else:
            # Clipped value loss (optional, helps with stability)
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.hp.clip_range_vf,
                self.hp.clip_range_vf
            )
            value_loss_unclipped = (returns - values) ** 2
            value_loss_clipped = (returns - values_clipped) ** 2
            value_loss = torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))
        
        # Entropy loss for exploration: S[π_θ](s)
        entropy_loss = -torch.mean(action_dist.entropy())
        
        # Combined PPO loss: L_PPO = L_CLIP(θ) - c1*L_VF(θ) + c2*S[π_θ](s)
        total_loss = policy_loss + self.hp.vf_coef * value_loss + self.hp.ent_coef * entropy_loss
        
        # Compute additional metrics for monitoring
        with torch.no_grad():
            # Approximate KL divergence for early stopping
            approx_kl = torch.mean(old_log_probs - log_probs)
            
            # Clip fraction (fraction of ratios that were clipped)
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.hp.clip_range).float())
            
            # Explained variance (how well value function predicts returns)
            y_pred, y_true = values, returns
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
        
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "total_loss": total_loss,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "explained_variance": explained_var
        }
    
    def update(self, rollout_buffer: RolloutBuffer, intrinsic_rewards: Optional[np.ndarray] = None, 
               lambda_coeff: Optional[float] = None) -> Dict[str, float]:
        """
        Update policy with enhanced rewards following RICE algorithm
        
        Implements the full PPO update procedure:
        1. Compute enhanced rewards (task + intrinsic)
        2. Normalize rewards if enabled
        3. Compute advantages and returns using GAE
        4. Update policy for multiple epochs with mini-batches
        5. Apply gradient clipping and early stopping
        
        Args:
            rollout_buffer: Buffer containing collected trajectories
            intrinsic_rewards: Optional array of intrinsic rewards (e.g., from RND)
            lambda_coeff: Coefficient for combining task and intrinsic rewards
            
        Returns:
            Dictionary containing training statistics
        """
        if lambda_coeff is None:
            lambda_coeff = self.hp.lambda_coeff
            
        # Add intrinsic rewards to buffer if provided
        if intrinsic_rewards is not None:
            rollout_buffer.intrinsic_rewards = intrinsic_rewards
            
        # Compute enhanced rewards: R_enhanced = R_task + λ * R_intrinsic
        rollout_buffer.compute_enhanced_rewards(lambda_coeff)
        
        # Update running reward statistics for normalization
        if self.hp.reward_normalization:
            self._update_reward_stats(rollout_buffer.enhanced_rewards)
            # Normalize enhanced rewards using running statistics
            rollout_buffer.enhanced_rewards = (rollout_buffer.enhanced_rewards - self.reward_mean) / (np.sqrt(self.reward_var) + 1e-8)
        
        # Compute advantages and returns using enhanced rewards
        last_obs = torch.as_tensor(rollout_buffer.observations[-1]).float().to(self.device)
        if last_obs.dim() == 1:
            last_obs = last_obs.unsqueeze(0)
            
        with torch.no_grad():
            last_value = self.value_net(last_obs).cpu().numpy().squeeze()
            
        rollout_buffer.compute_returns_and_advantages(
            last_value, self.hp.gamma, self.hp.gae_lambda, use_enhanced_rewards=True
        )
        
        # Collect training statistics across epochs
        epoch_stats = []
        
        # Update policy for multiple epochs (typical PPO procedure)
        for epoch in range(self.hp.n_epochs):
            kl_sum = 0
            n_batches = 0
            epoch_losses = {
                "policy_loss": 0,
                "value_loss": 0,
                "entropy_loss": 0,
                "total_loss": 0,
                "clip_fraction": 0,
                "explained_variance": 0
            }
            
            # Process mini-batches
            for batch in rollout_buffer.get_batches(self.hp.batch_size):
                batch_losses = self.train_step(batch)
                
                # Accumulate statistics
                kl_sum += batch_losses["approx_kl"].item()
                for key in epoch_losses:
                    if key in batch_losses:
                        epoch_losses[key] += batch_losses[key].item()
                n_batches += 1
                
            # Compute epoch averages
            mean_kl = kl_sum / n_batches
            for key in epoch_losses:
                epoch_losses[key] /= n_batches
            epoch_losses["kl"] = mean_kl
            
            # Early stopping based on KL divergence (important for PPO stability)
            if mean_kl > 1.5 * self.hp.target_kl:
                logger.info(f"Early stopping at epoch {epoch} due to large KL divergence: {mean_kl:.4f}")
                break
                
            epoch_stats.append(epoch_losses)
        
        # Compute final statistics across all epochs
        final_stats = {}
        if epoch_stats:
            for key in epoch_stats[0]:
                final_stats[key] = np.mean([s[key] for s in epoch_stats])
        
        # Add reward statistics
        final_stats.update({
            "mean_reward": np.mean(rollout_buffer.rewards),
            "mean_intrinsic_reward": np.mean(rollout_buffer.intrinsic_rewards),
            "mean_enhanced_reward": np.mean(rollout_buffer.enhanced_rewards),
            "reward_std": np.std(rollout_buffer.enhanced_rewards),
            "lambda_coeff": lambda_coeff
        })
        
        # Update training statistics for monitoring
        for key, value in final_stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        return final_stats
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Execute single training step with gradient update
        
        Args:
            batch: Mini-batch of training data
            
        Returns:
            Dictionary containing loss values and metrics
        """
        # Compute all loss components
        losses = self.compute_loss(batch)
        
        # Backward pass and optimization
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        
        losses["total_loss"].backward()
        
        # Gradient clipping for stability (important for PPO)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hp.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.hp.max_grad_norm)
        
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        return losses
    
    def _update_reward_stats(self, rewards: np.ndarray) -> None:
        """Update running statistics for reward normalization"""
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)
        
        # Update running statistics using Welford's online algorithm
        delta = batch_mean - self.reward_mean
        self.reward_count += batch_count
        self.reward_mean += delta * batch_count / self.reward_count
        
        # Update variance
        delta2 = batch_mean - self.reward_mean
        self.reward_var = (self.reward_var * (self.reward_count - batch_count) + 
                          batch_var * batch_count + 
                          delta * delta2 * (self.reward_count - batch_count) * batch_count / self.reward_count) / self.reward_count
        
    def train(self, env, total_timesteps: int, refinement_config: Optional[Dict] = None, 
              eval_callback: Optional[callable] = None, eval_freq: int = 10000) -> None:
        """
        Train the PPO agent with optional RICE refinement
        
        Implements the complete training loop:
        1. Collect rollouts from environment
        2. Apply RICE refinement (critical state initialization)
        3. Compute intrinsic rewards if RND network provided
        4. Update policy using PPO algorithm
        5. Monitor training progress and statistics
        
        Args:
            env: Training environment
            total_timesteps: Total number of environment steps for training
            refinement_config: Optional configuration for RICE refinement:
                - p: Probability of using critical states (float between 0 and 1)
                - critical_states: List of critical states identified by explanation
                - rnd_network: RND network for computing exploration bonuses
                - lambda_coeff: Coefficient for intrinsic rewards
            eval_callback: Optional evaluation callback function
            eval_freq: Frequency of evaluation (in timesteps)
        """
        n_updates = total_timesteps // self.hp.n_steps
        timesteps_collected = 0
        
        logger.info(f"Starting PPO training for {total_timesteps} timesteps ({n_updates} updates)")
        
        for update in range(n_updates):
            # Reset buffer for new rollout
            self.rollout_buffer = RolloutBuffer(self.hp.n_steps, self.observation_dim, self.action_dim)
            
            # Collect rollouts
            obs = env.reset()
            episode_rewards = []
            episode_reward = 0
            
            for step in range(self.hp.n_steps):
                # Apply RICE refinement: mix critical states with random resets
                if refinement_config is not None and np.random.random() < refinement_config.get("p", 0.0):
                    critical_states = refinement_config.get("critical_states", [])
                    if len(critical_states) > 0:
                        critical_idx = np.random.randint(len(critical_states))
                        critical_state = critical_states[critical_idx]
                        
                        # Reset environment to critical state (implementation depends on environment)
                        if hasattr(env, 'set_state'):
                            env.set_state(critical_state)
                            obs = critical_state
                        elif hasattr(env, 'reset_to_state'):
                            obs = env.reset_to_state(critical_state)
                        else:
                            logger.warning("Environment does not support state setting for RICE refinement")
                
                # Select action using current policy
                action, value, log_prob = self.select_action(obs)
                
                # Execute action in environment
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Compute intrinsic reward if RND network provided
                intrinsic_reward = 0.0
                if refinement_config is not None and "rnd_network" in refinement_config:
                    rnd_network = refinement_config["rnd_network"]
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(obs).float().to(self.device)
                        if obs_tensor.dim() == 1:
                            obs_tensor = obs_tensor.unsqueeze(0)
                        intrinsic_reward = rnd_network.compute_intrinsic_reward(obs_tensor).item()
                
                # Store transition in buffer
                self.rollout_buffer.add(obs, action, reward, done, value, log_prob, intrinsic_reward)
                
                obs = next_obs
                timesteps_collected += 1
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    obs = env.reset()
            
            # Prepare update parameters
            intrinsic_rewards = self.rollout_buffer.intrinsic_rewards
            lambda_coeff = refinement_config.get("lambda_coeff", self.hp.lambda_coeff) if refinement_config else self.hp.lambda_coeff
            
            # Update policy with collected data
            update_stats = self.update(self.rollout_buffer, intrinsic_rewards, lambda_coeff)
            
            # Logging and monitoring
            if update % 10 == 0:
                mean_episode_reward = np.mean(episode_rewards) if episode_rewards else 0
                logger.info(
                    f"Update {update}/{n_updates} (timesteps: {timesteps_collected}): "
                    f"Policy Loss: {update_stats.get('policy_loss', 0):.4f}, "
                    f"Value Loss: {update_stats.get('value_loss', 0):.4f}, "
                    f"KL Div: {update_stats.get('kl', 0):.4f}, "
                    f"Episode Reward: {mean_episode_reward:.2f}, "
                    f"Enhanced Reward: {update_stats.get('mean_enhanced_reward', 0):.4f}"
                )
            
            # Evaluation callback
            if eval_callback is not None and timesteps_collected % eval_freq == 0:
                eval_callback(self, timesteps_collected)
        
        logger.info("PPO training completed")
    
    def evaluate(self, env, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate the trained policy
        
        Args:
            env: Evaluation environment
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary containing evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                if deterministic:
                    # Use mean action for deterministic evaluation
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(obs).float().to(self.device)
                        if obs_tensor.dim() == 1:
                            obs_tensor = obs_tensor.unsqueeze(0)
                        action_mean, _ = self.policy(obs_tensor)
                        action = action_mean.cpu().numpy().squeeze()
                else:
                    action, _, _ = self.select_action(obs)
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths)
        }
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Return training statistics for analysis"""
        return self.training_stats.copy()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and training statistics"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'hyperparams': self.hp,
            'training_stats': self.training_stats,
            'reward_stats': {
                'reward_mean': self.reward_mean,
                'reward_var': self.reward_var,
                'reward_count': self.reward_count
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model and restore training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        
        if 'reward_stats' in checkpoint:
            reward_stats = checkpoint['reward_stats']
            self.reward_mean = reward_stats['reward_mean']
            self.reward_var = reward_stats['reward_var']
            self.reward_count = reward_stats['reward_count']
        
        logger.info(f"Model loaded from {filepath}")