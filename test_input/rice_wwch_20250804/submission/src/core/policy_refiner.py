"""
RICE Policy Refiner implementation that integrates mask network explanations with RND exploration.
Implements Algorithm 2: RICE Refining from the paper.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .mask_network import MaskNetwork
from ..models.ppo import PPO


@dataclass
class RefinementConfig:
    """Configuration for RICE policy refinement"""
    # Mixing probability for critical vs initial states (p in paper)
    mix_prob: float = 0.25
    # Exploration bonus coefficient (λ in paper)
    exploration_lambda: float = 0.01
    # RND network hidden dimensions
    rnd_hidden_dims: List[int] = (256, 256)
    # Training iterations
    n_iterations: int = 500
    # PPO parameters
    ppo_learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_clip_range: float = 0.2
    # Entropy coefficient for PPO exploration (paper mentions 0.01)
    entropy_coef: float = 0.01
    # Rollout collection parameters
    max_episode_steps: int = 1000
    rollout_buffer_size: int = 2048
    # RND specific parameters
    rnd_learning_rate: float = 1e-4
    rnd_update_frequency: int = 1
    intrinsic_reward_norm: bool = True
    intrinsic_reward_clip: float = 5.0
    # Numerical stability parameters
    epsilon: float = 1e-8
    gradient_clip_norm: float = 0.5
    # Network initialization
    use_orthogonal_init: bool = True
    

class RNDNetwork(torch.nn.Module):
    """
    Random Network Distillation implementation for exploration bonus.
    
    RND provides intrinsic motivation by measuring prediction error:
    R_RND(s) = ||f_target(s) - f_predictor(s)||_2^2
    
    Where f_target is fixed random network and f_predictor is trainable.
    Paper formula: R_RND = ||f(s_next) - f_hat(s_next)||^2
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], epsilon: float = 1e-8):
        super().__init__()
        
        self.epsilon = epsilon  # For numerical stability
        
        # Fixed random target network - never updated (f in paper)
        self.target = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[1])
        )
        
        # Trainable predictor network - learns to match target (f_hat in paper)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[1])
        )
        
        # Initialize networks with proper weight initialization
        self._initialize_networks()
        
        # Freeze target network weights permanently
        for param in self.target.parameters():
            param.requires_grad = False
            
        # Running statistics for intrinsic reward normalization
        self.register_buffer('reward_mean', torch.zeros(1))
        self.register_buffer('reward_var', torch.ones(1))
        self.register_buffer('reward_count', torch.zeros(1))
        
        # Numerical monitoring
        self.register_buffer('nan_count', torch.zeros(1))
        self.register_buffer('inf_count', torch.zeros(1))
            
    def _initialize_networks(self):
        """
        Initialize networks using Xavier/Kaiming initialization as mentioned in paper.
        Uses orthogonal initialization for better training stability.
        """
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                # Orthogonal initialization for better gradient flow
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.constant_(m.bias, 0.0)
        
        # Initialize both networks with different random seeds
        self.target.apply(init_weights)
        
        # Re-initialize predictor with different weights
        torch.manual_seed(torch.initial_seed() + 1)
        self.predictor.apply(init_weights)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate RND prediction error as exploration bonus with numerical stability.
        
        Paper formula: R_RND = ||f(s) - f_hat(s)||^2
        
        Args:
            x: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Prediction error tensor of shape (batch_size,)
        """
        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            self.nan_count += 1
            print(f"Warning: NaN/Inf detected in RND input at step {self.nan_count}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        with torch.no_grad():
            target_feature = self.target(x)
            # Add small noise to prevent identical outputs
            target_feature = target_feature + torch.randn_like(target_feature) * self.epsilon
            
        predict_feature = self.predictor(x)
        
        # L2 squared error with numerical stability
        # Paper uses squared L2 norm: ||f(s) - f_hat(s)||^2
        diff = target_feature - predict_feature
        error = torch.sum(diff ** 2, dim=-1)
        
        # Clamp to prevent extreme values that could cause numerical instability
        error = torch.clamp(error, min=self.epsilon, max=1e6)
        
        # Check for NaN/Inf in output
        if torch.isnan(error).any() or torch.isinf(error).any():
            self.inf_count += 1
            print(f"Warning: NaN/Inf detected in RND output at step {self.inf_count}")
            error = torch.nan_to_num(error, nan=1.0, posinf=1e6, neginf=self.epsilon)
        
        return error

    def compute_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute RND predictor loss for training with numerical stability.
        
        Loss: L_RND = ||f_target(s) - f_predictor(s)||_2^2
        
        Args:
            states: Batch of states
            
        Returns:
            MSE loss between predictor and target outputs
        """
        # Check input validity
        if torch.isnan(states).any() or torch.isinf(states).any():
            states = torch.nan_to_num(states, nan=0.0, posinf=1e6, neginf=-1e6)
        
        with torch.no_grad():
            target_features = self.target(states)
            # Add small noise to target to prevent overfitting
            target_features = target_features + torch.randn_like(target_features) * self.epsilon
            
        predicted_features = self.predictor(states)
        
        # MSE loss with numerical stability
        loss = torch.nn.functional.mse_loss(predicted_features, target_features, reduction='mean')
        
        # Add small regularization term to prevent collapse
        reg_loss = self.epsilon * torch.mean(predicted_features ** 2)
        total_loss = loss + reg_loss
        
        # Ensure loss is finite
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN/Inf loss detected in RND, using fallback loss")
            total_loss = torch.tensor(1.0, device=states.device, requires_grad=True)
        
        return total_loss
    
    def update_reward_stats(self, rewards: torch.Tensor):
        """
        Update running statistics for intrinsic reward normalization.
        
        Uses Welford's online algorithm for numerical stability.
        """
        # Filter out invalid rewards
        valid_mask = torch.isfinite(rewards)
        if not valid_mask.any():
            return
            
        valid_rewards = rewards[valid_mask]
        batch_size = valid_rewards.size(0)
        
        if batch_size == 0:
            return
            
        batch_mean = valid_rewards.mean()
        batch_var = valid_rewards.var(unbiased=False)
        
        # Update count
        new_count = self.reward_count + batch_size
        
        if new_count == 0:
            return
        
        # Update mean using Welford's algorithm
        delta = batch_mean - self.reward_mean
        new_mean = self.reward_mean + delta * batch_size / new_count
        
        # Update variance using Welford's algorithm
        delta2 = batch_mean - new_mean
        new_var = (self.reward_var * self.reward_count + 
                  batch_var * batch_size + 
                  delta * delta2 * self.reward_count * batch_size / new_count) / new_count
        
        # Ensure variance is non-negative
        new_var = torch.clamp(new_var, min=self.epsilon)
        
        # Update buffers
        self.reward_mean.copy_(new_mean)
        self.reward_var.copy_(new_var)
        self.reward_count.copy_(new_count)
    
    def normalize_rewards(self, rewards: torch.Tensor, clip_value: float = 5.0) -> torch.Tensor:
        """
        Normalize intrinsic rewards using running statistics with numerical stability.
        
        Normalized reward: (r - μ) / σ
        """
        # Filter out invalid rewards
        valid_mask = torch.isfinite(rewards)
        if not valid_mask.any():
            return torch.zeros_like(rewards)
        
        if self.reward_count > 1:
            std = torch.sqrt(self.reward_var + self.epsilon)
            normalized = (rewards - self.reward_mean) / std
            # Clip to prevent extreme values
            normalized = torch.clamp(normalized, -clip_value, clip_value)
            
            # Handle invalid values
            normalized = torch.where(valid_mask, normalized, torch.zeros_like(normalized))
            return normalized
        else:
            return torch.clamp(rewards, -clip_value, clip_value)


class MixedDistribution:
    """
    Implements mixed initial state distribution from Algorithm 2.
    
    With probability p: sample critical state using mask network importance
    With probability 1-p: sample default initial state
    """
    
    def __init__(self, mix_prob: float = 0.25):
        self.mix_prob = mix_prob
        
    def sample_initial_state(self, env, pretrained_policy, mask_network, device):
        """
        Sample initial state from mixed distribution.
        
        Algorithm 2 Line 3: s_0 ~ mixed distribution
        """
        if np.random.random() < self.mix_prob:
            # Sample critical state using mask network
            return self._sample_critical_state(env, pretrained_policy, mask_network, device)
        else:
            # Sample default initial state
            return env.reset()
    
    def _sample_critical_state(self, env, pretrained_policy, mask_network, device):
        """
        Sample critical state based on mask network importance scores.
        
        1. Generate trajectory using pretrained policy
        2. Compute importance scores using mask network
        3. Sample state proportional to importance
        """
        # Generate trajectory using pretrained policy
        state = env.reset()
        trajectory_states = []
        done = False
        
        while not done and len(trajectory_states) < 1000:  # Prevent infinite loops
            trajectory_states.append(state.copy())
            
            # Get action from pretrained policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = pretrained_policy.predict(state_tensor, deterministic=False)[0]
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                
            # Take environment step
            next_state, _, done, _ = env.step(action)
            state = next_state
            
        if len(trajectory_states) == 0:
            return env.reset()
            
        # Convert trajectory to tensor
        trajectory_tensor = torch.FloatTensor(np.array(trajectory_states)).to(device)
        
        # Get importance scores from mask network
        with torch.no_grad():
            importance_scores = mask_network.get_importance_scores(trajectory_tensor)
            
        # Convert to probabilities with numerical stability
        if importance_scores.sum() > 1e-8:
            probs = importance_scores.cpu().numpy()
            probs = np.maximum(probs, 1e-8)  # Prevent zero probabilities
            probs = probs / probs.sum()  # Normalize to probability distribution
            
            # Sample state index based on importance
            selected_idx = np.random.choice(len(trajectory_states), p=probs)
            return trajectory_states[selected_idx]
        else:
            # Fallback to random state if all scores are zero
            selected_idx = np.random.choice(len(trajectory_states))
            return trajectory_states[selected_idx]


class PolicyRefiner:
    """
    RICE Policy Refiner implementation that uses explanation mask network
    and RND exploration to improve a pre-trained policy.
    
    Implements Algorithm 2: RICE Refining
    """
    
    def __init__(
        self,
        env,
        pretrained_policy: PPO,
        mask_network: MaskNetwork,
        config: RefinementConfig,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.env = env
        self.pretrained_policy = pretrained_policy
        self.mask_network = mask_network
        self.config = config
        self.device = device
        
        # Initialize refined policy as copy of pretrained (Algorithm 2 Line 1)
        self.refined_policy = PPO(
            env,
            learning_rate=config.ppo_learning_rate,
            n_steps=config.ppo_n_steps,
            batch_size=config.ppo_batch_size, 
            n_epochs=config.ppo_n_epochs,
            clip_range=config.ppo_clip_range,
            device=device
        )
        
        # Copy pretrained policy weights
        self.refined_policy.policy.load_state_dict(
            pretrained_policy.policy.state_dict()
        )
        
        # Initialize RND network for exploration bonus (Algorithm 2)
        state_dim = env.observation_space.shape[0]
        self.rnd = RNDNetwork(
            input_dim=state_dim,
            hidden_dims=config.rnd_hidden_dims,
            epsilon=config.epsilon
        ).to(device)
        
        # RND optimizer for predictor network only with gradient clipping
        self.rnd_optimizer = torch.optim.Adam(
            self.rnd.predictor.parameters(),
            lr=config.rnd_learning_rate,
            eps=config.epsilon  # Numerical stability for Adam
        )
        
        # Mixed distribution for initial state sampling
        self.mixed_distribution = MixedDistribution(config.mix_prob)
        
        # Rollout buffer for collecting experience
        self.rollout_buffer = []
        
        # Training step counter for RND update frequency
        self.training_step = 0
        
        # Numerical monitoring
        self.numerical_issues = 0

    def _collect_rollout(self, initial_state: np.ndarray) -> Dict[str, List]:
        """
        Collect rollout data starting from given initial state.
        
        Algorithm 2 Lines 4-8: Collect trajectory with augmented rewards
        """
        rollout_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "log_probs": [],
            "values": [],
            "intrinsic_rewards": [],
            "extrinsic_rewards": []
        }
        
        state = initial_state.copy()
        done = False
        step_count = 0
        
        while not done and step_count < self.config.max_episode_steps:
            # Convert state to tensor with numerical stability
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Check for invalid states
            if torch.isnan(state_tensor).any() or torch.isinf(state_tensor).any():
                print(f"Warning: Invalid state detected at step {step_count}")
                state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Get action, log_prob, and value from refined policy
            with torch.no_grad():
                action, log_prob = self.refined_policy.policy.get_action_and_log_prob(state_tensor)
                value = self.refined_policy.policy.get_value(state_tensor)
                
            # Convert action to numpy if needed
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy().flatten()
            else:
                action_np = action
                
            # Clip action to valid range to prevent environment errors
            if hasattr(self.env.action_space, 'low') and hasattr(self.env.action_space, 'high'):
                action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)
                
            # Take environment step
            try:
                next_state, extrinsic_reward, done, info = self.env.step(action_np)
            except Exception as e:
                print(f"Environment step failed: {e}")
                break
            
            # Calculate intrinsic reward using RND on next state (Paper formula)
            # R_RND = ||f(s_next) - f_hat(s_next)||^2
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                intrinsic_reward = self.rnd(next_state_tensor).item()
                
            # Ensure rewards are finite
            if not np.isfinite(extrinsic_reward):
                extrinsic_reward = 0.0
            if not np.isfinite(intrinsic_reward):
                intrinsic_reward = 0.0
                
            # Store raw intrinsic reward for statistics update
            rollout_data["intrinsic_rewards"].append(intrinsic_reward)
            rollout_data["extrinsic_rewards"].append(extrinsic_reward)
            
            # Augmented reward: R_aug = R_ext + λ * R_int (Algorithm 2 Line 7)
            augmented_reward = extrinsic_reward + self.config.exploration_lambda * intrinsic_reward
            
            # Store transition
            rollout_data["states"].append(state.copy())
            rollout_data["actions"].append(action_np)
            rollout_data["rewards"].append(augmented_reward)
            rollout_data["next_states"].append(next_state.copy())
            rollout_data["dones"].append(done)
            rollout_data["log_probs"].append(log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob)
            rollout_data["values"].append(value.item() if isinstance(value, torch.Tensor) else value)
            
            state = next_state
            step_count += 1
            
        return rollout_data

    def _update_networks(self, rollout_data: Dict[str, List]):
        """
        Update both policy and RND networks with numerical stability.
        
        Algorithm 2 Line 8: Update π and RND networks
        """
        if len(rollout_data["states"]) == 0:
            return {"policy_loss": 0.0, "rnd_loss": 0.0, "intrinsic_reward_mean": 0.0}
            
        # Convert rollout data to tensors with numerical checks
        states = torch.FloatTensor(np.array(rollout_data["states"])).to(self.device)
        next_states = torch.FloatTensor(np.array(rollout_data["next_states"])).to(self.device)
        actions = torch.FloatTensor(np.array(rollout_data["actions"])).to(self.device)
        rewards = torch.FloatTensor(rollout_data["rewards"]).to(self.device)
        dones = torch.BoolTensor(rollout_data["dones"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout_data["log_probs"]).to(self.device)
        old_values = torch.FloatTensor(rollout_data["values"]).to(self.device)
        intrinsic_rewards = torch.FloatTensor(rollout_data["intrinsic_rewards"]).to(self.device)
        
        # Check for numerical issues
        tensors_to_check = [states, next_states, actions, rewards, old_log_probs, old_values, intrinsic_rewards]
        for i, tensor in enumerate(tensors_to_check):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                self.numerical_issues += 1
                print(f"Warning: Numerical issue in tensor {i} at training step {self.training_step}")
                # Clean the tensor
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
                tensors_to_check[i] = tensor
        
        states, next_states, actions, rewards, old_log_probs, old_values, intrinsic_rewards = tensors_to_check
        
        # Update RND network statistics and normalize intrinsic rewards
        if self.config.intrinsic_reward_norm:
            self.rnd.update_reward_stats(intrinsic_rewards)
            normalized_intrinsic = self.rnd.normalize_rewards(
                intrinsic_rewards, self.config.intrinsic_reward_clip
            )
            # Recompute augmented rewards with normalized intrinsic rewards
            extrinsic_rewards = torch.FloatTensor(rollout_data["extrinsic_rewards"]).to(self.device)
            rewards = extrinsic_rewards + self.config.exploration_lambda * normalized_intrinsic
        
        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, old_values, dones)
        
        # Update policy using PPO
        policy_loss = self._update_policy(states, actions, old_log_probs, advantages, returns)
        
        # Update RND predictor network
        rnd_loss = 0.0
        if self.training_step % self.config.rnd_update_frequency == 0:
            rnd_loss = self._update_rnd(next_states)  # Use next states as per paper
        
        self.training_step += 1
        
        return {
            "policy_loss": policy_loss,
            "rnd_loss": rnd_loss,
            "intrinsic_reward_mean": intrinsic_rewards.mean().item(),
            "intrinsic_reward_std": intrinsic_rewards.std().item(),
            "numerical_issues": self.numerical_issues
        }

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, 
                     gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) with numerical stability.
        
        GAE(γ,λ): A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        next_value = 0  # Assume terminal value is 0
        
        # Compute GAE backwards through time
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_value = values[t + 1]
                
            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            
            # Ensure delta is finite
            if torch.isnan(delta) or torch.isinf(delta):
                delta = torch.tensor(0.0, device=rewards.device)
            
            # GAE: A_t = δ_t + γλA_{t+1}
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            
            # Ensure gae is finite
            if torch.isnan(gae) or torch.isinf(gae):
                gae = torch.tensor(0.0, device=rewards.device)
            
            advantages[t] = gae
            
            # Return: R_t = A_t + V(s_t)
            returns[t] = gae + values[t]
            
        return advantages, returns

    def _update_policy(self, states: torch.Tensor, actions: torch.Tensor, 
                      old_log_probs: torch.Tensor, advantages: torch.Tensor, 
                      returns: torch.Tensor) -> float:
        """
        Update policy using PPO clipped objective with entropy bonus.
        
        PPO Loss: L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)] + c_ent * H(π)
        where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        """
        # Normalize advantages with numerical stability
        if advantages.std() > self.config.epsilon:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.config.epsilon)
        else:
            advantages = advantages - advantages.mean()
        
        total_policy_loss = 0.0
        n_updates = 0
        
        # Multiple epochs of policy updates
        for epoch in range(self.config.ppo_n_epochs):
            # Mini-batch updates
            batch_size = self.config.ppo_batch_size
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy outputs
                _, new_log_probs = self.refined_policy.policy.get_action_and_log_prob(
                    batch_states, batch_actions
                )
                new_values = self.refined_policy.policy.get_value(batch_states).squeeze()
                
                # Get entropy for exploration bonus (paper mentions entropy coefficient)
                entropy = self.refined_policy.policy.get_entropy(batch_states)
                
                # Compute probability ratio with numerical stability
                log_ratio = new_log_probs - batch_old_log_probs
                log_ratio = torch.clamp(log_ratio, -20, 20)  # Prevent extreme ratios
                ratio = torch.exp(log_ratio)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.ppo_clip_range, 
                                  1.0 + self.config.ppo_clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = torch.nn.functional.mse_loss(new_values, batch_returns)
                
                # Entropy bonus for exploration (paper mentions entropy coefficient = 0.01)
                entropy_loss = -self.config.entropy_coef * entropy.mean()
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss + entropy_loss
                
                # Check for numerical issues
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Warning: Invalid loss detected, skipping update")
                    continue
                
                # Update policy with gradient clipping
                self.refined_policy.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.refined_policy.policy.parameters(), 
                    self.config.gradient_clip_norm
                )
                self.refined_policy.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                n_updates += 1
                
        return total_policy_loss / max(n_updates, 1)

    def _update_rnd(self, states: torch.Tensor) -> float:
        """
        Update RND predictor network to minimize prediction error with gradient clipping.
        
        RND Loss: L_RND = ||f_target(s) - f_predictor(s)||_2^2
        
        Args:
            states: Batch of states to train RND on
            
        Returns:
            Average RND loss over the batch
        """
        total_rnd_loss = 0.0
        n_updates = 0
        
        # Mini-batch updates for RND
        batch_size = min(self.config.ppo_batch_size, len(states))
        indices = torch.randperm(len(states))
        
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            batch_states = states[batch_indices]
            
            # Compute RND loss
            rnd_loss = self.rnd.compute_loss(batch_states)
            
            # Check for numerical issues
            if torch.isnan(rnd_loss) or torch.isinf(rnd_loss):
                print("Warning: Invalid RND loss detected, skipping update")
                continue
            
            # Update RND predictor with gradient clipping
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.rnd.predictor.parameters(), 
                self.config.gradient_clip_norm
            )
            self.rnd_optimizer.step()
            
            total_rnd_loss += rnd_loss.item()
            n_updates += 1
            
        return total_rnd_loss / max(n_updates, 1)

    def refine(self, num_steps: int = None) -> Dict[str, List[float]]:
        """
        Main refinement loop implementing Algorithm 2: RICE Refining.
        
        Algorithm 2:
        1. Initialize π_refined ← π_pretrained
        2. for step = 1 to num_steps do:
        3.   s_0 ~ mixed distribution
        4.   Collect trajectory τ = {s_0, a_0, r_0, ...}
        5.   for each (s_t, a_t, r_t) in τ do:
        6.     r_int ← RND(s_t)
        7.     r_aug ← r_t + λ * r_int
        8.   Update π and RND networks
        
        Returns:
            Dictionary containing training metrics
        """
        if num_steps is None:
            num_steps = self.config.n_iterations
            
        metrics = {
            "episode_rewards": [],
            "extrinsic_rewards": [],
            "intrinsic_rewards": [],
            "intrinsic_reward_mean": [],
            "intrinsic_reward_std": [],
            "policy_losses": [],
            "rnd_losses": [],
            "episode_lengths": [],
            "numerical_issues": []
        }
        
        print(f"Starting RICE policy refinement for {num_steps} steps...")
        print(f"Configuration: λ={self.config.exploration_lambda}, p={self.config.mix_prob}")
        print(f"Numerical stability: ε={self.config.epsilon}, grad_clip={self.config.gradient_clip_norm}")
        print(f"Entropy coefficient: {self.config.entropy_coef}")
        
        for step in range(num_steps):
            # Algorithm 2 Line 3: Sample initial state from mixed distribution
            try:
                initial_state = self.mixed_distribution.sample_initial_state(
                    self.env, self.pretrained_policy, self.mask_network, self.device
                )
            except Exception as e:
                print(f"Warning: Failed to sample initial state at step {step}: {e}")
                initial_state = self.env.reset()
            
            # Algorithm 2 Lines 4-7: Collect rollout with augmented rewards
            rollout_data = self._collect_rollout(initial_state)
            
            if len(rollout_data["states"]) == 0:
                continue
                
            # Algorithm 2 Line 8: Update networks
            update_metrics = self._update_networks(rollout_data)
            
            # Record metrics
            extrinsic_reward = sum(rollout_data["extrinsic_rewards"])
            intrinsic_reward = sum(rollout_data["intrinsic_rewards"])
            total_reward = sum(rollout_data["rewards"])
            
            metrics["episode_rewards"].append(total_reward)
            metrics["extrinsic_rewards"].append(extrinsic_reward)
            metrics["intrinsic_rewards"].append(intrinsic_reward)
            metrics["intrinsic_reward_mean"].append(update_metrics.get("intrinsic_reward_mean", 0.0))
            metrics["intrinsic_reward_std"].append(update_metrics.get("intrinsic_reward_std", 0.0))
            metrics["policy_losses"].append(update_metrics["policy_loss"])
            metrics["rnd_losses"].append(update_metrics["rnd_loss"])
            metrics["episode_lengths"].append(len(rollout_data["states"]))
            metrics["numerical_issues"].append(update_metrics.get("numerical_issues", 0))
            
            # Logging with numerical monitoring
            if step % 50 == 0:
                avg_extrinsic = np.mean(metrics["extrinsic_rewards"][-50:]) if metrics["extrinsic_rewards"] else 0
                avg_intrinsic = np.mean(metrics["intrinsic_rewards"][-50:]) if metrics["intrinsic_rewards"] else 0
                avg_total = np.mean(metrics["episode_rewards"][-50:]) if metrics["episode_rewards"] else 0
                avg_rnd_loss = np.mean(metrics["rnd_losses"][-50:]) if metrics["rnd_losses"] else 0
                total_numerical_issues = sum(metrics["numerical_issues"])
                
                print(f"Step {step:4d}: "
                      f"Ext={avg_extrinsic:6.2f}, "
                      f"Int={avg_intrinsic:6.2f}, "
                      f"Total={avg_total:6.2f}, "
                      f"RND_Loss={avg_rnd_loss:.4f}, "
                      f"NumIssues={total_numerical_issues}")
                
        print("RICE policy refinement completed!")
        print(f"Total numerical issues encountered: {sum(metrics['numerical_issues'])}")
        return metrics
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate the refined policy performance.
        
        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < self.config.max_episode_steps:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Check for invalid states
                if torch.isnan(state_tensor).any() or torch.isinf(state_tensor).any():
                    state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
                
                with torch.no_grad():
                    action = self.refined_policy.predict(state_tensor, deterministic=deterministic)[0]
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                
                # Clip action to valid range
                if hasattr(self.env.action_space, 'low') and hasattr(self.env.action_space, 'high'):
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                
                try:
                    next_state, reward, done, _ = self.env.step(action)
                    if np.isfinite(reward):
                        episode_reward += reward
                    episode_length += 1
                    state = next_state
                except Exception as e:
                    print(f"Evaluation step failed: {e}")
                    break
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards)
        }
    
    def save(self, path: str):
        """Save refined policy and RND networks"""
        torch.save({
            "policy_state_dict": self.refined_policy.policy.state_dict(),
            "rnd_predictor_state_dict": self.rnd.predictor.state_dict(),
            "rnd_target_state_dict": self.rnd.target.state_dict(),
            "rnd_reward_mean": self.rnd.reward_mean,
            "rnd_reward_var": self.rnd.reward_var,
            "rnd_reward_count": self.rnd.reward_count,
            "config": self.config,
            "training_step": self.training_step,
            "numerical_issues": self.numerical_issues
        }, path)
        
    def load(self, path: str):
        """Load refined policy and RND networks"""
        checkpoint = torch.load(path, map_location=self.device)
        self.refined_policy.policy.load_state_dict(
            checkpoint["policy_state_dict"]
        )
        self.rnd.predictor.load_state_dict(
            checkpoint["rnd_predictor_state_dict"]
        )
        self.rnd.target.load_state_dict(
            checkpoint["rnd_target_state_dict"]
        )
        
        # Restore RND statistics
        if "rnd_reward_mean" in checkpoint:
            self.rnd.reward_mean.copy_(checkpoint["rnd_reward_mean"])
            self.rnd.reward_var.copy_(checkpoint["rnd_reward_var"])
            self.rnd.reward_count.copy_(checkpoint["rnd_reward_count"])
            
        if "training_step" in checkpoint:
            self.training_step = checkpoint["training_step"]
            
        if "numerical_issues" in checkpoint:
            self.numerical_issues = checkpoint["numerical_issues"]