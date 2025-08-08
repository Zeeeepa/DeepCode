"""
RICE Policy Refiner implementation that integrates mask network explanations with RND exploration.
Implements Algorithm 2: RICE Refining from the paper.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import logging

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
    # Rollout collection parameters
    max_episode_steps: int = 1000
    rollout_buffer_size: int = 2048
    

class RNDNetwork(torch.nn.Module):
    """
    Random Network Distillation implementation for exploration bonus.
    Implements the RND exploration reward: R_RND = ||f(s) - f̂(s)||²
    where f is the fixed target network and f̂ is the trainable predictor.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # Fixed random target network f(s) - never updated
        self.target = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[1])
        )
        
        # Trainable predictor network f̂(s) - learns to predict target
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[1])
        )
        
        # Freeze target network weights
        for param in self.target.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate RND prediction error/exploration bonus.
        Returns ||f(s) - f̂(s)||² for each state in the batch.
        """
        with torch.no_grad():
            target_feature = self.target(x)
        predict_feature = self.predictor(x)
        
        # L2 error between prediction and target (exploration bonus)
        error = torch.norm(target_feature - predict_feature, dim=-1, p=2)
        return error

    def get_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Calculate MSE loss for updating predictor network"""
        with torch.no_grad():
            target_feature = self.target(states)
        predict_feature = self.predictor(states)
        
        # MSE loss for predictor training
        loss = torch.nn.functional.mse_loss(predict_feature, target_feature)
        return loss


class MixedInitialDistribution:
    """
    Implements the mixed initial state distribution from Algorithm 2.
    Samples from critical states (based on mask network) with probability p,
    or from default initial states with probability (1-p).
    """
    
    def __init__(self, mix_prob: float = 0.25):
        self.mix_prob = mix_prob
        
    def sample_initial_state(self, env, pretrained_policy, mask_network) -> np.ndarray:
        """
        Sample initial state from mixed distribution:
        - With probability p: sample from critical states identified by mask network
        - With probability (1-p): sample from default initial state distribution
        """
        if np.random.random() < self.mix_prob:
            return self._sample_critical_state(env, pretrained_policy, mask_network)
        else:
            return env.reset()
    
    def _sample_critical_state(self, env, pretrained_policy, mask_network) -> np.ndarray:
        """
        Sample a critical state by:
        1. Collecting a trajectory from pretrained policy
        2. Computing importance scores using mask network
        3. Sampling state proportional to importance scores
        """
        # Collect trajectory from pretrained policy
        state = env.reset()
        trajectory_states = []
        done = False
        
        while not done and len(trajectory_states) < 1000:  # Prevent infinite loops
            trajectory_states.append(state.copy())
            action = pretrained_policy.predict(state)[0]
            state, _, done, _ = env.step(action)
            
        if len(trajectory_states) == 0:
            return env.reset()
            
        # Convert to tensor for mask network
        states_tensor = torch.FloatTensor(np.array(trajectory_states))
        if torch.cuda.is_available():
            states_tensor = states_tensor.cuda()
            
        # Get importance scores from mask network
        with torch.no_grad():
            importance_scores = mask_network.get_mask(states_tensor)
            
        # Convert to numpy and normalize to probabilities
        scores = importance_scores.cpu().numpy()
        if scores.sum() > 0:
            probs = scores / scores.sum()
        else:
            probs = np.ones(len(scores)) / len(scores)
            
        # Sample state index based on importance scores
        idx = np.random.choice(len(trajectory_states), p=probs)
        return trajectory_states[idx]


class PolicyRefiner:
    """
    RICE Policy Refiner implementation that uses explanation mask network
    and RND exploration to improve a pre-trained policy.
    
    Implements Algorithm 2: RICE Refining from the paper.
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
        
        # Initialize refined policy as copy of pretrained
        self.refined_policy = PPO(
            env,
            learning_rate=config.ppo_learning_rate,
            n_steps=config.ppo_n_steps,
            batch_size=config.ppo_batch_size, 
            n_epochs=config.ppo_n_epochs,
            clip_range=config.ppo_clip_range,
            device=device
        )
        self.refined_policy.policy.load_state_dict(
            pretrained_policy.policy.state_dict()
        )
        
        # Initialize RND network for exploration
        state_dim = env.observation_space.shape[0]
        self.rnd = RNDNetwork(
            input_dim=state_dim,
            hidden_dims=config.rnd_hidden_dims
        ).to(device)
        
        # Optimizer for RND predictor network
        self.rnd_optimizer = torch.optim.Adam(
            self.rnd.predictor.parameters(),
            lr=config.ppo_learning_rate
        )
        
        # Mixed initial distribution sampler
        self.mixed_distribution = MixedInitialDistribution(config.mix_prob)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _collect_rollout(self, initial_state: np.ndarray) -> Dict[str, List]:
        """
        Collect rollout data starting from given initial state.
        Computes augmented rewards: R'_t = R_t + λ * R_RND
        """
        rollout_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'augmented_rewards': [],
            'next_states': [],
            'dones': [],
            'rnd_bonuses': []
        }
        
        # Reset environment to initial state
        state = initial_state.copy()
        done = False
        step_count = 0
        
        while not done and step_count < self.config.max_episode_steps:
            # Get action from current refined policy
            action = self.refined_policy.predict(state)[0]
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Calculate RND exploration bonus
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                rnd_bonus = self.rnd(state_tensor).item()
            
            # Compute augmented reward: R'_t = R_t + λ * R_RND
            augmented_reward = reward + self.config.exploration_lambda * rnd_bonus
            
            # Store transition
            rollout_data['states'].append(state.copy())
            rollout_data['actions'].append(action)
            rollout_data['rewards'].append(reward)
            rollout_data['augmented_rewards'].append(augmented_reward)
            rollout_data['next_states'].append(next_state.copy())
            rollout_data['dones'].append(done)
            rollout_data['rnd_bonuses'].append(rnd_bonus)
            
            state = next_state
            step_count += 1
            
        return rollout_data

    def _update_policy(self, rollout_data: Dict[str, List]):
        """Update refined policy using PPO with augmented rewards"""
        # Convert rollout data to tensors
        states = torch.FloatTensor(np.array(rollout_data['states'])).to(self.device)
        actions = torch.LongTensor(rollout_data['actions']).to(self.device)
        augmented_rewards = torch.FloatTensor(rollout_data['augmented_rewards']).to(self.device)
        next_states = torch.FloatTensor(np.array(rollout_data['next_states'])).to(self.device)
        dones = torch.BoolTensor(rollout_data['dones']).to(self.device)
        
        # Store transitions in PPO buffer using augmented rewards
        for i in range(len(states)):
            self.refined_policy.store_transition(
                rollout_data['states'][i],
                rollout_data['actions'][i], 
                rollout_data['augmented_rewards'][i],
                rollout_data['next_states'][i],
                rollout_data['dones'][i]
            )
        
        # Update policy using PPO
        self.refined_policy.update()

    def _update_rnd_network(self, rollout_data: Dict[str, List]) -> float:
        """Update RND predictor network to minimize prediction error"""
        states = torch.FloatTensor(np.array(rollout_data['states'])).to(self.device)
        
        # Calculate RND loss
        rnd_loss = self.rnd.get_loss(states)
        
        # Update RND predictor
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()
        
        return rnd_loss.item()

    def _log_progress(self, step_count: int, rollout_data: Dict[str, List]):
        """Log training progress"""
        episode_reward = sum(rollout_data['rewards'])
        avg_rnd_bonus = np.mean(rollout_data['rnd_bonuses'])
        
        self.logger.info(
            f"Step {step_count}: Episode Reward = {episode_reward:.2f}, "
            f"Avg RND Bonus = {avg_rnd_bonus:.4f}"
        )

    def refine_policy(self, pretrained_policy, env, num_steps: int = 100000) -> Dict[str, List[float]]:
        """
        Main policy refinement algorithm (Algorithm 2 from paper).
        
        Implements the complete RICE refinement process:
        1. Initialize policy as copy of pretrained policy
        2. For each iteration:
           a. Sample initial state from mixed distribution
           b. Collect rollout with augmented rewards
           c. Update policy using PPO
           d. Update RND predictor network
        
        Args:
            pretrained_policy: Pre-trained policy to refine
            env: Environment for training
            num_steps: Total number of environment steps for training
            
        Returns:
            Dictionary containing training metrics
        """
        # Initialize policy as copy of pretrained policy
        self.refined_policy.policy.load_state_dict(pretrained_policy.policy.state_dict())
        
        metrics = {
            "episode_rewards": [],
            "rnd_losses": [],
            "exploration_bonuses": [],
            "step_counts": []
        }
        
        step_count = 0
        episode_count = 0
        
        self.logger.info(f"Starting RICE policy refinement for {num_steps} steps")
        
        while step_count < num_steps:
            # Reset environment using mixed initial distribution (Algorithm 2, line 4)
            initial_state = self.mixed_distribution.sample_initial_state(
                env, pretrained_policy, self.mask_network
            )
            
            # Collect rollout data with augmented rewards (Algorithm 2, lines 5-8)
            rollout_data = self._collect_rollout(initial_state)
            
            # Update policy using PPO with augmented rewards (Algorithm 2, line 9)
            self._update_policy(rollout_data)
            
            # Update RND predictor network (Algorithm 2, line 10)
            rnd_loss = self._update_rnd_network(rollout_data)
            
            # Update step count and metrics
            episode_steps = len(rollout_data['states'])
            step_count += episode_steps
            episode_count += 1
            
            # Record metrics
            episode_reward = sum(rollout_data['rewards'])
            avg_exploration_bonus = np.mean(rollout_data['rnd_bonuses'])
            
            metrics["episode_rewards"].append(episode_reward)
            metrics["rnd_losses"].append(rnd_loss)
            metrics["exploration_bonuses"].append(avg_exploration_bonus)
            metrics["step_counts"].append(step_count)
            
            # Log progress periodically
            if step_count % 10000 == 0 or step_count >= num_steps:
                self._log_progress(step_count, rollout_data)
                
        self.logger.info(
            f"RICE refinement completed. Total episodes: {episode_count}, "
            f"Final step count: {step_count}"
        )
        
        return metrics

    def refine(self) -> Dict[str, List[float]]:
        """
        Legacy interface for backward compatibility.
        Runs RICE refinement process using configured number of iterations.
        """
        return self.refine_policy(
            self.pretrained_policy, 
            self.env, 
            num_steps=self.config.n_iterations * self.config.max_episode_steps
        )

    def _get_mixed_initial_state(self) -> np.ndarray:
        """
        Legacy method for backward compatibility.
        Get initial state using mixed distribution of default initial states
        and critical states based on explanation mask.
        """
        return self.mixed_distribution.sample_initial_state(
            self.env, self.pretrained_policy, self.mask_network
        )
    
    def save(self, path: str):
        """Save refined policy and RND networks"""
        torch.save({
            "policy_state_dict": self.refined_policy.policy.state_dict(),
            "rnd_predictor_state_dict": self.rnd.predictor.state_dict(),
            "rnd_target_state_dict": self.rnd.target.state_dict(),
            "config": self.config
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