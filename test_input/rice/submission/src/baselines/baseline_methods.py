```python
"""
Baseline Methods Implementation for RICE Paper Reproduction

This module implements all baseline methods used in the RICE paper for fair comparison:
1. PPO Fine-tuning: Continue training with reduced learning rate
2. StateMask-R: Equivalent to RICE with p=1, λ=0
3. Random Explanation: Random selection of critical states

All baselines use identical experimental settings for fair comparison.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BaselineConfig:
    """Configuration for baseline experiments"""
    # Training parameters
    total_timesteps: int = 100000
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    
    # PPO Fine-tuning specific
    ppo_learning_rate_reduction: float = 0.1  # Reduce LR by 10x
    ppo_original_lr: float = 3e-4
    
    # StateMask-R specific (RICE with p=1, λ=0)
    statemask_r_p: float = 1.0
    statemask_r_lambda: float = 0.0
    
    # Random explanation specific
    random_seed: int = 42
    explanation_ratio: float = 0.1  # 10% of states as critical
    
    # Common parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "./baseline_results/"

class BaselineMethod(ABC):
    """Abstract base class for all baseline methods"""
    
    def __init__(self, config: BaselineConfig, env: gym.Env, pretrained_model_path: str):
        self.config = config
        self.env = env
        self.pretrained_model_path = pretrained_model_path
        self.results = {}
        
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Train the baseline method"""
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the baseline method"""
        pass
    
    def save_results(self, filepath: str):
        """Save results to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

class PPOFineTuningBaseline(BaselineMethod):
    """
    PPO Fine-tuning Baseline
    
    Continues training the pretrained PPO model with reduced learning rate.
    This serves as a simple baseline to show that just continuing training
    doesn't achieve the same improvements as RICE.
    """
    
    def __init__(self, config: BaselineConfig, env: gym.Env, pretrained_model_path: str):
        super().__init__(config, env, pretrained_model_path)
        self.model = None
        
    def train(self) -> Dict[str, Any]:
        """
        Fine-tune PPO model with reduced learning rate
        
        Algorithm:
        1. Load pretrained PPO model
        2. Reduce learning rate by specified factor
        3. Continue training for specified timesteps
        4. Track performance during training
        """
        logger.info("Starting PPO Fine-tuning Baseline Training")
        
        # Load pretrained model
        self.model = PPO.load(self.pretrained_model_path, env=self.env)
        
        # Reduce learning rate
        new_lr = self.config.ppo_original_lr * self.config.ppo_learning_rate_reduction
        self.model.learning_rate = new_lr
        
        # Update optimizer learning rate
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        logger.info(f"Reduced learning rate from {self.config.ppo_original_lr} to {new_lr}")
        
        # Training callback to track progress
        class TrainingCallback(BaseCallback):
            def __init__(self, baseline_instance):
                super().__init__()
                self.baseline = baseline_instance
                self.episode_rewards = []
                
            def _on_step(self) -> bool:
                if len(self.locals.get('infos', [])) > 0:
                    for info in self.locals['infos']:
                        if 'episode' in info:
                            self.baseline.results.setdefault('training_rewards', []).append(
                                info['episode']['r']
                            )
                return True
        
        callback = TrainingCallback(self)
        
        # Continue training
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        logger.info("PPO Fine-tuning training completed")
        
        return {
            'method': 'PPO Fine-tuning',
            'final_lr': new_lr,
            'training_timesteps': self.config.total_timesteps
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the fine-tuned PPO model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        logger.info("Evaluating PPO Fine-tuning Baseline")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(self.config.n_eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        self.results['evaluation'] = results
        logger.info(f"PPO Fine-tuning Results: {results}")
        
        return results

class StateMaskRBaseline(BaselineMethod):
    """
    StateMask-R Baseline
    
    Implements StateMask-R which is equivalent to RICE with p=1 and λ=0.
    This means all states are considered critical (p=1) and no regularization (λ=0).
    """
    
    def __init__(self, config: BaselineConfig, env: gym.Env, pretrained_model_path: str):
        super().__init__(config, env, pretrained_model_path)
        self.model = None
        self.state_mask = None
        
    def _generate_state_mask(self, state_dim: int) -> torch.Tensor:
        """
        Generate state mask for StateMask-R
        
        With p=1, all state dimensions are considered critical
        """
        # p=1 means all dimensions are critical (mask = 1 for all dimensions)
        mask = torch.ones(state_dim, dtype=torch.float32, device=self.config.device)
        return mask
    
    def _apply_state_mask(self, state: torch.Tensor) -> torch.Tensor:
        """Apply state mask to input state"""
        if self.state_mask is None:
            self.state_mask = self._generate_state_mask(state.shape[-1])
        
        # Element-wise multiplication with mask
        masked_state = state * self.state_mask
        return masked_state
    
    def train(self) -> Dict[str, Any]:
        """
        Train StateMask-R baseline
        
        Algorithm (StateMask-R ≡ RICE with p=1, λ=0):
        1. Load pretrained model
        2. Generate state mask (all 1s for p=1)
        3. Fine-tune with masked states
        4. No regularization term (λ=0)
        """
        logger.info("Starting StateMask-R Baseline Training")
        
        # Load pretrained model
        self.model = PPO.load(self.pretrained_model_path, env=self.env)
        
        # Get state dimension from environment
        state_dim = self.env.observation_space.shape[0]
        self.state_mask = self._generate_state_mask(state_dim)
        
        logger.info(f"Generated state mask with p={self.config.statemask_r_p} (all critical)")
        
        # Custom training loop with state masking
        class MaskedTrainingCallback(BaseCallback):
            def __init__(self, baseline_instance):
                super().__init__()
                self.baseline = baseline_instance
                
            def _on_step(self) -> bool:
                # Apply state masking during training
                if hasattr(self.training_env, 'buf_obs'):
                    obs = torch.tensor(self.training_env.buf_obs, device=self.baseline.config.device)
                    masked_obs = self.baseline._apply_state_mask(obs)
                    self.training_env.buf_obs = masked_obs.cpu().numpy()
                return True
        
        callback = MaskedTrainingCallback(self)
        
        # Continue training with masked states
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        logger.info("StateMask-R training completed")
        
        return {
            'method': 'StateMask-R',
            'p_value': self.config.statemask_r_p,
            'lambda_value': self.config.statemask_r_lambda,
            'mask_sum': float(self.state_mask.sum()),
            'training_timesteps': self.config.total_timesteps
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate StateMask-R model with state masking"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        logger.info("Evaluating StateMask-R Baseline")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(self.config.n_eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Apply state mask during evaluation
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.config.device)
                masked_obs = self._apply_state_mask(obs_tensor)
                
                action, _ = self.model.predict(masked_obs.cpu().numpy(), deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        self.results['evaluation'] = results
        logger.info(f"StateMask-R Results: {results}")
        
        return results

class RandomExplanationBaseline(BaselineMethod):
    """
    Random Explanation Baseline
    
    Randomly selects critical states instead of using the RICE explanation method.
    This tests whether the specific explanation method in RICE is important
    or if random selection would work just as well.
    """
    
    def __init__(self, config: BaselineConfig, env: gym.Env, pretrained_model_path: str):
        super().__init__(config, env, pretrained_model_path)
        self.model = None
        self.random_mask = None
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        
    def _generate_random_mask(self, state_dim: int) -> torch.Tensor:
        """
        Generate random state mask
        
        Randomly selects explanation_ratio of state dimensions as critical
        """
        n_critical = max(1, int(state_dim * self.config.explanation_ratio))
        
        # Create mask with zeros
        mask = torch.zeros(state_dim, dtype=torch.float32, device=self.config.device)
        
        # Randomly select critical dimensions
        critical_indices = np.random.choice(state_dim, size=n_critical, replace=False)
        mask[critical_indices] = 1.0
        
        logger.info(f"Random mask: {n_critical}/{state_dim} dimensions critical")
        return mask
    
    def _apply_random_mask(self, state: torch.Tensor) -> torch.Tensor:
        """Apply random mask to input state"""
        if self.random_mask is None:
            self.random_mask = self._generate_random_mask(state.shape[-1])
        
        # Apply mask - zero out non-critical dimensions
        masked_state = state * self.random_mask
        return masked_state
    
    def train(self) -> Dict[str, Any]:
        """
        Train with random explanation baseline
        
        Algorithm:
        1. Load pretrained model
        2. Generate random state mask
        3. Fine-tune with randomly masked states
        4. Compare against RICE's learned explanations
        """
        logger.info("Starting Random Explanation Baseline Training")
        
        # Load pretrained model
        self.model = PPO.load(self.pretrained_model_path, env=self.env)
        
        # Get state dimension and generate random mask
        state_dim = self.env.observation_space.shape[0]
        self.random_mask = self._generate_random_mask(state_dim)
        
        logger.info(f"Generated random mask with {self.random_mask.sum().item()}/{state_dim} critical dimensions")
        
        # Custom training with random masking
        class RandomMaskCallback(BaseCallback):
            def __init__(self, baseline_instance):
                super().__init__()
                self.baseline = baseline_instance
                
            def _on_step(self) -> bool:
                # Apply random masking during training
                if hasattr(self.training_env, 'buf_obs'):
                    obs = torch.tensor(self.training_env.buf_obs, device=self.baseline.config.device)
                    masked_obs = self.baseline._apply_random_mask(obs)
                    self.training_env.buf_obs = masked_obs.cpu().numpy()
                return True
        
        callback = RandomMaskCallback(self)
        
        # Train with random masking
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        logger.info("Random Explanation training completed")
        
        return {
            'method': 'Random Explanation',
            'explanation_ratio': self.config.explanation_ratio,
            'critical_dimensions': int(self.random_mask.sum().item()),
            'total_dimensions': len(self.random_mask),
            'random_seed': self.config.random_seed,
            'training_timesteps': self.config.total_timesteps
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate random explanation model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        logger.info("Evaluating Random Explanation Baseline")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(self.config.n_eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Apply random mask during evaluation
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.config.device)
                masked_obs = self._apply_random_mask(obs_tensor)
                
                action, _ = self.model.predict(masked_obs.cpu().numpy(), deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        self.results['evaluation'] = results
        logger.info(f"Random Explanation Results: {results}")
        
        return results

class BaselineExperimentRunner:
    """
    Experiment runner for all baseline methods
    
    Ensures fair comparison by using identical experimental settings
    """
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.results = {}
        
    def run_all_baselines(self, env: gym.Env, pretrained_model_path: str) -> Dict[str, Dict]:
        """
        Run all baseline experiments with identical settings
        
        Returns:
            Dictionary containing results from all baseline methods
        """
        logger.info("Starting comprehensive baseline evaluation")
        
        baselines = {
            'ppo_finetuning': PPOFineTuningBaseline(self.config, env, pretrained_model_path),
            'statemask_r': StateMaskRBaseline(self.config, env, pretrained_model_path),
            'random_explanation': RandomExplanationBaseline(self.config, env, pretrained_model_path)
        }
        
        for name, baseline in baselines.items():
            logger.info(f"Running {name} baseline")
            
            try:
                # Train baseline
                training_info = baseline.train()
                
                # Evaluate baseline
                eval_results = baseline.evaluate()
                
                # Store results
                self.results[name] = {
                    'training_info': training_info,
                    'evaluation_results': eval_results,
                    'config': self.config.__dict__
                }
                
                # Save individual results
                baseline.save_results(f"{self.config.save_path}/{name}_results.json")
                
                logger.info(f"Completed {name} baseline")
                
            except Exception as e:
                logger.error(f"Error running {name} baseline: {str(e)}")
                self.results[name] = {'error': str(e)}
        
        # Save comprehensive results
        self._save_comprehensive_results()
        
        return self.results
    
    def _save_comprehensive_results(self):
        """Save comprehensive comparison results"""
        import json
        import os
        
        os.makedirs(self.config.save_path, exist_ok=True)
        
        # Create comparison summary
        comparison = {
            'experiment_config': self.config.__dict__,
            'baseline_results': self.results,
            'summary': self._generate_summary()
        }
        
        with open(f"{self.config.save_path}/baseline_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
            
        logger.info(f"Comprehensive results saved to {self.config.save_path}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for comparison"""
        summary = {}
        
        for baseline_name, results in self.results.items():
            if 'evaluation_results' in results:
                eval_results = results['evaluation_results']
                summary[baseline_name] = {
                    'mean_reward': eval_results.get('mean_reward', 0),
                    'std_reward': eval_results.get('std_reward', 0),
                    'success_rate': 1.0 if eval_results.get('mean_reward', 0) > 0 else 0.0
                }
        
        # Rank baselines by performance
        if summary:
            ranked = sorted(summary.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
            summary['ranking'] = [name for name, _ in ranked]
        
        return summary

def create_baseline_experiment(env_name: str, pretrained_model_path: str, 
                             config: Optional[BaselineConfig] = None) -> BaselineExperimentRunner:
    """
    Factory function to create baseline experiment
    
    Args:
        env_name: Name of the environment
        pretrained_model_path: Path to pretrained model
        config: Optional configuration (uses default if None)
        
    Returns:
        Configured experiment runner
    """
    if config is None:
        config = BaselineConfig()
    
    return BaselineExperimentRunner(config)

# Example usage and testing
if __name__ == "__main__":
    # Example configuration for testing
    config = BaselineConfig(
        total_timesteps=50000,  # Reduced for testing
        eval_freq=5000,
        n_eval_episodes=5,
        save_path="./test_baseline_results/"
    )
    
    # This would be used in actual experiments
    # env = gym.make("HalfCheetah-v4")
    # pretrained_path = "path/to/pretrained/model.zip"
    # 
    # runner = BaselineExperimentRunner(config)
    # results = runner.run_all_baselines(env, pretrained_path)
    # 
    # print("Baseline comparison completed!")
    # print(f"Results saved to {config.save_path}")
    
    logger.info("Baseline methods implementation completed")
```