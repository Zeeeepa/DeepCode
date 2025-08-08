"""
Configuration file for StateMask paper reproduction.
Contains all hyperparameters and environment configurations as specified in Appendix C.3 Table 3.
Implements L4-level reproduction targeting numerical alignment with paper results.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import os

class Config:
    """
    Central configuration class for StateMask reproduction.
    Based on paper's Appendix C.3 Table 3 and experimental settings.
    
    This configuration implements L4-level reproduction requirements:
    - Exact hyperparameter alignment with paper specifications
    - Environment-specific parameter tuning for optimal results
    - Network architectures matching Stable Baselines3 defaults as used in paper
    - Precise numerical settings for reproducible results
    """
    
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Random seeds for reproducibility - critical for L4 reproduction
        self.seed = 42
        self.torch_seed = 42
        self.numpy_seed = 42
        
        # Environment configurations - EXACT values from Appendix C.3 Table 3
        # These parameters are critical for achieving paper-level performance
        self.environments = {
            'Hopper-v3': {
                'mixture_prob': 0.25,  # p in paper - probability of using critical states
                'rnd_coeff': 0.001,    # λ for RND exploration reward
                'mask_reward_bias': 0.0001,  # α for mask network reward bias
                'max_episode_steps': 1000,
                'network_arch': [64, 64],  # 2-layer 64-node MLP for MuJoCo (SB3 default)
                'algorithm': 'sac',  # Continuous control uses SAC
                'env_type': 'mujoco',
            },
            'Walker2d-v3': {
                'mixture_prob': 0.25,  # p = 0.25 for Walker2d
                'rnd_coeff': 0.01,     # λ = 0.01 for Walker2d (higher exploration)
                'mask_reward_bias': 0.0001,  # α = 0.0001 consistent across MuJoCo
                'max_episode_steps': 1000,
                'network_arch': [64, 64],
                'algorithm': 'sac',
                'env_type': 'mujoco',
            },
            'Reacher-v2': {
                'mixture_prob': 0.50,  # p = 0.50 for Reacher (higher critical state usage)
                'rnd_coeff': 0.001,    # λ = 0.001 for Reacher
                'mask_reward_bias': 0.0001,  # α = 0.0001 consistent
                'max_episode_steps': 50,   # Shorter episodes for Reacher
                'network_arch': [64, 64],
                'algorithm': 'sac',
                'env_type': 'mujoco',
            },
            'HalfCheetah-v3': {
                'mixture_prob': 0.50,  # p = 0.50 for HalfCheetah
                'rnd_coeff': 0.01,     # λ = 0.01 for HalfCheetah (higher exploration)
                'mask_reward_bias': 0.0001,  # α = 0.0001 consistent
                'max_episode_steps': 1000,
                'network_arch': [64, 64],
                'algorithm': 'sac',
                'env_type': 'mujoco',
            },
            'SelfishMining': {
                'mixture_prob': 0.25,  # Default for custom environment
                'rnd_coeff': 0.001,    # Conservative exploration for discrete env
                'mask_reward_bias': 0.0001,
                'max_episode_steps': 200,
                'network_arch': [128, 128, 128, 128],  # 4-layer 128-node MLP for complex discrete env
                'algorithm': 'ppo',  # Discrete environment uses PPO
                'env_type': 'discrete',
            }
        }
        
        # PPO Hyperparameters - optimized for L4 reproduction
        # Based on SB3 defaults with paper-specific adjustments
        self.ppo_config = {
            'learning_rate': 3e-4,      # Standard learning rate
            'n_steps': 2048,            # Steps per rollout
            'batch_size': 64,           # Batch size for updates
            'n_epochs': 10,             # Optimization epochs per rollout
            'gamma': 0.99,              # Discount factor
            'gae_lambda': 0.95,         # GAE lambda parameter
            'clip_range': 0.2,          # PPO clipping parameter
            'clip_range_vf': None,      # Value function clipping (disabled)
            'ent_coef': 0.0,            # Entropy coefficient (disabled for deterministic)
            'vf_coef': 0.5,             # Value function coefficient
            'max_grad_norm': 0.5,       # Gradient clipping
            'use_sde': False,           # State-dependent exploration (disabled)
            'sde_sample_freq': -1,
            'target_kl': None,          # Target KL divergence (disabled)
            'policy_kwargs': {
                'activation_fn': torch.nn.ReLU,
                'net_arch': None,  # Will be set per environment
            }
        }
        
        # SAC Hyperparameters - optimized for continuous control reproduction
        # Critical parameters for achieving paper-level performance in MuJoCo
        self.sac_config = {
            'learning_rate': 3e-4,      # Learning rate for all networks
            'buffer_size': 1000000,     # Replay buffer size
            'learning_starts': 100,     # Steps before learning starts
            'batch_size': 256,          # Batch size for updates
            'tau': 0.005,               # Soft update coefficient
            'gamma': 0.99,              # Discount factor
            'train_freq': 1,            # Training frequency
            'gradient_steps': 1,        # Gradient steps per update
            'ent_coef': 'auto',         # Automatic entropy coefficient tuning
            'target_update_interval': 1, # Target network update frequency
            'target_entropy': 'auto',   # Automatic target entropy
            'use_sde': False,           # State-dependent exploration
            'sde_sample_freq': -1,
            'policy_kwargs': {
                'activation_fn': torch.nn.ReLU,
                'net_arch': None,  # Will be set per environment
            }
        }
        
        # Network Architecture Configuration - matching SB3 MlpPolicy defaults
        # As specified in supplementary information about SB3 usage
        self.network_config = {
            'activation_fn': torch.nn.ReLU,  # ReLU activation as in SB3 default
            'net_arch': None,  # Will be set per environment
            'features_extractor_class': None,
            'features_extractor_kwargs': None,
            'normalize_images': True,
            'optimizer_class': torch.optim.Adam,
            'optimizer_kwargs': {},
            'squash_output': False,  # For SAC continuous actions
        }
        
        # StateMask specific parameters - tuned for L4 reproduction
        self.statemask_config = {
            'mask_threshold': 0.5,      # Threshold for binary mask generation
            'mask_learning_rate': 1e-3, # Learning rate for mask network
            'mask_epochs': 100,         # Training epochs for mask network
            'mask_batch_size': 32,      # Batch size for mask training
            'explanation_samples': 1000, # Number of samples for explanation generation
            'refinement_iterations': 10, # Iterations for StateMask-R refinement
            'importance_threshold': 0.1, # Threshold for feature importance
            'mask_regularization': 1e-4, # L2 regularization for mask network
            'explanation_method': 'gradient', # Gradient-based explanation
        }
        
        # RND (Random Network Distillation) Configuration
        # Critical for exploration in StateMask algorithm
        self.rnd_config = {
            'predictor_lr': 1e-4,       # Learning rate for predictor network
            'target_lr': 0.0,           # Target network is frozen
            'predictor_epochs': 1,      # Training epochs per update
            'normalize_rewards': True,   # Normalize RND rewards
            'reward_scale': 1.0,        # Scaling factor for RND rewards
            'update_frequency': 1,      # Update frequency for RND
            'network_arch': [128, 128], # RND network architecture
            'obs_norm_clip': 5.0,       # Observation normalization clipping
        }
        
        # Training Configuration - optimized for paper reproduction
        self.training_config = {
            'total_timesteps': 1000000, # Total training timesteps
            'eval_freq': 10000,         # Evaluation frequency
            'n_eval_episodes': 10,      # Number of evaluation episodes
            'save_freq': 50000,         # Model saving frequency
            'log_interval': 1,          # Logging interval
            'verbose': 1,               # Verbosity level
            'deterministic_eval': True, # Deterministic evaluation
        }
        
        # Evaluation Configuration - for consistent performance measurement
        self.eval_config = {
            'n_episodes': 100,          # Episodes for final evaluation
            'deterministic': True,      # Deterministic policy evaluation
            'render': False,            # Disable rendering for speed
            'return_episode_rewards': True,
            'eval_env_seed': 123,       # Fixed seed for evaluation environment
        }
        
        # Logging and Output Configuration
        self.logging_config = {
            'log_dir': './logs',
            'tensorboard_log': './tensorboard_logs',
            'save_path': './models',
            'results_path': './results',
            'plots_path': './plots',
            'checkpoint_path': './checkpoints',
        }
        
        # Numerical precision settings for L4 reproduction
        self.precision_config = {
            'torch_dtype': torch.float32,  # Consistent floating point precision
            'numpy_dtype': np.float32,
            'reward_scaling': 1.0,         # Reward scaling factor
            'obs_scaling': 1.0,            # Observation scaling factor
            'action_scaling': 1.0,         # Action scaling factor
        }
        
        # Environment wrapper configurations
        self.wrapper_config = {
            'normalize_observations': True,  # Normalize observations
            'normalize_rewards': False,      # Don't normalize rewards (handled by algorithm)
            'clip_observations': 10.0,       # Observation clipping
            'clip_rewards': 10.0,           # Reward clipping
            'frame_stack': 1,               # Frame stacking (1 for non-visual envs)
        }
        
        # Ensure directories exist
        self._create_directories()
    
    def get_env_config(self, env_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Dictionary containing environment-specific configuration
        """
        if env_name not in self.environments:
            raise ValueError(f"Environment {env_name} not configured. "
                           f"Available environments: {list(self.environments.keys())}")
        
        config = self.environments[env_name].copy()
        return config
    
    def get_algorithm_config(self, env_name: str) -> Dict[str, Any]:
        """
        Get configuration for the appropriate algorithm based on environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Dictionary containing algorithm-specific configuration
        """
        env_config = self.get_env_config(env_name)
        algorithm = env_config['algorithm']
        
        if algorithm == 'ppo':
            config = self.ppo_config.copy()
        elif algorithm == 'sac':
            config = self.sac_config.copy()
        else:
            raise ValueError(f"Algorithm {algorithm} not supported")
        
        # Set network architecture
        config['policy_kwargs']['net_arch'] = env_config['network_arch']
        
        return config
    
    def get_network_config(self, env_name: str) -> Dict[str, Any]:
        """
        Get network architecture configuration for a specific environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Dictionary containing network configuration
        """
        config = self.network_config.copy()
        env_config = self.get_env_config(env_name)
        config['net_arch'] = env_config['network_arch']
        return config
    
    def set_seeds(self):
        """Set random seeds for reproducibility - critical for L4 reproduction."""
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.numpy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.torch_seed)
            torch.cuda.manual_seed_all(self.torch_seed)
            # Additional CUDA determinism for L4 reproduction
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _create_directories(self):
        """Create necessary directories for logging and saving."""
        for path in self.logging_config.values():
            os.makedirs(path, exist_ok=True)
    
    def get_mixture_distribution_params(self, env_name: str) -> Tuple[float, float]:
        """
        Get mixture distribution parameters for the specified environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Tuple of (mixture_probability, complement_probability)
        """
        p = self.get_env_config(env_name)['mixture_prob']
        return p, 1.0 - p
    
    def get_rnd_coefficient(self, env_name: str) -> float:
        """
        Get RND exploration reward coefficient for the specified environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            RND coefficient λ
        """
        return self.get_env_config(env_name)['rnd_coeff']
    
    def get_mask_reward_bias(self, env_name: str) -> float:
        """
        Get mask network reward bias for the specified environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Mask reward bias α
        """
        return self.get_env_config(env_name)['mask_reward_bias']
    
    def get_algorithm_name(self, env_name: str) -> str:
        """
        Get the appropriate algorithm name for the environment.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Algorithm name ('ppo' or 'sac')
        """
        return self.get_env_config(env_name)['algorithm']
    
    def get_complete_config(self, env_name: str) -> Dict[str, Any]:
        """
        Get complete configuration for an environment including all relevant settings.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Complete configuration dictionary
        """
        env_config = self.get_env_config(env_name)
        algorithm_config = self.get_algorithm_config(env_name)
        network_config = self.get_network_config(env_name)
        
        return {
            'env_config': env_config,
            'algorithm_config': algorithm_config,
            'network_config': network_config,
            'statemask_config': self.statemask_config,
            'rnd_config': self.rnd_config,
            'training_config': self.training_config,
            'eval_config': self.eval_config,
            'precision_config': self.precision_config,
            'wrapper_config': self.wrapper_config,
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with custom values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
    
    def validate_config(self):
        """Validate configuration parameters for L4 reproduction requirements."""
        # Validate mixture probabilities
        for env_name, config in self.environments.items():
            p = config['mixture_prob']
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"Mixture probability for {env_name} must be in [0, 1], got {p}")
        
        # Validate RND coefficients
        for env_name, config in self.environments.items():
            rnd_coeff = config['rnd_coeff']
            if rnd_coeff < 0:
                raise ValueError(f"RND coefficient for {env_name} must be non-negative, got {rnd_coeff}")
        
        # Validate network architectures
        for env_name, config in self.environments.items():
            arch = config['network_arch']
            if not isinstance(arch, list) or not all(isinstance(x, int) and x > 0 for x in arch):
                raise ValueError(f"Network architecture for {env_name} must be a list of positive integers")
        
        # Validate algorithm assignments
        for env_name, config in self.environments.items():
            algorithm = config['algorithm']
            if algorithm not in ['ppo', 'sac']:
                raise ValueError(f"Algorithm for {env_name} must be 'ppo' or 'sac', got {algorithm}")
        
        # Validate critical hyperparameters match paper specifications
        paper_specs = {
            'Hopper-v3': {'mixture_prob': 0.25, 'rnd_coeff': 0.001, 'mask_reward_bias': 0.0001},
            'Walker2d-v3': {'mixture_prob': 0.25, 'rnd_coeff': 0.01, 'mask_reward_bias': 0.0001},
            'Reacher-v2': {'mixture_prob': 0.50, 'rnd_coeff': 0.001, 'mask_reward_bias': 0.0001},
            'HalfCheetah-v3': {'mixture_prob': 0.50, 'rnd_coeff': 0.01, 'mask_reward_bias': 0.0001},
        }
        
        for env_name, expected in paper_specs.items():
            if env_name in self.environments:
                actual = self.environments[env_name]
                for param, expected_value in expected.items():
                    if actual[param] != expected_value:
                        print(f"WARNING: {env_name} {param} = {actual[param]}, "
                              f"paper specifies {expected_value}")
        
        print("Configuration validation passed.")
        print("L4 reproduction requirements validated.")

# Global configuration instance
config = Config()

# Convenience functions for easy access
def get_env_config(env_name: str) -> Dict[str, Any]:
    """Get environment configuration."""
    return config.get_env_config(env_name)

def get_algorithm_config(env_name: str) -> Dict[str, Any]:
    """Get algorithm configuration for environment."""
    return config.get_algorithm_config(env_name)

def get_network_config(env_name: str) -> Dict[str, Any]:
    """Get network configuration."""
    return config.get_network_config(env_name)

def get_complete_config(env_name: str) -> Dict[str, Any]:
    """Get complete configuration for environment."""
    return config.get_complete_config(env_name)

def set_seeds():
    """Set random seeds for reproducibility."""
    config.set_seeds()

# Environment-specific helper functions with L4 optimization
def get_hopper_config() -> Dict[str, Any]:
    """Get Hopper-v3 specific configuration optimized for L4 reproduction."""
    return config.get_complete_config('Hopper-v3')

def get_walker2d_config() -> Dict[str, Any]:
    """Get Walker2d-v3 specific configuration optimized for L4 reproduction."""
    return config.get_complete_config('Walker2d-v3')

def get_reacher_config() -> Dict[str, Any]:
    """Get Reacher-v2 specific configuration optimized for L4 reproduction."""
    return config.get_complete_config('Reacher-v2')

def get_halfcheetah_config() -> Dict[str, Any]:
    """Get HalfCheetah-v3 specific configuration optimized for L4 reproduction."""
    return config.get_complete_config('HalfCheetah-v3')

def get_selfish_mining_config() -> Dict[str, Any]:
    """Get SelfishMining specific configuration optimized for L4 reproduction."""
    return config.get_complete_config('SelfishMining')

# Paper-specific parameter access functions
def get_mixture_prob(env_name: str) -> float:
    """Get mixture probability p for environment."""
    return config.get_mixture_distribution_params(env_name)[0]

def get_rnd_coeff(env_name: str) -> float:
    """Get RND coefficient λ for environment."""
    return config.get_rnd_coefficient(env_name)

def get_mask_bias(env_name: str) -> float:
    """Get mask reward bias α for environment."""
    return config.get_mask_reward_bias(env_name)

# Validate configuration on import
if __name__ == "__main__":
    config.validate_config()
    print("Configuration loaded successfully!")
    print(f"Available environments: {list(config.environments.keys())}")
    print(f"Device: {config.device}")
    print("L4 reproduction configuration ready.")
    
    # Print key parameters for verification
    print("\nKey hyperparameters (from Appendix C.3 Table 3):")
    for env_name in ['Hopper-v3', 'Walker2d-v3', 'Reacher-v2', 'HalfCheetah-v3']:
        if env_name in config.environments:
            env_config = config.environments[env_name]
            print(f"{env_name}: p={env_config['mixture_prob']}, "
                  f"λ={env_config['rnd_coeff']}, α={env_config['mask_reward_bias']}")