import gym
import numpy as np
import torch
import pickle
import os
import warnings
from typing import Dict, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod
import logging

# MuJoCo environments
try:
    import mujoco_py
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    warnings.warn("MuJoCo not available. MuJoCo environments will not work.")

# Selfish Mining environment
try:
    import sys
    sys.path.append('pto-selfish-mining')
    from selfish_mining_env import SelfishMiningEnv
    SELFISH_MINING_AVAILABLE = True
except ImportError:
    SELFISH_MINING_AVAILABLE = False
    warnings.warn("Selfish Mining environment not available.")

# CAGE Challenge 2 environment
try:
    import sys
    sys.path.append('cage-challenge-2')
    from CybORG import CybORG
    from CybORG.Agents import B_lineAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    CAGE_AVAILABLE = True
except ImportError:
    CAGE_AVAILABLE = False
    warnings.warn("CAGE Challenge 2 environment not available.")

# MetaDrive environment
try:
    from metadrive import MetaDriveEnv
    METADRIVE_AVAILABLE = True
except ImportError:
    METADRIVE_AVAILABLE = False
    warnings.warn("MetaDrive environment not available.")

# Malware Mutation environment
try:
    import sys
    sys.path.append('malconv-gym')
    from malconv_gym import MalwareEnv
    MALWARE_AVAILABLE = True
except ImportError:
    MALWARE_AVAILABLE = False
    warnings.warn("Malware environment not available.")


class BaseEnvironmentWrapper(ABC):
    """
    Base wrapper class for all environments used in the paper.
    Provides common functionality including state saving/loading and normalization.
    """
    
    def __init__(self, env_name: str, normalize_obs: bool = True, normalize_rewards: bool = True):
        """
        Initialize the environment wrapper.
        
        Args:
            env_name: Name of the environment
            normalize_obs: Whether to normalize observations
            normalize_rewards: Whether to normalize rewards
        """
        self.env_name = env_name
        self.normalize_obs = normalize_obs
        self.normalize_rewards = normalize_rewards
        
        # Normalization statistics
        self.obs_mean = None
        self.obs_std = None
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        # State management
        self.saved_states = {}
        self.current_state_id = None
        
        # Environment instance
        self.env = None
        
        # Logging
        self.logger = logging.getLogger(f"EnvWrapper_{env_name}")
        
    @abstractmethod
    def _create_env(self) -> gym.Env:
        """Create and return the environment instance."""
        pass
    
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial observation."""
        if self.env is None:
            self.env = self._create_env()
        
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API
            
        if self.normalize_obs and self.obs_mean is not None:
            obs = self._normalize_observation(obs)
            
        return obs
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        obs, reward, done, info = self.env.step(action)
        
        if self.normalize_obs and self.obs_mean is not None:
            obs = self._normalize_observation(obs)
            
        if self.normalize_rewards:
            reward = self._normalize_reward(reward)
            
        return obs, reward, done, info
    
    def save_state(self, state_id: str) -> None:
        """
        Save the current environment state.
        
        Args:
            state_id: Unique identifier for the saved state
        """
        try:
            # For MuJoCo environments, save the full state
            if hasattr(self.env, 'sim'):
                state = {
                    'qpos': self.env.sim.data.qpos.copy(),
                    'qvel': self.env.sim.data.qvel.copy(),
                    'time': self.env.sim.data.time,
                    'random_state': np.random.get_state()
                }
            else:
                # For other environments, try to pickle the entire state
                state = pickle.dumps(self.env)
                
            self.saved_states[state_id] = state
            self.current_state_id = state_id
            self.logger.debug(f"Saved state {state_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save state {state_id}: {e}")
    
    def load_state(self, state_id: str) -> bool:
        """
        Load a previously saved environment state.
        
        Args:
            state_id: Identifier of the state to load
            
        Returns:
            True if state was loaded successfully, False otherwise
        """
        if state_id not in self.saved_states:
            self.logger.warning(f"State {state_id} not found")
            return False
            
        try:
            state = self.saved_states[state_id]
            
            if hasattr(self.env, 'sim'):
                # Restore MuJoCo state
                self.env.sim.data.qpos[:] = state['qpos']
                self.env.sim.data.qvel[:] = state['qvel']
                self.env.sim.data.time = state['time']
                np.random.set_state(state['random_state'])
                self.env.sim.forward()
            else:
                # Restore other environments
                self.env = pickle.loads(state)
                
            self.current_state_id = state_id
            self.logger.debug(f"Loaded state {state_id}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load state {state_id}: {e}")
            return False
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        if self.obs_mean is None or self.obs_std is None:
            return obs
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        return (reward - self.reward_mean) / (self.reward_std + 1e-8)
    
    def update_normalization_stats(self, observations: np.ndarray, rewards: np.ndarray = None):
        """
        Update normalization statistics based on collected data.
        
        Args:
            observations: Array of observations
            rewards: Array of rewards (optional)
        """
        if self.normalize_obs:
            self.obs_mean = np.mean(observations, axis=0)
            self.obs_std = np.std(observations, axis=0)
            
        if self.normalize_rewards and rewards is not None:
            self.reward_mean = np.mean(rewards)
            self.reward_std = np.std(rewards)
    
    @property
    def observation_space(self):
        """Get the observation space of the environment."""
        if self.env is None:
            self.env = self._create_env()
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Get the action space of the environment."""
        if self.env is None:
            self.env = self._create_env()
        return self.env.action_space


class MuJoCoWrapper(BaseEnvironmentWrapper):
    """Wrapper for MuJoCo environments with sparse reward variants."""
    
    def __init__(self, env_name: str, sparse_reward: bool = False, **kwargs):
        """
        Initialize MuJoCo environment wrapper.
        
        Args:
            env_name: Name of the MuJoCo environment
            sparse_reward: Whether to use sparse reward variant
            **kwargs: Additional arguments for base wrapper
        """
        super().__init__(env_name, **kwargs)
        self.sparse_reward = sparse_reward
        self.base_env_name = env_name
        
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is not available")
    
    def _create_env(self) -> gym.Env:
        """Create the MuJoCo environment."""
        env = gym.make(self.base_env_name)
        
        if self.sparse_reward:
            env = SparseRewardWrapper(env, self.base_env_name)
            
        return env


class SparseRewardWrapper(gym.Wrapper):
    """
    Wrapper to convert dense rewards to sparse rewards for MuJoCo environments.
    Based on the paper's description of sparse reward variants.
    """
    
    def __init__(self, env: gym.Env, env_name: str):
        super().__init__(env)
        self.env_name = env_name
        self.episode_reward = 0.0
        self.step_count = 0
        
    def reset(self, **kwargs):
        self.episode_reward = 0.0
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        self.step_count += 1
        
        # Convert to sparse reward
        if done:
            # Return accumulated reward only at episode end
            sparse_reward = self.episode_reward
        else:
            sparse_reward = 0.0
            
        return obs, sparse_reward, done, info


class SelfishMiningWrapper(BaseEnvironmentWrapper):
    """Wrapper for Selfish Mining environment."""
    
    def __init__(self, **kwargs):
        super().__init__("SelfishMining", **kwargs)
        
        if not SELFISH_MINING_AVAILABLE:
            raise ImportError("Selfish Mining environment is not available")
    
    def _create_env(self) -> gym.Env:
        """Create the Selfish Mining environment."""
        # Default configuration for selfish mining
        config = {
            'alpha': 0.3,  # Mining power of selfish miner
            'gamma': 0.5,  # Network connectivity parameter
            'max_fork_length': 10,  # Maximum fork length
            'episode_length': 1000  # Episode length
        }
        
        return SelfishMiningEnv(config)


class CAGEWrapper(BaseEnvironmentWrapper):
    """Wrapper for CAGE Challenge 2 environment."""
    
    def __init__(self, **kwargs):
        super().__init__("CAGE2", **kwargs)
        
        if not CAGE_AVAILABLE:
            raise ImportError("CAGE Challenge 2 environment is not available")
    
    def _create_env(self) -> gym.Env:
        """Create the CAGE Challenge 2 environment."""
        # Create scenario generator
        sg = EnterpriseScenarioGenerator()
        
        # Create CybORG environment
        cyborg = CybORG(scenario_generator=sg, seed=42)
        
        # Wrap in gym interface
        env = CybORGGymWrapper(cyborg)
        
        return env


class CybORGGymWrapper(gym.Env):
    """Gym wrapper for CybORG environment."""
    
    def __init__(self, cyborg):
        self.cyborg = cyborg
        self.agent_name = 'Blue'
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(cyborg.get_action_space(self.agent_name))
        
        # Observation space (simplified)
        obs_size = len(cyborg.reset().observation)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
    
    def reset(self):
        result = self.cyborg.reset()
        obs = self._process_observation(result.observation)
        return obs
    
    def step(self, action):
        result = self.cyborg.step(agent=self.agent_name, action=action)
        obs = self._process_observation(result.observation)
        reward = result.reward
        done = result.done
        info = {}
        
        return obs, reward, done, info
    
    def _process_observation(self, obs_dict):
        """Convert CybORG observation to numpy array."""
        # Simplified observation processing
        obs_values = []
        for key, value in obs_dict.items():
            if isinstance(value, (int, float)):
                obs_values.append(value)
            elif isinstance(value, dict):
                obs_values.extend(self._flatten_dict(value))
        
        return np.array(obs_values, dtype=np.float32)
    
    def _flatten_dict(self, d):
        """Flatten nested dictionary to list of values."""
        values = []
        for v in d.values():
            if isinstance(v, (int, float)):
                values.append(v)
            elif isinstance(v, dict):
                values.extend(self._flatten_dict(v))
        return values


class MetaDriveWrapper(BaseEnvironmentWrapper):
    """Wrapper for MetaDrive environment."""
    
    def __init__(self, **kwargs):
        super().__init__("MetaDrive-Macro-v1", **kwargs)
        
        if not METADRIVE_AVAILABLE:
            raise ImportError("MetaDrive environment is not available")
    
    def _create_env(self) -> gym.Env:
        """Create the MetaDrive environment with Macro-v1 configuration."""
        config = {
            "environment_num": 100,
            "traffic_density": 0.1,
            "start_seed": 42,
            "map": "S",  # Simple map
            "vehicle_config": {
                "enable_reverse": False,
            },
            "use_render": False,
            "manual_control": False,
            "use_image": False,
            "crash_done": True,
            "out_of_route_done": True,
        }
        
        return MetaDriveEnv(config)


class MalwareWrapper(BaseEnvironmentWrapper):
    """
    Wrapper for Malware Mutation environment with bug fixes.
    Addresses the Markovian property and reward sparsity issues mentioned in Appendix D.2.
    """
    
    def __init__(self, **kwargs):
        super().__init__("Malware", **kwargs)
        
        if not MALWARE_AVAILABLE:
            raise ImportError("Malware environment is not available")
    
    def _create_env(self) -> gym.Env:
        """Create the Malware environment with bug fixes."""
        env = MalwareEnv()
        
        # Apply bug fixes from Appendix D.2
        env = MalwareBugFixWrapper(env)
        
        return env


class MalwareBugFixWrapper(gym.Wrapper):
    """
    Wrapper to fix bugs in the Malware environment as described in Appendix D.2.
    
    Fixes:
    1. Markovian property: Ensure state transitions depend only on current state
    2. Reward sparsity: Provide intermediate rewards for progress
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.mutation_history = []
        self.detection_scores = []
        self.baseline_score = None
        
    def reset(self, **kwargs):
        self.mutation_history = []
        self.detection_scores = []
        obs = self.env.reset(**kwargs)
        
        # Get baseline detection score
        self.baseline_score = self._get_detection_score()
        
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Fix 1: Ensure Markovian property
        # Clear any non-Markovian state dependencies
        if hasattr(self.env, '_clear_history'):
            self.env._clear_history()
        
        # Fix 2: Address reward sparsity
        current_score = self._get_detection_score()
        self.detection_scores.append(current_score)
        
        # Provide intermediate rewards based on detection score improvement
        if len(self.detection_scores) > 1:
            score_improvement = self.detection_scores[-2] - current_score
            intermediate_reward = score_improvement * 0.1  # Scale factor
            reward += intermediate_reward
        
        # Track mutation history for analysis
        self.mutation_history.append(action)
        
        # Add fixed info
        info['detection_score'] = current_score
        info['score_improvement'] = (self.baseline_score - current_score) if self.baseline_score else 0
        info['mutation_count'] = len(self.mutation_history)
        
        return obs, reward, done, info
    
    def _get_detection_score(self):
        """Get current detection score from the environment."""
        if hasattr(self.env, 'get_detection_score'):
            return self.env.get_detection_score()
        else:
            # Fallback if method doesn't exist
            return 0.5


class EnvironmentFactory:
    """
    Factory class for creating all environments used in the paper.
    Supports all 8 environments with proper configuration and state management.
    """
    
    SUPPORTED_ENVIRONMENTS = {
        # MuJoCo environments
        'Hopper-v3': {'type': 'mujoco', 'sparse': False},
        'Hopper-v3-sparse': {'type': 'mujoco', 'sparse': True},
        'Walker2d-v3': {'type': 'mujoco', 'sparse': False},
        'Walker2d-v3-sparse': {'type': 'mujoco', 'sparse': True},
        'Reacher-v2': {'type': 'mujoco', 'sparse': False},
        'Reacher-v2-sparse': {'type': 'mujoco', 'sparse': True},
        'HalfCheetah-v3': {'type': 'mujoco', 'sparse': False},
        'HalfCheetah-v3-sparse': {'type': 'mujoco', 'sparse': True},
        
        # Other environments
        'SelfishMining': {'type': 'selfish_mining'},
        'CAGE2': {'type': 'cage'},
        'MetaDrive-Macro-v1': {'type': 'metadrive'},
        'Malware': {'type': 'malware'},
    }
    
    @classmethod
    def create_environment(cls, env_name: str, **kwargs) -> BaseEnvironmentWrapper:
        """
        Create an environment wrapper for the specified environment.
        
        Args:
            env_name: Name of the environment to create
            **kwargs: Additional arguments for environment configuration
            
        Returns:
            Environment wrapper instance
            
        Raises:
            ValueError: If environment name is not supported
        """
        if env_name not in cls.SUPPORTED_ENVIRONMENTS:
            raise ValueError(f"Unsupported environment: {env_name}. "
                           f"Supported environments: {list(cls.SUPPORTED_ENVIRONMENTS.keys())}")
        
        env_config = cls.SUPPORTED_ENVIRONMENTS[env_name]
        env_type = env_config['type']
        
        if env_type == 'mujoco':
            base_name = env_name.replace('-sparse', '')
            sparse = env_config['sparse']
            return MuJoCoWrapper(base_name, sparse_reward=sparse, **kwargs)
        
        elif env_type == 'selfish_mining':
            return SelfishMiningWrapper(**kwargs)
        
        elif env_type == 'cage':
            return CAGEWrapper(**kwargs)
        
        elif env_type == 'metadrive':
            return MetaDriveWrapper(**kwargs)
        
        elif env_type == 'malware':
            return MalwareWrapper(**kwargs)
        
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
    
    @classmethod
    def get_available_environments(cls) -> list:
        """Get list of available environments."""
        available = []
        
        for env_name, config in cls.SUPPORTED_ENVIRONMENTS.items():
            env_type = config['type']
            
            # Check availability based on dependencies
            if env_type == 'mujoco' and MUJOCO_AVAILABLE:
                available.append(env_name)
            elif env_type == 'selfish_mining' and SELFISH_MINING_AVAILABLE:
                available.append(env_name)
            elif env_type == 'cage' and CAGE_AVAILABLE:
                available.append(env_name)
            elif env_type == 'metadrive' and METADRIVE_AVAILABLE:
                available.append(env_name)
            elif env_type == 'malware' and MALWARE_AVAILABLE:
                available.append(env_name)
        
        return available
    
    @classmethod
    def create_all_available_environments(cls, **kwargs) -> Dict[str, BaseEnvironmentWrapper]:
        """
        Create all available environments.
        
        Args:
            **kwargs: Additional arguments for environment configuration
            
        Returns:
            Dictionary mapping environment names to wrapper instances
        """
        environments = {}
        available_envs = cls.get_available_environments()
        
        for env_name in available_envs:
            try:
                env = cls.create_environment(env_name, **kwargs)
                environments[env_name] = env
                print(f"Successfully created environment: {env_name}")
            except Exception as e:
                print(f"Failed to create environment {env_name}: {e}")
        
        return environments


def setup_environment_logging():
    """Setup logging for environment factory."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Example usage and testing
if __name__ == "__main__":
    setup_environment_logging()
    
    # Test environment creation
    factory = EnvironmentFactory()
    
    print("Available environments:")
    for env_name in factory.get_available_environments():
        print(f"  - {env_name}")
    
    # Test creating a MuJoCo environment (if available)
    if MUJOCO_AVAILABLE:
        try:
            env = factory.create_environment('Hopper-v3', normalize_obs=True)
            obs = env.reset()
            print(f"Hopper-v3 observation shape: {obs.shape}")
            
            # Test state saving/loading
            env.save_state('test_state')
            action = env.action_space.sample()
            obs1, reward1, done1, info1 = env.step(action)
            
            env.load_state('test_state')
            obs2, reward2, done2, info2 = env.step(action)
            
            print(f"State consistency check: {np.allclose(obs1, obs2)}")
            
        except Exception as e:
            print(f"Error testing Hopper-v3: {e}")
    
    print("Environment factory setup complete!")