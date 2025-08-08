"""Environment wrappers for RICE implementation.

This module provides wrapper classes for gym environments to support:
1. State resetting functionality for critical states
2. Environment monitoring for metrics collection
3. Reward handling for dense/sparse rewards
4. Integration with MuJoCo environments
"""

import os
import gym
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque

@dataclass
class EnvConfig:
    """Configuration for environment setup"""
    env_name: str
    is_sparse: bool = False
    reward_threshold: float = 0.6  # For sparse reward setting
    max_episode_steps: int = 1000
    history_size: int = 100  # For storing metrics

class RICEEnvWrapper(gym.Wrapper):
    """Environment wrapper for RICE algorithm implementation.
    
    Adds functionality for:
    - Resetting to specific states for critical state exploration
    - Monitoring metrics and rewards
    - Handling both dense and sparse reward settings
    """
    
    def __init__(self, config: EnvConfig):
        """Initialize wrapper with given configuration.
        
        Args:
            config: Environment configuration parameters
        """
        env = gym.make(config.env_name)
        super().__init__(env)
        
        self.config = config
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Metrics tracking
        self.reward_history = deque(maxlen=config.history_size)
        self.length_history = deque(maxlen=config.history_size)
        self.state_visitation = {}  # Track state visitation frequency
        
        # State management
        self._last_state = None
        self._state_cache = {}
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute environment step and process rewards/metrics.
        
        Args:
            action: Action to take in environment
            
        Returns:
            state: Next state
            reward: Reward value
            done: Episode termination flag 
            info: Additional information
        """
        state, reward, done, info = self.env.step(action)
        
        # Handle sparse rewards if configured
        if self.config.is_sparse:
            if isinstance(state, np.ndarray):
                pos_x = state[0] if len(state.shape) == 1 else state[..., 0]
                reward = float(pos_x > self.config.reward_threshold)
            
        # Track metrics
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Store state for potential resetting
        self._last_state = state.copy()
        state_key = self._get_state_key(state)
        self.state_visitation[state_key] = self.state_visitation.get(state_key, 0) + 1
        
        # Handle episode completion
        if done:
            self.reward_history.append(self.current_episode_reward)
            self.length_history.append(self.current_episode_length)
            info.update({
                'episode': {
                    'r': self.current_episode_reward,
                    'l': self.current_episode_length
                }
            })
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        return state, reward, done, info
        
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment to initial state or specified state.
        
        Args:
            **kwargs: Additional arguments including target_state if resetting to specific state
            
        Returns:
            Initial or target state
        """
        if 'target_state' in kwargs:
            return self.reset_to_state(kwargs['target_state'])
            
        state = self.env.reset()
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self._last_state = state.copy()
        return state
        
    def reset_to_state(self, target_state: np.ndarray) -> np.ndarray:
        """Reset environment to a specific target state.
        
        This is crucial for RICE algorithm to explore from critical states.
        
        Args:
            target_state: State to reset environment to
            
        Returns:
            Reset state
        """
        # For MuJoCo envs, we need to set internal state
        if hasattr(self.env, 'sim'):
            qpos = target_state[:self.env.model.nq]
            qvel = target_state[self.env.model.nq:]
            self.env.set_state(qpos, qvel)
            return self.env.get_obs()
        
        # For other environments, attempt direct state setting
        if hasattr(self.env, 'state'):
            self.env.state = target_state
            return target_state
            
        # Fallback to normal reset if state setting not supported
        return self.reset()
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current environment metrics.
        
        Returns:
            Dictionary of metrics including:
            - Average reward
            - Average episode length
            - State visitation statistics
        """
        metrics = {
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'avg_length': np.mean(self.length_history) if self.length_history else 0.0,
            'num_episodes': len(self.reward_history),
            'unique_states_visited': len(self.state_visitation)
        }
        return metrics
        
    def _get_state_key(self, state: np.ndarray) -> str:
        """Convert state array to hashable key for tracking.
        
        Args:
            state: Environment state array
            
        Returns:
            Hashable state key
        """
        if isinstance(state, np.ndarray):
            return state.tobytes()
        return str(state)
        
    def save_state(self, key: str):
        """Save current state for later restoration.
        
        Args:
            key: Identifier for saved state
        """
        if hasattr(self.env, 'sim'):
            self._state_cache[key] = {
                'qpos': self.env.sim.data.qpos.copy(),
                'qvel': self.env.sim.data.qvel.copy()
            }
        else:
            self._state_cache[key] = self._last_state.copy()
            
    def load_state(self, key: str) -> Optional[np.ndarray]:
        """Restore previously saved state.
        
        Args:
            key: Identifier for state to restore
            
        Returns:
            Restored state if successful, None otherwise
        """
        if key not in self._state_cache:
            return None
            
        saved_state = self._state_cache[key]
        if hasattr(self.env, 'sim'):
            self.env.set_state(saved_state['qpos'], saved_state['qvel'])
            return self.env.get_obs()
        
        return self.reset_to_state(saved_state)