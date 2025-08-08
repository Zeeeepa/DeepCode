"""Environment wrappers for RICE implementation.

This module provides wrapper classes for gym environments to support:
1. State resetting functionality for critical states
2. Environment monitoring for metrics collection
3. Reward handling for dense/sparse rewards
4. Integration with MuJoCo environments

Key RICE Algorithm Dependencies:
- State reset capability for critical state exploration
- Simulator-based state manipulation for counterfactual analysis
- Robust state management for explanation generation
"""

import os
import gym
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
from collections import deque
import warnings

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
    
    This wrapper is critical for RICE algorithm as it provides:
    - State reset functionality for critical state exploration
    - Simulator-based state manipulation for counterfactual analysis
    - Metrics tracking for explanation quality assessment
    
    The RICE paper emphasizes the importance of being able to reset the
    environment to arbitrary states for generating explanations through
    counterfactual analysis.
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
        
        # Metrics tracking for RICE algorithm evaluation
        self.reward_history = deque(maxlen=config.history_size)
        self.length_history = deque(maxlen=config.history_size)
        self.state_visitation = {}  # Track state visitation frequency
        
        # State management - crucial for RICE algorithm
        self._last_state = None
        self._state_cache = {}
        self._state_reset_supported = self._check_state_reset_support()
        
        # Initialize state tracking for critical state identification
        self._critical_states = []
        self._state_importance_scores = {}
        
    def _check_state_reset_support(self) -> bool:
        """Check if the environment supports state reset functionality.
        
        This is essential for RICE algorithm which requires the ability to
        reset the environment to arbitrary states for explanation generation.
        
        Returns:
            True if state reset is supported, False otherwise
        """
        # Check for MuJoCo environment support
        if hasattr(self.env, 'unwrapped'):
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, 'set_state') and hasattr(unwrapped, 'sim'):
                return True
            if hasattr(unwrapped, 'state'):
                return True
                
        # Check for direct environment support
        if hasattr(self.env, 'set_state') or hasattr(self.env, 'state'):
            return True
            
        warnings.warn(
            "Environment does not support state reset. "
            "RICE algorithm functionality will be limited."
        )
        return False

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
        # This supports the paper's experiments with both dense and sparse rewards
        if self.config.is_sparse:
            if isinstance(state, np.ndarray):
                # For locomotion tasks, use position-based sparse rewards
                pos_x = state[0] if len(state.shape) == 1 else state[..., 0]
                reward = float(pos_x > self.config.reward_threshold)
            
        # Track metrics for RICE algorithm evaluation
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Store state for potential resetting - critical for RICE
        self._last_state = state.copy() if isinstance(state, np.ndarray) else state
        state_key = self._get_state_key(state)
        self.state_visitation[state_key] = self.state_visitation.get(state_key, 0) + 1
        
        # Track state importance for critical state identification
        self._update_state_importance(state, reward)
        
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
        self._last_state = state.copy() if isinstance(state, np.ndarray) else state
        return state
        
    def reset_to_state(self, target_state: Union[np.ndarray, Dict]) -> np.ndarray:
        """Reset environment to a specific target state.
        
        This is the core functionality required by RICE algorithm for:
        1. Critical state exploration
        2. Counterfactual analysis for explanation generation
        3. State-based importance sampling
        
        The paper emphasizes this as a key simulator dependency for the algorithm.
        
        Args:
            target_state: State to reset environment to. Can be:
                - np.ndarray: Direct state vector
                - Dict: State dictionary with 'qpos' and 'qvel' for MuJoCo
                
        Returns:
            Reset state observation
            
        Raises:
            NotImplementedError: If environment doesn't support state reset
        """
        if not self._state_reset_supported:
            warnings.warn(
                "State reset not supported. Using default reset. "
                "RICE algorithm functionality will be limited."
            )
            return self.reset()
            
        try:
            # Handle MuJoCo environments (primary target for RICE paper)
            if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'sim'):
                unwrapped = self.env.unwrapped
                
                if isinstance(target_state, dict):
                    # State provided as dictionary with qpos/qvel
                    qpos = target_state.get('qpos', unwrapped.sim.data.qpos.copy())
                    qvel = target_state.get('qvel', unwrapped.sim.data.qvel.copy())
                elif isinstance(target_state, np.ndarray):
                    # State provided as concatenated vector
                    nq = unwrapped.model.nq
                    qpos = target_state[:nq]
                    qvel = target_state[nq:nq+unwrapped.model.nv] if len(target_state) > nq else np.zeros(unwrapped.model.nv)
                else:
                    raise ValueError(f"Unsupported target_state type: {type(target_state)}")
                
                # Set the state in MuJoCo simulator
                unwrapped.set_state(qpos, qvel)
                
                # Get observation after state reset
                if hasattr(unwrapped, '_get_obs'):
                    obs = unwrapped._get_obs()
                elif hasattr(unwrapped, 'get_obs'):
                    obs = unwrapped.get_obs()
                else:
                    # Fallback: step with zero action to get observation
                    obs, _, _, _ = unwrapped.step(np.zeros(unwrapped.action_space.shape))
                    # Reset to the target state again since step moved us
                    unwrapped.set_state(qpos, qvel)
                    obs, _, _, _ = unwrapped.step(np.zeros(unwrapped.action_space.shape))
                    unwrapped.set_state(qpos, qvel)
                    if hasattr(unwrapped, '_get_obs'):
                        obs = unwrapped._get_obs()
                
                self._last_state = obs.copy() if isinstance(obs, np.ndarray) else obs
                return obs
                
            # Handle generic Gym environments with state attribute
            elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'state'):
                if isinstance(target_state, dict):
                    # Extract state vector from dictionary
                    state_vector = target_state.get('state', target_state.get('qpos', target_state))
                else:
                    state_vector = target_state
                    
                self.env.unwrapped.state = state_vector
                
                # Get observation
                if hasattr(self.env.unwrapped, '_get_obs'):
                    obs = self.env.unwrapped._get_obs()
                elif hasattr(self.env.unwrapped, 'get_obs'):
                    obs = self.env.unwrapped.get_obs()
                else:
                    obs = state_vector
                    
                self._last_state = obs.copy() if isinstance(obs, np.ndarray) else obs
                return obs
                
            # Handle environments with direct set_state method
            elif hasattr(self.env, 'set_state'):
                if isinstance(target_state, dict):
                    qpos = target_state.get('qpos')
                    qvel = target_state.get('qvel')
                    if qpos is not None and qvel is not None:
                        self.env.set_state(qpos, qvel)
                    else:
                        self.env.set_state(target_state)
                else:
                    self.env.set_state(target_state)
                    
                obs = self.env._get_obs() if hasattr(self.env, '_get_obs') else target_state
                self._last_state = obs.copy() if isinstance(obs, np.ndarray) else obs
                return obs
                
            else:
                raise NotImplementedError(
                    "Environment doesn't support state reset. "
                    "Required for RICE algorithm functionality."
                )
                
        except Exception as e:
            warnings.warn(
                f"State reset failed: {e}. Using default reset. "
                "This may impact RICE algorithm performance."
            )
            return self.reset()
    
    def get_current_state_dict(self) -> Dict:
        """Get current environment state as dictionary for saving/loading.
        
        This is used by RICE algorithm to save critical states for later analysis.
        
        Returns:
            Dictionary containing current state information
        """
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'sim'):
            # MuJoCo environment
            return {
                'qpos': self.env.unwrapped.sim.data.qpos.copy(),
                'qvel': self.env.unwrapped.sim.data.qvel.copy(),
                'obs': self._last_state.copy() if isinstance(self._last_state, np.ndarray) else self._last_state
            }
        elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'state'):
            # Generic Gym environment
            return {
                'state': self.env.unwrapped.state.copy() if isinstance(self.env.unwrapped.state, np.ndarray) else self.env.unwrapped.state,
                'obs': self._last_state.copy() if isinstance(self._last_state, np.ndarray) else self._last_state
            }
        else:
            # Fallback
            return {
                'obs': self._last_state.copy() if isinstance(self._last_state, np.ndarray) else self._last_state
            }
    
    def _update_state_importance(self, state: np.ndarray, reward: float):
        """Update state importance scores for critical state identification.
        
        This supports RICE algorithm's need to identify critical states
        that significantly impact agent behavior.
        
        Args:
            state: Current state
            reward: Reward received in this state
        """
        state_key = self._get_state_key(state)
        if state_key not in self._state_importance_scores:
            self._state_importance_scores[state_key] = {
                'total_reward': 0.0,
                'visit_count': 0,
                'avg_reward': 0.0
            }
        
        scores = self._state_importance_scores[state_key]
        scores['total_reward'] += reward
        scores['visit_count'] += 1
        scores['avg_reward'] = scores['total_reward'] / scores['visit_count']
        
    def get_critical_states(self, top_k: int = 10) -> List[Tuple[str, Dict]]:
        """Get top-k critical states based on importance scores.
        
        Critical states are those with high average rewards or high visitation,
        which are important for RICE algorithm's explanation generation.
        
        Args:
            top_k: Number of top critical states to return
            
        Returns:
            List of (state_key, importance_info) tuples
        """
        # Sort states by average reward (descending)
        sorted_states = sorted(
            self._state_importance_scores.items(),
            key=lambda x: x[1]['avg_reward'],
            reverse=True
        )
        
        return sorted_states[:top_k]
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current environment metrics.
        
        Returns:
            Dictionary of metrics including:
            - Average reward
            - Average episode length
            - State visitation statistics
            - State reset capability status
        """
        metrics = {
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'avg_length': np.mean(self.length_history) if self.length_history else 0.0,
            'num_episodes': len(self.reward_history),
            'unique_states_visited': len(self.state_visitation),
            'state_reset_supported': self._state_reset_supported,
            'critical_states_identified': len(self._state_importance_scores)
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
            # Round to reduce precision for better state grouping
            rounded_state = np.round(state, decimals=3)
            return rounded_state.tobytes()
        return str(state)
        
    def save_state(self, key: str):
        """Save current state for later restoration.
        
        This is used by RICE algorithm to save important states during
        exploration for later analysis and explanation generation.
        
        Args:
            key: Identifier for saved state
        """
        self._state_cache[key] = self.get_current_state_dict()
            
    def load_state(self, key: str) -> Optional[np.ndarray]:
        """Restore previously saved state.
        
        Args:
            key: Identifier for state to restore
            
        Returns:
            Restored state observation if successful, None otherwise
        """
        if key not in self._state_cache:
            return None
            
        saved_state = self._state_cache[key]
        return self.reset_to_state(saved_state)
    
    def is_state_reset_supported(self) -> bool:
        """Check if state reset is supported.
        
        Returns:
            True if state reset functionality is available
        """
        return self._state_reset_supported