"""
Mixed Initial State Distribution Module

This module implements the mixed initial state distribution mechanism described in the RICE paper,
which combines default initial states with critical states identified through explanation methods.
The mixing helps prevent overfitting while enabling effective exploration.

Key Algorithm Components:
- Mixed sampling with probability p for critical vs default states
- Critical state selection based on importance/value
- State buffer management with capacity constraints
- Environment state reset functionality
- Integration with trajectory collection and state importance evaluation
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import random
import pickle
import os
import logging
from collections import deque
import copy

@dataclass
class StateInfo:
    """Data class for storing state information with metadata"""
    state: np.ndarray  # The actual state vector/array
    is_critical: bool  # Whether this is a critical state
    value: float  # Value/importance of the state (higher = more important)
    metadata: Dict  # Additional metadata (e.g., episode info, timestamp)
    timestamp: float = 0.0  # When this state was added
    access_count: int = 0  # How many times this state has been sampled
    trajectory_id: Optional[int] = None  # ID of the trajectory this state came from
    step_in_trajectory: Optional[int] = None  # Step number in the trajectory

class TrajectoryCollector:
    """
    Collects trajectories and identifies critical states based on importance scores.
    This is essential for the RICE algorithm's state identification process.
    """
    
    def __init__(self, max_trajectories: int = 1000):
        self.max_trajectories = max_trajectories
        self.trajectories = deque(maxlen=max_trajectories)
        self.trajectory_counter = 0
        
    def add_trajectory(self, states: List[np.ndarray], rewards: List[float], 
                      actions: List[np.ndarray], importance_scores: Optional[List[float]] = None):
        """Add a complete trajectory with importance scores"""
        trajectory = {
            'id': self.trajectory_counter,
            'states': states,
            'rewards': rewards,
            'actions': actions,
            'importance_scores': importance_scores or [0.0] * len(states),
            'total_reward': sum(rewards)
        }
        self.trajectories.append(trajectory)
        self.trajectory_counter += 1
        
    def get_critical_states(self, threshold_percentile: float = 80.0) -> List[Tuple[np.ndarray, float]]:
        """Extract critical states based on importance score threshold"""
        all_states_with_scores = []
        
        for traj in self.trajectories:
            for i, (state, score) in enumerate(zip(traj['states'], traj['importance_scores'])):
                all_states_with_scores.append((state, score, traj['id'], i))
        
        if not all_states_with_scores:
            return []
            
        # Calculate threshold based on percentile
        scores = [item[1] for item in all_states_with_scores]
        threshold = np.percentile(scores, threshold_percentile)
        
        # Filter critical states
        critical_states = [(state, score) for state, score, _, _ in all_states_with_scores 
                          if score >= threshold]
        
        return critical_states

class MixedStateDistribution:
    """
    Implements mixed initial state distribution that combines default initial states
    with critical states identified through explanation methods.
    
    Algorithm (from RICE paper Algorithm 2):
    1. With probability p, sample from critical state buffer
    2. With probability (1-p), sample from default initial distribution
    3. Critical states are selected based on importance/value scores
    4. Buffers are maintained with capacity constraints
    
    Key Features:
    - Maintains separate buffers for default and critical states
    - Implements configurable mixing ratio p
    - Supports weighted sampling based on state values/importance
    - Handles state persistence and loading
    - Provides environment state reset functionality
    - Integrates with trajectory collection for critical state identification
    """
    
    def __init__(
        self,
        mixing_ratio: float = 0.25,  # p parameter controlling mix of critical vs default states
        max_buffer_size: int = 10000,
        state_dim: Union[int, Tuple[int, ...]] = None,
        device: str = "cpu",
        critical_state_selection: str = "weighted",  # "weighted", "top_k", "uniform"
        importance_threshold_percentile: float = 80.0,  # Percentile threshold for critical states
        trajectory_buffer_size: int = 1000
    ):
        """
        Initialize the mixed state distribution.
        
        Args:
            mixing_ratio: Float between 0 and 1 controlling ratio of critical states vs default states (p in paper)
            max_buffer_size: Maximum number of states to store in each buffer
            state_dim: Dimensions of the state space
            device: Device to store tensors on
            critical_state_selection: Method for selecting critical states ("weighted", "top_k", "uniform")
            importance_threshold_percentile: Percentile threshold for identifying critical states
            trajectory_buffer_size: Maximum number of trajectories to store
        """
        self.mixing_ratio = mixing_ratio  # This is the 'p' parameter from the paper
        self.max_buffer_size = max_buffer_size
        self.state_dim = state_dim
        self.device = device
        self.critical_state_selection = critical_state_selection
        self.importance_threshold_percentile = importance_threshold_percentile
        
        # Initialize state buffers
        self.default_states: List[StateInfo] = []
        self.critical_states: List[StateInfo] = []
        
        # Trajectory collector for critical state identification
        self.trajectory_collector = TrajectoryCollector(max_trajectories=trajectory_buffer_size)
        
        # Statistics for monitoring
        self.total_samples = 0
        self.critical_samples = 0
        self.default_samples = 0
        self.stats = {
            "n_default_states": 0,
            "n_critical_states": 0,
            "avg_default_value": 0.0,
            "avg_critical_value": 0.0,
            "critical_sample_ratio": 0.0,
            "buffer_utilization": 0.0,
            "n_trajectories": 0
        }
        
        # Logger for debugging
        self.logger = logging.getLogger(__name__)

    def collect_trajectory(self, states: List[np.ndarray], rewards: List[float], 
                          actions: List[np.ndarray], importance_scores: Optional[List[float]] = None):
        """
        Collect a trajectory and automatically identify critical states.
        
        Args:
            states: List of states in the trajectory
            rewards: List of rewards for each step
            actions: List of actions taken
            importance_scores: Optional importance scores for each state (from explanation method)
        """
        # Add trajectory to collector
        self.trajectory_collector.add_trajectory(states, rewards, actions, importance_scores)
        
        # If importance scores are provided, immediately add states to buffers
        if importance_scores is not None:
            self._process_trajectory_states(states, importance_scores)
        
        # Update statistics
        self.stats["n_trajectories"] = len(self.trajectory_collector.trajectories)
        self._update_stats()

    def _process_trajectory_states(self, states: List[np.ndarray], importance_scores: List[float]):
        """Process states from a trajectory and add them to appropriate buffers"""
        # Calculate threshold for this batch
        if len(importance_scores) > 0:
            threshold = np.percentile(importance_scores, self.importance_threshold_percentile)
            
            for state, score in zip(states, importance_scores):
                is_critical = score >= threshold
                self.add_state(
                    state=state,
                    is_critical=is_critical,
                    value=score,
                    metadata={"source": "trajectory", "importance_score": score}
                )

    def update_critical_states_from_trajectories(self):
        """
        Update critical states buffer from collected trajectories.
        This implements the critical state identification process from the RICE paper.
        """
        critical_states_data = self.trajectory_collector.get_critical_states(
            threshold_percentile=self.importance_threshold_percentile
        )
        
        for state, importance_score in critical_states_data:
            self.add_state(
                state=state,
                is_critical=True,
                value=importance_score,
                metadata={"source": "trajectory_analysis", "importance_score": importance_score}
            )
        
        self.logger.info(f"Updated critical states buffer with {len(critical_states_data)} states")

    def add_state(
        self,
        state: np.ndarray,
        is_critical: bool = False,
        value: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a new state to the appropriate buffer.
        
        Args:
            state: The state to add
            is_critical: Whether this is a critical state
            value: Value/importance of the state (higher values = more important)
            metadata: Optional metadata about the state
        """
        if metadata is None:
            metadata = {}
            
        state_info = StateInfo(
            state=np.array(state),
            is_critical=is_critical,
            value=value,
            metadata=metadata,
            timestamp=self.total_samples
        )
        
        if is_critical:
            self._add_critical_state(state_info)
        else:
            self._add_default_state(state_info)
            
        self._update_stats()

    def add_critical_state(self, state: np.ndarray, importance_score: float, metadata: Optional[Dict] = None):
        """
        Add a critical state with importance score (convenience method).
        
        Args:
            state: The critical state to add
            importance_score: Importance/value score of the state
            metadata: Optional metadata
        """
        self.add_state(
            state=state,
            is_critical=True,
            value=importance_score,
            metadata=metadata or {"source": "manual", "importance_score": importance_score}
        )

    def _add_critical_state(self, state_info: StateInfo) -> None:
        """Add state to critical buffer with value-based replacement policy"""
        if len(self.critical_states) >= self.max_buffer_size:
            # Remove lowest value state if buffer is full
            self.critical_states.sort(key=lambda x: x.value)
            removed_state = self.critical_states.pop(0)
            self.logger.debug(f"Removed critical state with value {removed_state.value}")
            
        self.critical_states.append(state_info)
        self.logger.debug(f"Added critical state with value {state_info.value}")

    def _add_default_state(self, state_info: StateInfo) -> None:
        """Add state to default buffer with random replacement policy"""
        if len(self.default_states) >= self.max_buffer_size:
            # Random replacement for default states
            idx = random.randint(0, len(self.default_states) - 1)
            self.default_states[idx] = state_info
        else:
            self.default_states.append(state_info)

    def _select_critical_state(self) -> StateInfo:
        """
        Select a critical state based on the configured selection method.
        
        Returns:
            Selected StateInfo object
        """
        if not self.critical_states:
            raise ValueError("No critical states available")
            
        if self.critical_state_selection == "weighted":
            # Value-weighted selection (higher values more likely)
            values = np.array([s.value for s in self.critical_states])
            # Add small epsilon to avoid division by zero and ensure positive weights
            values = np.maximum(values, 1e-8)
            # Normalize to probabilities
            probs = values / values.sum()
            idx = np.random.choice(len(self.critical_states), p=probs)
            
        elif self.critical_state_selection == "top_k":
            # Select from top-k most valuable states
            k = min(5, len(self.critical_states))
            sorted_states = sorted(self.critical_states, key=lambda x: x.value, reverse=True)
            idx = random.randint(0, k-1)
            selected_state = sorted_states[idx]
            # Find index in original list
            idx = self.critical_states.index(selected_state)
            
        else:  # uniform
            # Uniform random selection
            idx = random.randint(0, len(self.critical_states) - 1)
            
        selected_state = self.critical_states[idx]
        selected_state.access_count += 1
        return selected_state

    def sample_initial_state(self, env, pretrained_policy=None, mask_network=None, p_critical: Optional[float] = None) -> np.ndarray:
        """
        Sample initial state from mixed distribution according to RICE Algorithm 2.
        
        Algorithm:
        1. With probability p (mixing_ratio), sample from critical states
        2. With probability (1-p), sample from default initial distribution
        3. If selected buffer is empty, fall back to the other or environment reset
        
        Args:
            env: Environment instance with reset() and optionally reset_to_state() methods
            pretrained_policy: Optional pretrained policy (for compatibility)
            mask_network: Optional mask network (for compatibility)
            p_critical: Optional override for mixing ratio
            
        Returns:
            Initial state as numpy array
        """
        self.total_samples += 1
        
        # Use provided p_critical or default mixing_ratio
        mixing_prob = p_critical if p_critical is not None else self.mixing_ratio
        
        # Core RICE algorithm: sample based on mixing ratio p
        use_critical = random.random() < mixing_prob
        
        if use_critical and self.critical_states:
            # Sample from critical states
            try:
                critical_state = self._select_critical_state()
                
                # Try to reset environment to the critical state
                if hasattr(env, 'reset_to_state'):
                    state = env.reset_to_state(critical_state.state)
                    self.critical_samples += 1
                    self.logger.debug(f"Reset to critical state with value {critical_state.value}")
                    return state
                elif hasattr(env, 'set_state'):
                    env.set_state(critical_state.state)
                    state = self._get_current_state(env)
                    self.critical_samples += 1
                    self.logger.debug(f"Set to critical state with value {critical_state.value}")
                    return state
                elif hasattr(env, 'sim') and hasattr(env.sim, 'set_state'):
                    # For MuJoCo environments
                    env.sim.set_state(critical_state.state)
                    env.sim.forward()
                    state = env._get_obs()
                    self.critical_samples += 1
                    self.logger.debug(f"Set MuJoCo state with value {critical_state.value}")
                    return state
                else:
                    # Environment doesn't support state reset, fall back to default
                    self.logger.warning("Environment doesn't support state reset, falling back to default")
                    state = env.reset()
                    self.default_samples += 1
                    return state
                    
            except Exception as e:
                self.logger.warning(f"Failed to reset to critical state: {e}, falling back to default")
                state = env.reset()
                self.default_samples += 1
                return state
                
        else:
            # Sample from default initial distribution
            state = env.reset()
            self.default_samples += 1
            self.logger.debug("Sampled from default initial distribution")
            return state

    def _get_current_state(self, env) -> np.ndarray:
        """Get current state from environment (helper method)"""
        if hasattr(env, '_get_obs'):
            return env._get_obs()
        elif hasattr(env, 'get_state'):
            return env.get_state()
        else:
            # Try to get observation
            try:
                return env.observation_space.sample()  # Fallback
            except:
                raise ValueError("Cannot get current state from environment")

    def sample(self, batch_size: int = 1) -> np.ndarray:
        """
        Sample states according to the mixed distribution (batch version).
        
        Args:
            batch_size: Number of states to sample
            
        Returns:
            Batch of sampled states as numpy array
        """
        # Determine number of each type to sample based on mixing ratio
        n_critical = int(batch_size * self.mixing_ratio)
        n_default = batch_size - n_critical
        
        sampled_states = []
        
        # Sample critical states
        if n_critical > 0 and self.critical_states:
            for _ in range(n_critical):
                try:
                    critical_state = self._select_critical_state()
                    sampled_states.append(critical_state.state)
                except ValueError:
                    # No critical states available, will fill from default
                    pass
                
        # Sample default states
        if n_default > 0 and self.default_states:
            default_indices = np.random.choice(
                len(self.default_states),
                size=min(n_default, len(self.default_states)),
                replace=True
            )
            default_samples = [self.default_states[i].state for i in default_indices]
            sampled_states.extend(default_samples)
            
        # Handle case where buffers are insufficient
        remaining = batch_size - len(sampled_states)
        if remaining > 0:
            # Fill remaining slots from available states
            available_states = self.critical_states + self.default_states
            if available_states:
                indices = np.random.choice(len(available_states), size=remaining, replace=True)
                samples = [available_states[i].state for i in indices]
                sampled_states.extend(samples)
            else:
                raise ValueError("No states available to sample from")
                
        if not sampled_states:
            raise ValueError("No states available to sample from")
            
        return np.stack(sampled_states)

    def evaluate_state_importance(self, state: np.ndarray, policy, mask_network=None, 
                                 explanation_method: Optional[Callable] = None) -> float:
        """
        Evaluate the importance of a state using explanation methods.
        
        Args:
            state: State to evaluate
            policy: Policy network
            mask_network: Optional mask network for StateMask
            explanation_method: Optional custom explanation method
            
        Returns:
            Importance score for the state
        """
        if explanation_method is not None:
            return explanation_method(state, policy, mask_network)
        
        # Default importance evaluation (can be replaced with specific explanation methods)
        try:
            # Simple value-based importance (using policy value if available)
            if hasattr(policy, 'predict_values'):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    value = policy.predict_values(state_tensor).item()
                    return abs(value)  # Use absolute value as importance
            else:
                # Fallback: use action probability entropy as importance
                if hasattr(policy, 'predict'):
                    action, _ = policy.predict(state, deterministic=False)
                    # Use action magnitude as proxy for importance
                    return np.linalg.norm(action)
                else:
                    return 0.0
        except Exception as e:
            self.logger.warning(f"Failed to evaluate state importance: {e}")
            return 0.0

    def _update_stats(self) -> None:
        """Update internal statistics"""
        self.stats["n_default_states"] = len(self.default_states)
        self.stats["n_critical_states"] = len(self.critical_states)
        
        if self.default_states:
            self.stats["avg_default_value"] = np.mean([s.value for s in self.default_states])
        else:
            self.stats["avg_default_value"] = 0.0
            
        if self.critical_states:
            self.stats["avg_critical_value"] = np.mean([s.value for s in self.critical_states])
        else:
            self.stats["avg_critical_value"] = 0.0
            
        if self.total_samples > 0:
            self.stats["critical_sample_ratio"] = self.critical_samples / self.total_samples
        else:
            self.stats["critical_sample_ratio"] = 0.0
            
        total_capacity = 2 * self.max_buffer_size
        current_usage = len(self.default_states) + len(self.critical_states)
        self.stats["buffer_utilization"] = current_usage / total_capacity

    def get_stats(self) -> Dict:
        """Get current statistics about the state distribution"""
        self._update_stats()
        return {
            **self.stats,
            "total_samples": self.total_samples,
            "critical_samples": self.critical_samples,
            "default_samples": self.default_samples,
            "mixing_ratio": self.mixing_ratio,
            "critical_state_selection": self.critical_state_selection,
            "importance_threshold_percentile": self.importance_threshold_percentile
        }

    def update_mixing_ratio(self, new_ratio: float) -> None:
        """Update the mixing ratio p"""
        if not 0 <= new_ratio <= 1:
            raise ValueError("Mixing ratio must be between 0 and 1")
        self.mixing_ratio = new_ratio
        self.logger.info(f"Updated mixing ratio to {new_ratio}")

    def update_importance_threshold(self, new_threshold: float) -> None:
        """Update the importance threshold percentile"""
        if not 0 <= new_threshold <= 100:
            raise ValueError("Importance threshold percentile must be between 0 and 100")
        self.importance_threshold_percentile = new_threshold
        self.logger.info(f"Updated importance threshold to {new_threshold}th percentile")

    def clear_buffers(self) -> None:
        """Clear all state buffers"""
        self.default_states.clear()
        self.critical_states.clear()
        self.trajectory_collector = TrajectoryCollector(max_trajectories=1000)
        self.logger.info("Cleared all state buffers and trajectory collector")

    def get_critical_state_values(self) -> np.ndarray:
        """Get values of all critical states for analysis"""
        if not self.critical_states:
            return np.array([])
        return np.array([s.value for s in self.critical_states])

    def get_default_state_values(self) -> np.ndarray:
        """Get values of all default states for analysis"""
        if not self.default_states:
            return np.array([])
        return np.array([s.value for s in self.default_states])

    def get_state_distribution_info(self) -> Dict:
        """Get detailed information about the state distribution"""
        critical_values = self.get_critical_state_values()
        default_values = self.get_default_state_values()
        
        return {
            "critical_states": {
                "count": len(self.critical_states),
                "values": critical_values.tolist() if len(critical_values) > 0 else [],
                "mean_value": np.mean(critical_values) if len(critical_values) > 0 else 0.0,
                "std_value": np.std(critical_values) if len(critical_values) > 0 else 0.0,
                "min_value": np.min(critical_values) if len(critical_values) > 0 else 0.0,
                "max_value": np.max(critical_values) if len(critical_values) > 0 else 0.0
            },
            "default_states": {
                "count": len(self.default_states),
                "values": default_values.tolist() if len(default_values) > 0 else [],
                "mean_value": np.mean(default_values) if len(default_values) > 0 else 0.0,
                "std_value": np.std(default_values) if len(default_values) > 0 else 0.0,
                "min_value": np.min(default_values) if len(default_values) > 0 else 0.0,
                "max_value": np.max(default_values) if len(default_values) > 0 else 0.0
            },
            "sampling_stats": self.get_stats()
        }

    def save(self, path: str) -> None:
        """Save the state distribution to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_data = {
            "default_states": self.default_states,
            "critical_states": self.critical_states,
            "stats": self.stats,
            "total_samples": self.total_samples,
            "critical_samples": self.critical_samples,
            "default_samples": self.default_samples,
            "mixing_ratio": self.mixing_ratio,
            "max_buffer_size": self.max_buffer_size,
            "state_dim": self.state_dim,
            "critical_state_selection": self.critical_state_selection,
            "importance_threshold_percentile": self.importance_threshold_percentile,
            "trajectory_collector": self.trajectory_collector
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        self.logger.info(f"Saved state distribution to {path}")

    def load(self, path: str) -> None:
        """Load the state distribution from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.default_states = data["default_states"]
        self.critical_states = data["critical_states"]
        self.stats = data["stats"]
        self.total_samples = data["total_samples"]
        self.critical_samples = data.get("critical_samples", 0)
        self.default_samples = data.get("default_samples", 0)
        self.mixing_ratio = data["mixing_ratio"]
        self.max_buffer_size = data["max_buffer_size"]
        self.state_dim = data["state_dim"]
        self.critical_state_selection = data.get("critical_state_selection", "weighted")
        self.importance_threshold_percentile = data.get("importance_threshold_percentile", 80.0)
        self.trajectory_collector = data.get("trajectory_collector", TrajectoryCollector(max_trajectories=1000))
        
        self.logger.info(f"Loaded state distribution from {path}")

    def reset_sampling_stats(self) -> None:
        """Reset sampling statistics (useful for evaluation)"""
        self.total_samples = 0
        self.critical_samples = 0
        self.default_samples = 0
        for state_info in self.critical_states + self.default_states:
            state_info.access_count = 0
        self.logger.info("Reset sampling statistics")