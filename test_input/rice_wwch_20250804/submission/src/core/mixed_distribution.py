"""
Mixed Initial State Distribution Module

This module implements the mixed initial state distribution mechanism described in the RICE paper,
which combines default initial states with critical states identified through explanation methods.
The mixing helps prevent overfitting while enabling effective exploration.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import random
import pickle
import os

@dataclass
class StateInfo:
    """Data class for storing state information with metadata"""
    state: np.ndarray  # The actual state vector/array
    is_critical: bool  # Whether this is a critical state
    value: float  # Value/importance of the state
    metadata: Dict  # Additional metadata (e.g., episode info, timestamp)

class MixedStateDistribution:
    """
    Implements mixed initial state distribution that combines default initial states
    with critical states identified through explanation methods.
    
    Key Features:
    - Maintains separate buffers for default and critical states
    - Implements configurable mixing ratio p
    - Supports weighted sampling based on state values/importance
    - Handles state persistence and loading
    - Identifies critical states using mask network importance scoring
    """
    
    def __init__(
        self,
        mixing_ratio: float = 0.25,  # p parameter controlling mix of critical vs default states
        max_buffer_size: int = 10000,
        state_dim: Union[int, Tuple[int, ...]] = None,
        device: str = "cpu",
        importance_threshold: float = 0.7  # Threshold for critical state selection
    ):
        """
        Initialize the mixed state distribution.
        
        Args:
            mixing_ratio: Float between 0 and 1 controlling ratio of critical states vs default states
            max_buffer_size: Maximum number of states to store in each buffer
            state_dim: Dimensions of the state space
            device: Device to store tensors on
            importance_threshold: Threshold for selecting critical states based on importance score
        """
        self.mixing_ratio = mixing_ratio
        self.max_buffer_size = max_buffer_size
        self.state_dim = state_dim
        self.device = device
        self.importance_threshold = importance_threshold
        
        # Initialize state buffers
        self.default_states: List[StateInfo] = []
        self.critical_states: List[StateInfo] = []
        
        # Statistics
        self.total_samples = 0
        self.stats = {
            "n_default_states": 0,
            "n_critical_states": 0,
            "avg_default_value": 0.0,
            "avg_critical_value": 0.0,
            "critical_states_found": 0,
            "trajectories_analyzed": 0
        }

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
            value: Value/importance of the state
            metadata: Optional metadata about the state
        """
        if metadata is None:
            metadata = {}
            
        state_info = StateInfo(
            state=np.array(state),
            is_critical=is_critical,
            value=value,
            metadata=metadata
        )
        
        if is_critical:
            if len(self.critical_states) >= self.max_buffer_size:
                # Remove lowest value state if buffer is full
                self.critical_states.sort(key=lambda x: x.value)
                self.critical_states.pop(0)
            self.critical_states.append(state_info)
            self.stats["n_critical_states"] = len(self.critical_states)
            if self.critical_states:
                self.stats["avg_critical_value"] = np.mean([s.value for s in self.critical_states])
        else:
            if len(self.default_states) >= self.max_buffer_size:
                # Random replacement for default states
                idx = random.randint(0, len(self.default_states) - 1)
                self.default_states[idx] = state_info
            else:
                self.default_states.append(state_info)
            self.stats["n_default_states"] = len(self.default_states)
            if self.default_states:
                self.stats["avg_default_value"] = np.mean([s.value for s in self.default_states])

    def find_critical_states(
        self,
        trajectories: List[Dict],
        mask_network: torch.nn.Module,
        pretrained_policy: torch.nn.Module,
        top_k: int = 100
    ) -> List[StateInfo]:
        """
        Identify critical states from trajectories using mask network importance scoring.
        
        This implements the core algorithm from RICE paper:
        1. For each state in trajectory, compute importance score as 1 - P(a^m=1|s)
        2. Select states with highest importance scores as critical states
        3. Add critical states to the buffer for mixed sampling
        
        Args:
            trajectories: List of trajectory dictionaries containing states and actions
            mask_network: Trained mask network for computing importance scores
            pretrained_policy: Pretrained policy for action probability computation
            top_k: Number of top critical states to select
            
        Returns:
            List of critical StateInfo objects
        """
        if not trajectories:
            return []
            
        mask_network.eval()
        pretrained_policy.eval()
        
        all_state_scores = []
        
        with torch.no_grad():
            for traj_idx, trajectory in enumerate(trajectories):
                states = trajectory.get('states', [])
                actions = trajectory.get('actions', [])
                
                if len(states) == 0 or len(actions) == 0:
                    continue
                    
                # Convert states to tensor
                if isinstance(states[0], np.ndarray):
                    state_tensor = torch.FloatTensor(np.stack(states)).to(self.device)
                else:
                    state_tensor = torch.FloatTensor(states).to(self.device)
                
                # Convert actions to tensor
                if isinstance(actions[0], np.ndarray):
                    action_tensor = torch.LongTensor(np.stack(actions)).to(self.device)
                else:
                    action_tensor = torch.LongTensor(actions).to(self.device)
                
                # Compute mask probabilities for each state-action pair
                # P(a^m=1|s) represents probability of masking the action
                mask_logits = mask_network(state_tensor)
                
                # Handle different mask network output formats
                if mask_logits.dim() == 3:  # [batch, seq_len, action_dim]
                    mask_logits = mask_logits.squeeze(1)
                elif mask_logits.dim() == 1:  # [batch]
                    mask_logits = mask_logits.unsqueeze(-1)
                
                # Get mask probabilities for the actual actions taken
                if action_tensor.dim() == 1:
                    action_tensor = action_tensor.unsqueeze(-1)
                
                # Compute P(a^m=1|s) for each state-action pair
                mask_probs = torch.sigmoid(mask_logits)
                
                # If mask_probs has action dimension, gather probabilities for actual actions
                if mask_probs.size(-1) > 1:
                    action_mask_probs = torch.gather(mask_probs, -1, action_tensor).squeeze(-1)
                else:
                    action_mask_probs = mask_probs.squeeze(-1)
                
                # Compute importance scores: 1 - P(a^m=1|s)
                # Higher importance means lower probability of masking (more critical)
                importance_scores = 1.0 - action_mask_probs
                
                # Store state-score pairs
                for i, (state, score) in enumerate(zip(states, importance_scores)):
                    all_state_scores.append({
                        'state': np.array(state),
                        'importance': float(score.cpu()),
                        'trajectory_idx': traj_idx,
                        'step_idx': i
                    })
        
        # Sort by importance score (descending)
        all_state_scores.sort(key=lambda x: x['importance'], reverse=True)
        
        # Select top-k critical states
        critical_states = []
        selected_count = 0
        
        for state_data in all_state_scores:
            if selected_count >= top_k:
                break
                
            # Only select states above importance threshold
            if state_data['importance'] >= self.importance_threshold:
                state_info = StateInfo(
                    state=state_data['state'],
                    is_critical=True,
                    value=state_data['importance'],
                    metadata={
                        'trajectory_idx': state_data['trajectory_idx'],
                        'step_idx': state_data['step_idx'],
                        'importance_score': state_data['importance']
                    }
                )
                critical_states.append(state_info)
                selected_count += 1
        
        # Add critical states to buffer
        for state_info in critical_states:
            self.add_state(
                state=state_info.state,
                is_critical=True,
                value=state_info.value,
                metadata=state_info.metadata
            )
        
        # Update statistics
        self.stats["critical_states_found"] += len(critical_states)
        self.stats["trajectories_analyzed"] += len(trajectories)
        
        return critical_states

    def sample_initial_state(
        self, 
        env, 
        pretrained_policy=None, 
        mask_network=None
    ) -> np.ndarray:
        """
        Sample initial state from mixed distribution according to RICE algorithm.
        
        This implements the mixed sampling strategy:
        - With probability p (mixing_ratio): sample from critical states
        - With probability 1-p: sample from default distribution (env.reset())
        
        Args:
            env: Environment to reset for default states
            pretrained_policy: Pretrained policy (for compatibility)
            mask_network: Mask network (for compatibility)
            
        Returns:
            Initial state as numpy array
        """
        if random.random() < self.mixing_ratio:
            # Sample from critical states with probability p
            if len(self.critical_states) > 0:
                # Value-weighted sampling from critical states
                values = np.array([s.value for s in self.critical_states])
                if values.sum() > 0:
                    probs = values / values.sum()
                    idx = np.random.choice(len(self.critical_states), p=probs)
                    state_info = self.critical_states[idx]
                    return state_info.state.copy()
                else:
                    # Uniform sampling if all values are zero
                    state_info = random.choice(self.critical_states)
                    return state_info.state.copy()
            else:
                # Fallback to default if no critical states available
                return env.reset()
        else:
            # Sample from default distribution with probability 1-p
            return env.reset()

    def sample(self, batch_size: int = 1) -> np.ndarray:
        """
        Sample states according to the mixed distribution.
        
        Args:
            batch_size: Number of states to sample
            
        Returns:
            Batch of sampled states as numpy array
        """
        self.total_samples += batch_size
        
        # Determine number of each type to sample
        n_critical = int(batch_size * self.mixing_ratio)
        n_default = batch_size - n_critical
        
        sampled_states = []
        
        # Sample critical states with value-weighted probability
        if n_critical > 0 and self.critical_states:
            values = np.array([s.value for s in self.critical_states])
            if values.sum() > 0:
                probs = values / values.sum()
                critical_indices = np.random.choice(
                    len(self.critical_states),
                    size=n_critical,
                    p=probs,
                    replace=True
                )
            else:
                critical_indices = np.random.choice(
                    len(self.critical_states),
                    size=n_critical,
                    replace=True
                )
            critical_samples = [self.critical_states[i].state for i in critical_indices]
            sampled_states.extend(critical_samples)
            
        # Sample default states uniformly
        if n_default > 0 and self.default_states:
            default_indices = np.random.choice(
                len(self.default_states),
                size=n_default,
                replace=True
            )
            default_samples = [self.default_states[i].state for i in default_indices]
            sampled_states.extend(default_samples)
            
        # If either buffer is empty, sample from the other
        remaining = batch_size - len(sampled_states)
        if remaining > 0:
            source = self.critical_states if self.critical_states else self.default_states
            if source:
                indices = np.random.choice(len(source), size=remaining, replace=True)
                samples = [source[i].state for i in indices]
                sampled_states.extend(samples)
                
        if not sampled_states:
            raise ValueError("No states available to sample from")
            
        return np.stack(sampled_states)

    def update_critical_states_from_trajectories(
        self,
        trajectories: List[Dict],
        mask_network: torch.nn.Module,
        pretrained_policy: torch.nn.Module,
        top_k: int = 100
    ) -> None:
        """
        Update critical states buffer by analyzing new trajectories.
        
        This is a convenience method that calls find_critical_states and
        automatically updates the internal buffer.
        
        Args:
            trajectories: New trajectories to analyze
            mask_network: Trained mask network
            pretrained_policy: Pretrained policy
            top_k: Number of top critical states to select
        """
        self.find_critical_states(
            trajectories=trajectories,
            mask_network=mask_network,
            pretrained_policy=pretrained_policy,
            top_k=top_k
        )

    def get_stats(self) -> Dict:
        """Get current statistics about the state distribution"""
        return {
            **self.stats,
            "total_samples": self.total_samples,
            "mixing_ratio": self.mixing_ratio,
            "importance_threshold": self.importance_threshold
        }

    def save(self, path: str) -> None:
        """Save the state distribution to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_data = {
            "default_states": self.default_states,
            "critical_states": self.critical_states,
            "stats": self.stats,
            "total_samples": self.total_samples,
            "mixing_ratio": self.mixing_ratio,
            "max_buffer_size": self.max_buffer_size,
            "state_dim": self.state_dim,
            "importance_threshold": self.importance_threshold
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, path: str) -> None:
        """Load the state distribution from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.default_states = data["default_states"]
        self.critical_states = data["critical_states"]
        self.stats = data["stats"]
        self.total_samples = data["total_samples"]
        self.mixing_ratio = data["mixing_ratio"]
        self.max_buffer_size = data["max_buffer_size"]
        self.state_dim = data["state_dim"]
        self.importance_threshold = data.get("importance_threshold", 0.7)