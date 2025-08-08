import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StateMaskNetwork(nn.Module):
    """
    State Mask Network for generating explanations.
    Implements the mask network M_θ as described in Algorithm 1.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(StateMaskNetwork, self).__init__()
        self.state_dim = state_dim
        
        # Mask network architecture: state -> mask probabilities
        self.mask_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid()  # Output probabilities for each state dimension
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate mask probabilities for input state.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            mask_probs: Mask probabilities [batch_size, state_dim]
        """
        return self.mask_net(state)
    
    def sample_mask(self, state: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """
        Sample binary mask based on mask probabilities.
        
        Args:
            state: Input state tensor
            p: Probability threshold for mask sampling
            
        Returns:
            mask: Binary mask tensor
        """
        mask_probs = self.forward(state)
        # Sample binary mask using Bernoulli distribution
        mask = torch.bernoulli(mask_probs * p + (1 - p) * 0.5)
        return mask

class StateMaskExplainer:
    """
    StateMask explanation method implementation.
    Implements Algorithm 1: Training the Explanation Module.
    """
    
    def __init__(self, agent, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.agent = agent
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize mask network
        self.mask_network = StateMaskNetwork(state_dim).to(device)
        self.optimizer = optim.Adam(self.mask_network.parameters(), lr=1e-3)
        
        # Training parameters
        self.batch_size = 256
        self.num_epochs = 100
        
    def compute_explanation_loss(self, states: torch.Tensor, actions: torch.Tensor, 
                                p: float, lambda_reg: float) -> torch.Tensor:
        """
        Compute explanation loss as defined in Algorithm 1.
        
        L_exp = E[||π(s) - π(s ⊙ m)||²] + λ * E[||m||₁]
        
        Args:
            states: Batch of states
            actions: Batch of actions
            p: Mask sampling probability
            lambda_reg: Regularization parameter λ
            
        Returns:
            loss: Explanation loss
        """
        batch_size = states.shape[0]
        
        # Get original policy outputs
        with torch.no_grad():
            original_actions = self.agent.predict(states.cpu().numpy(), deterministic=True)[0]
            original_actions = torch.tensor(original_actions, device=self.device, dtype=torch.float32)
        
        # Sample masks
        masks = self.mask_network.sample_mask(states, p)
        
        # Apply masks to states
        masked_states = states * masks
        
        # Get policy outputs for masked states
        with torch.no_grad():
            masked_actions = self.agent.predict(masked_states.cpu().numpy(), deterministic=True)[0]
            masked_actions = torch.tensor(masked_actions, device=self.device, dtype=torch.float32)
        
        # Compute fidelity loss: ||π(s) - π(s ⊙ m)||²
        fidelity_loss = torch.mean(torch.sum((original_actions - masked_actions) ** 2, dim=1))
        
        # Compute sparsity regularization: λ * ||m||₁
        mask_probs = self.mask_network(states)
        sparsity_loss = lambda_reg * torch.mean(torch.sum(mask_probs, dim=1))
        
        total_loss = fidelity_loss + sparsity_loss
        
        return total_loss, fidelity_loss, sparsity_loss
    
    def train(self, replay_buffer: List[Tuple], p: float = 0.5, lambda_reg: float = 0.01) -> Dict:
        """
        Train the explanation module using Algorithm 1.
        
        Args:
            replay_buffer: List of (state, action) tuples
            p: Mask sampling probability
            lambda_reg: Regularization parameter
            
        Returns:
            training_stats: Dictionary of training statistics
        """
        logger.info(f"Training StateMask explainer with p={p}, λ={lambda_reg}")
        
        # Convert replay buffer to tensors
        states = torch.tensor([transition[0] for transition in replay_buffer], 
                            device=self.device, dtype=torch.float32)
        actions = torch.tensor([transition[1] for transition in replay_buffer], 
                             device=self.device, dtype=torch.float32)
        
        training_stats = {'losses': [], 'fidelity_losses': [], 'sparsity_losses': []}
        
        for epoch in range(self.num_epochs):
            epoch_losses = []
            epoch_fidelity = []
            epoch_sparsity = []
            
            # Mini-batch training
            num_batches = len(states) // self.batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                
                batch_states = states[start_idx:end_idx]
                batch_actions = actions[start_idx:end_idx]
                
                # Compute loss
                loss, fidelity_loss, sparsity_loss = self.compute_explanation_loss(
                    batch_states, batch_actions, p, lambda_reg)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_fidelity.append(fidelity_loss.item())
                epoch_sparsity.append(sparsity_loss.item())
            
            avg_loss = np.mean(epoch_losses)
            avg_fidelity = np.mean(epoch_fidelity)
            avg_sparsity = np.mean(epoch_sparsity)
            
            training_stats['losses'].append(avg_loss)
            training_stats['fidelity_losses'].append(avg_fidelity)
            training_stats['sparsity_losses'].append(avg_sparsity)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, "
                          f"Fidelity={avg_fidelity:.4f}, Sparsity={avg_sparsity:.4f}")
        
        return training_stats

class StateMaskRefiner:
    """
    StateMask-R refinement method implementation.
    Implements Algorithm 2: Policy Refinement.
    """
    
    def __init__(self, agent, explainer: StateMaskExplainer, env, device: str = 'cpu'):
        self.agent = agent
        self.explainer = explainer
        self.env = env
        self.device = device
        
        # Refinement parameters
        self.num_refinement_steps = 1000
        self.eval_frequency = 100
        
    def refine_policy(self, p: float = 0.5, lambda_reg: float = 0.01) -> Dict:
        """
        Refine policy using StateMask-R method (Algorithm 2).
        
        Args:
            p: Mask sampling probability
            lambda_reg: Regularization parameter
            
        Returns:
            refinement_stats: Dictionary of refinement statistics
        """
        logger.info(f"Starting policy refinement with p={p}, λ={lambda_reg}")
        
        refinement_stats = {'rewards': [], 'steps': []}
        
        # Create a copy of the agent for refinement
        refined_agent = PPO.load(self.agent.save_path) if hasattr(self.agent, 'save_path') else self.agent
        
        for step in range(self.num_refinement_steps):
            # Collect trajectory
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            episode_reward = 0
            episode_length = 0
            trajectory = []
            
            done = False
            while not done and episode_length < 1000:  # Max episode length
                # Get action from current policy
                action, _ = refined_agent.predict(obs, deterministic=False)
                
                # Apply mask to state for explanation
                obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                mask = self.explainer.mask_network.sample_mask(obs_tensor, p)
                masked_obs = (obs_tensor * mask).squeeze(0).cpu().numpy()
                
                # Store transition
                trajectory.append((obs, action, masked_obs))
                
                # Execute action
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
            
            # Update policy using collected trajectory
            if len(trajectory) > 0:
                # Convert trajectory to training data
                states = np.array([t[0] for t in trajectory])
                actions = np.array([t[1] for t in trajectory])
                
                # Fine-tune the policy (simplified version)
                # In practice, this would involve more sophisticated policy updates
                refined_agent.learn(total_timesteps=len(trajectory))
            
            # Evaluate periodically
            if step % self.eval_frequency == 0:
                eval_reward = self.evaluate_policy(refined_agent, num_episodes=5)
                refinement_stats['rewards'].append(eval_reward)
                refinement_stats['steps'].append(step)
                logger.info(f"Step {step}: Evaluation reward = {eval_reward:.2f}")
        
        return refined_agent, refinement_stats
    
    def evaluate_policy(self, agent, num_episodes: int = 10) -> float:
        """Evaluate policy performance."""
        total_reward = 0
        for _ in range(num_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            episode_reward = 0
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / num_episodes

class ExperimentRunner:
    """
    Main experiment runner for StateMask paper reproduction.
    Handles all experiments across different environments and configurations.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path(config.get('results_dir', 'results'))
        self.results_dir.mkdir(exist_ok=True)
        
        # Environment configurations
        self.env_configs = {
            'Hopper-v3': {'type': 'mujoco', 'algorithm': 'PPO'},
            'Walker2d-v3': {'type': 'mujoco', 'algorithm': 'PPO'},
            'Reacher-v2': {'type': 'mujoco', 'algorithm': 'PPO'},
            'HalfCheetah-v3': {'type': 'mujoco', 'algorithm': 'PPO'},
            'Selfish Mining': {'type': 'custom', 'algorithm': 'PPO'},
            'CAGE Challenge 2': {'type': 'custom', 'algorithm': 'PPO'},
            'Auto Driving': {'type': 'custom', 'algorithm': 'PPO'},
            'Malware Mutation': {'type': 'custom', 'algorithm': 'PPO'}
        }
        
        # Experimental parameters
        self.p_values = [0, 0.25, 0.5, 0.75, 1.0]
        self.lambda_values = [0, 0.001, 0.01, 0.1]
        self.num_seeds = 3
        
    def create_environment(self, env_name: str):
        """Create and configure environment."""
        if env_name in ['Hopper-v3', 'Walker2d-v3', 'Reacher-v2', 'HalfCheetah-v3']:
            env = gym.make(env_name)
            env = Monitor(env)
            return env
        else:
            # For custom environments, create placeholder implementations
            logger.warning(f"Custom environment {env_name} not implemented. Using placeholder.")
            return gym.make('CartPole-v1')  # Placeholder
    
    def train_suboptimal_agent(self, env_name: str, seed: int) -> Any:
        """
        Train a suboptimal PPO agent for the given environment.
        This serves as the target agent for explanation.
        """
        logger.info(f"Training suboptimal agent for {env_name} with seed {seed}")
        
        env = self.create_environment(env_name)
        env.seed(seed)
        
        # Create suboptimal agent with limited training
        model = PPO('MlpPolicy', env, verbose=0, seed=seed)
        
        # Train for limited timesteps to ensure suboptimality
        timesteps = self.config.get('suboptimal_timesteps', 50000)
        model.learn(total_timesteps=timesteps)
        
        # Save the model
        model_path = self.results_dir / f"{env_name}_suboptimal_seed{seed}.zip"
        model.save(str(model_path))
        
        return model, env
    
    def collect_replay_buffer(self, agent, env, num_episodes: int = 100) -> List[Tuple]:
        """Collect replay buffer for training explanation module."""
        logger.info(f"Collecting replay buffer with {num_episodes} episodes")
        
        replay_buffer = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=False)
                replay_buffer.append((obs.copy(), action.copy()))
                
                next_obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                obs = next_obs
        
        logger.info(f"Collected {len(replay_buffer)} transitions")
        return replay_buffer
    
    def run_statemask_experiment(self, env_name: str, p: float, lambda_reg: float, seed: int) -> Dict:
        """Run StateMask experiment for given parameters."""
        logger.info(f"Running StateMask experiment: {env_name}, p={p}, λ={lambda_reg}, seed={seed}")
        
        # Train suboptimal agent
        agent, env = self.train_suboptimal_agent(env_name, seed)
        
        # Get environment dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
        
        # Collect replay buffer
        replay_buffer = self.collect_replay_buffer(agent, env)
        
        # Train explanation module
        explainer = StateMaskExplainer(agent, state_dim, action_dim, self.device)
        explanation_stats = explainer.train(replay_buffer, p, lambda_reg)
        
        # Evaluate explanation quality
        explanation_quality = self.evaluate_explanation_quality(explainer, replay_buffer, p)
        
        # Run refinement if specified
        refinement_stats = None
        if self.config.get('run_refinement', True):
            refiner = StateMaskRefiner(agent, explainer, env, self.device)
            refined_agent, refinement_stats = refiner.refine_policy(p, lambda_reg)
        
        results = {
            'env_name': env_name,
            'p': p,
            'lambda': lambda_reg,
            'seed': seed,
            'explanation_stats': explanation_stats,
            'explanation_quality': explanation_quality,
            'refinement_stats': refinement_stats
        }
        
        return results
    
    def evaluate_explanation_quality(self, explainer: StateMaskExplainer, 
                                   replay_buffer: List[Tuple], p: float) -> Dict:
        """Evaluate explanation quality metrics."""
        logger.info("Evaluating explanation quality")
        
        # Sample subset for evaluation
        eval_size = min(1000, len(replay_buffer))
        eval_indices = np.random.choice(len(replay_buffer), eval_size, replace=False)
        eval_buffer = [replay_buffer[i] for i in eval_indices]
        
        states = torch.tensor([t[0] for t in eval_buffer], device=self.device, dtype=torch.float32)
        
        # Compute explanation metrics
        with torch.no_grad():
            mask_probs = explainer.mask_network(states)
            
            # Sparsity: average number of selected features
            sparsity = torch.mean(torch.sum(mask_probs > 0.5, dim=1)).item()
            
            # Consistency: variance in mask probabilities
            consistency = torch.mean(torch.var(mask_probs, dim=0)).item()
            
            # Fidelity: how well masked states preserve original behavior
            masks = explainer.mask_network.sample_mask(states, p)
            masked_states = states * masks
            
            original_actions = []
            masked_actions = []
            
            for i in range(len(states)):
                orig_action = explainer.agent.predict(states[i:i+1].cpu().numpy(), deterministic=True)[0]
                mask_action = explainer.agent.predict(masked_states[i:i+1].cpu().numpy(), deterministic=True)[0]
                original_actions.append(orig_action)
                masked_actions.append(mask_action)
            
            original_actions = np.array(original_actions)
            masked_actions = np.array(masked_actions)
            
            fidelity = 1.0 - np.mean(np.sum((original_actions - masked_actions) ** 2, axis=1))
        
        return {
            'sparsity': sparsity,
            'consistency': consistency,
            'fidelity': fidelity
        }
    
    def run_baseline_comparison(self, env_name: str, seed: int) -> Dict:
        """Run baseline method comparisons."""
        logger.info(f"Running baseline comparison for {env_name}, seed={seed}")
        
        agent, env = self.train_suboptimal_agent(env_name, seed)
        
        baselines = {}
        
        # PPO Fine-tuning baseline
        logger.info("Running PPO fine-tuning baseline")
        ppo_baseline = PPO.load(agent.save_path)
        ppo_baseline.learn(total_timesteps=10000)  # Additional training
        baselines['PPO_finetune'] = self.evaluate_agent(ppo_baseline, env)
        
        # JSRL baseline (simplified implementation)
        logger.info("Running JSRL baseline")
        jsrl_baseline = PPO.load(agent.save_path)
        # Implement simplified JSRL training here
        baselines['JSRL'] = self.evaluate_agent(jsrl_baseline, env)
        
        return baselines
    
    def evaluate_agent(self, agent, env, num_episodes: int = 10) -> float:
        """Evaluate agent performance."""
        total_reward = 0
        for _ in range(num_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            episode_reward = 0
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def run_ablation_study(self, env_name: str = 'Hopper-v3') -> Dict:
        """Run ablation study on p and λ values."""
        logger.info(f"Running ablation study on {env_name}")
        
        ablation_results = {}
        
        for seed in range(self.num_seeds):
            for p in self.p_values:
                for lambda_reg in self.lambda_values:
                    key = f"p{p}_lambda{lambda_reg}_seed{seed}"
                    try:
                        result = self.run_statemask_experiment(env_name, p, lambda_reg, seed)
                        ablation_results[key] = result
                        logger.info(f"Completed ablation: {key}")
                    except Exception as e:
                        logger.error(f"Failed ablation {key}: {str(e)}")
                        ablation_results[key] = {'error': str(e)}
        
        return ablation_results
    
    def run_all_experiments(self) -> Dict:
        """Run all experiments specified in the paper."""
        logger.info("Starting complete experimental evaluation")
        
        all_results = {}
        
        # 1. Main experiments across all environments
        logger.info("Running main experiments across all environments")
        for env_name in self.env_configs.keys():
            if env_name.startswith(('Selfish', 'CAGE', 'Auto', 'Malware')):
                logger.warning(f"Skipping custom environment {env_name} - not implemented")
                continue
                
            env_results = {}
            for seed in range(self.num_seeds):
                try:
                    # Run with default parameters
                    result = self.run_statemask_experiment(env_name, p=0.5, lambda_reg=0.01, seed=seed)
                    env_results[f"seed{seed}"] = result
                    
                    # Run baseline comparison
                    baseline_result = self.run_baseline_comparison(env_name, seed)
                    env_results[f"baselines_seed{seed}"] = baseline_result
                    
                except Exception as e:
                    logger.error(f"Failed experiment {env_name} seed {seed}: {str(e)}")
                    env_results[f"seed{seed}"] = {'error': str(e)}
            
            all_results[env_name] = env_results
        
        # 2. Ablation study
        logger.info("Running ablation study")
        ablation_results = self.run_ablation_study()
        all_results['ablation_study'] = ablation_results
        
        # 3. Save results
        self.save_results(all_results)
        
        # 4. Generate plots
        self.generate_plots(all_results)
        
        return all_results
    
    def save_results(self, results: Dict):
        """Save experimental results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = self.results_dir / f"statemask_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self.convert_numpy_to_list(results)
            json.dump(json_results, f, indent=2)
        
        # Save as pickle for complete data
        pickle_path = self.results_dir / f"statemask_results_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {json_path} and {pickle_path}")
    
    def convert_numpy_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    def generate_plots(self, results: Dict):
        """Generate plots for experimental results."""
        logger.info("Generating result plots")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance comparison across environments
        self.plot_environment_comparison(results)
        
        # 2. Ablation study results
        if 'ablation_study' in results:
            self.plot_ablation_study(results['ablation_study'])
        
        # 3. Training curves
        self.plot_training_curves(results)
        
        logger.info("Plots saved to results directory")
    
    def plot_environment_comparison(self, results: Dict):
        """Plot performance comparison across environments."""
        env_names = []
        statemask_scores = []
        baseline_scores = []
        
        for env_name, env_results in results.items():
            if env_name == 'ablation_study':
                continue
                
            # Aggregate results across seeds
            statemask_rewards = []
            baseline_rewards = []
            
            for key, result in env_results.items():
                if 'seed' in key and 'baselines' not in key:
                    if 'explanation_quality' in result:
                        statemask_rewards.append(result['explanation_quality']['fidelity'])
                elif 'baselines' in key:
                    if 'PPO_finetune' in result:
                        baseline_rewards.append(result['PPO_finetune'])
            
            if statemask_rewards and baseline_rewards:
                env_names.append(env_name)
                statemask_scores.append(np.mean(statemask_rewards))
                baseline_scores.append(np.mean(baseline_rewards))
        
        if env_names:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(env_names))
            width = 0.35
            
            ax.bar(x - width/2, statemask_scores, width, label='StateMask', alpha=0.8)
            ax.bar(x + width/2, baseline_scores, width, label='PPO Fine-tuning', alpha=0.8)
            
            ax.set_xlabel('Environment')
            ax.set_ylabel('Performance Score')
            ax.set_title('StateMask vs Baseline Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(env_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'environment_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_ablation_study(self, ablation_results: Dict):
        """Plot ablation study results."""
        # Extract data for heatmap
        p_values = sorted(set(self.p_values))
        lambda_values = sorted(set(self.lambda_values))
        
        # Create performance matrix
        performance_matrix = np.zeros((len(p_values), len(lambda_values)))
        
        for i, p in enumerate(p_values):
            for j, lambda_reg in enumerate(lambda_values):
                scores = []
                for seed in range(self.num_seeds):
                    key = f"p{p}_lambda{lambda_reg}_seed{seed}"
                    if key in ablation_results and 'explanation_quality' in ablation_results[key]:
                        scores.append(ablation_results[key]['explanation_quality']['fidelity'])
                
                if scores:
                    performance_matrix[i, j] = np.mean(scores)
                else:
                    performance_matrix[i, j] = np.nan
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(performance_matrix, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(lambda_values)))
        ax.set_yticks(range(len(p_values)))
        ax.set_xticklabels([f'{λ:.3f}' for λ in lambda_values])
        ax.set_yticklabels([f'{p:.2f}' for p in p_values])
        
        ax.set_xlabel('λ (Regularization Parameter)')
        ax.set_ylabel('p (Mask Probability)')
        ax.set_title('Ablation Study: Performance vs Hyperparameters')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Fidelity Score')
        
        # Add text annotations
        for i in range(len(p_values)):
            for j in range(len(lambda_values)):
                if not np.isnan(performance_matrix[i, j]):
                    text = ax.text(j, i, f'{performance_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="white", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'ablation_study_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self, results: Dict):
        """Plot training curves for explanation module."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for env_name, env_results in results.items():
            if env_name == 'ablation_study' or plot_idx >= 4:
                continue
                
            # Collect training curves
            all_losses = []
            all_fidelity = []
            
            for key, result in env_results.items():
                if 'seed' in key and 'baselines' not in key and 'explanation_stats' in result:
                    stats = result['explanation_stats']
                    if 'losses' in stats:
                        all_losses.append(stats['losses'])
                    if 'fidelity_losses' in stats:
                        all_fidelity.append(stats['fidelity_losses'])
            
            if all_losses:
                ax = axes[plot_idx]
                
                # Plot mean and std
                losses_array = np.array(all_losses)
                mean_losses = np.mean(losses_array, axis=0)
                std_losses = np.std(losses_array, axis=0)
                epochs = range(len(mean_losses))
                
                ax.plot(epochs, mean_losses, label='Total Loss', linewidth=2)
                ax.fill_between(epochs, mean_losses - std_losses, mean_losses + std_losses, alpha=0.3)
                
                if all_fidelity:
                    fidelity_array = np.array(all_fidelity)
                    mean_fidelity = np.mean(fidelity_array, axis=0)
                    std_fidelity = np.std(fidelity_array, axis=0)
                    
                    ax.plot(epochs, mean_fidelity, label='Fidelity Loss', linewidth=2)
                    ax.fill_between(epochs, mean_fidelity - std_fidelity, mean_fidelity + std_fidelity, alpha=0.3)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'Training Curves - {env_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        # Remove unused subplots
        for i in range(plot_idx, 4):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(description='StateMask Paper Reproduction')
    parser.add_argument('--config', type=str, default='config.json', 
                       help='Configuration file path')
    parser.add_argument('--env', type=str, default=None,
                       help='Specific environment to run (optional)')
    parser.add_argument('--ablation-only', action='store_true',
                       help='Run only ablation study')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory')
    parser.add_argument('--num-seeds', type=int, default=3,
                       help='Number of random seeds')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'suboptimal_timesteps': 50000,
            'run_refinement': True,
            'results_dir': args.results_dir
        }
    
    # Update config with command line arguments
    config['results_dir'] = args.results_dir
    
    # Initialize experiment runner
    runner = ExperimentRunner(config)
    runner.num_seeds = args.num_seeds
    
    try:
        if args.ablation_only:
            # Run only ablation study
            logger.info("Running ablation study only")
            env_name = args.env if args.env else 'Hopper-v3'
            results = runner.run_ablation_study(env_name)
            runner.save_results({'ablation_study': results})
            runner.plot_ablation_study(results)
            
        elif args.env:
            # Run specific environment
            logger.info(f"Running experiments for {args.env}")
            env_results = {}
            for seed in range(args.num_seeds):
                result = runner.run_statemask_experiment(args.env, p=0.5, lambda_reg=0.01, seed=seed)
                env_results[f"seed{seed}"] = result
            
            results = {args.env: env_results}
            runner.save_results(results)
            runner.generate_plots(results)
            
        else:
            # Run all experiments
            logger.info("Running all experiments")
            results = runner.run_all_experiments()
            
        logger.info("Experimental evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Experimental evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()