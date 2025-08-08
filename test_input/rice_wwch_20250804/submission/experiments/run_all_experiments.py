import argparse
import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional, Any

# Import environment modules
try:
    import gym
    import mujoco_py
except ImportError:
    print("Warning: MuJoCo environments not available")

# Import custom environments
from environments.selfish_mining import SelfishMiningEnv
from environments.cage_challenge import CageChallenge2Env
from environments.auto_driving import AutoDrivingEnv
from environments.malware_mutation import MalwareMutationEnv

# Import algorithm modules
from algorithms.ppo import PPOAgent
from algorithms.rice import RICEAgent
from algorithms.mask_network import MaskNetwork
from algorithms.baselines import StateMaskR, JSRL

class ExperimentRunner:
    """
    Complete experiment runner for RICE paper reproduction.
    
    This class implements all experiments described in the paper:
    - Pretraining phase: Train suboptimal PPO agents
    - Mask network training: Train explanation module using Algorithm 1
    - Policy refinement: RICE refinement using Algorithm 2
    - Baseline comparisons: PPO fine-tuning, StateMask-R, JSRL
    - Ablation studies: Different p and λ values
    - Multi-seed experiments: At least 3 random seeds per experiment
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize experiment runner with configuration.
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = defaultdict(list)
        
        # Setup logging
        self.setup_logging()
        
        # Environment configurations
        self.env_configs = {
            'Hopper-v3': {'type': 'mujoco', 'max_episode_steps': 1000},
            'Walker2d-v3': {'type': 'mujoco', 'max_episode_steps': 1000},
            'Reacher-v2': {'type': 'mujoco', 'max_episode_steps': 50},
            'HalfCheetah-v3': {'type': 'mujoco', 'max_episode_steps': 1000},
            'SelfishMining': {'type': 'custom', 'max_episode_steps': 200},
            'CageChallenge2': {'type': 'custom', 'max_episode_steps': 100},
            'AutoDriving': {'type': 'custom', 'max_episode_steps': 500},
            'MalwareMutation': {'type': 'custom', 'max_episode_steps': 300}
        }
        
        # Experiment parameters from paper
        self.default_params = {
            'pretraining_episodes': 1000,
            'mask_training_episodes': 500,
            'refinement_episodes': 200,
            'p_values': [0.1, 0.3, 0.5, 0.7, 0.9],  # Masking probabilities
            'lambda_values': [0.01, 0.1, 1.0, 10.0],  # Regularization weights
            'learning_rates': {'ppo': 3e-4, 'mask': 1e-3, 'rice': 1e-4},
            'batch_size': 64,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5
        }
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = os.path.join(self.config['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_environment(self, env_name: str):
        """
        Create environment instance based on environment name.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Environment instance
        """
        env_config = self.env_configs[env_name]
        
        if env_config['type'] == 'mujoco':
            try:
                env = gym.make(env_name)
                return env
            except Exception as e:
                self.logger.error(f"Failed to create MuJoCo environment {env_name}: {e}")
                return None
                
        elif env_config['type'] == 'custom':
            if env_name == 'SelfishMining':
                return SelfishMiningEnv()
            elif env_name == 'CageChallenge2':
                return CageChallenge2Env()
            elif env_name == 'AutoDriving':
                return AutoDrivingEnv()
            elif env_name == 'MalwareMutation':
                return MalwareMutationEnv()
        
        raise ValueError(f"Unknown environment: {env_name}")
        
    def set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
    def pretrain_suboptimal_agent(self, env_name: str, seed: int) -> PPOAgent:
        """
        Phase 1: Pretrain suboptimal PPO agent.
        
        This implements the pretraining phase where we train a PPO agent
        with limited episodes to create a suboptimal policy that can be
        later improved through RICE.
        
        Args:
            env_name: Environment name
            seed: Random seed
            
        Returns:
            Trained suboptimal PPO agent
        """
        self.logger.info(f"Starting pretraining for {env_name} with seed {seed}")
        
        env = self.create_environment(env_name)
        if env is None:
            return None
            
        # Initialize PPO agent with suboptimal training configuration
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n,
            lr=self.default_params['learning_rates']['ppo'],
            gamma=self.default_params['gamma'],
            gae_lambda=self.default_params['gae_lambda'],
            clip_epsilon=self.default_params['clip_epsilon'],
            entropy_coef=self.default_params['entropy_coef'],
            value_coef=self.default_params['value_coef'],
            device=self.device
        )
        
        # Training loop for suboptimal agent
        episode_rewards = []
        for episode in range(self.default_params['pretraining_episodes']):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
            episode_rewards.append(episode_reward)
            
            # Update agent every batch_size episodes
            if (episode + 1) % self.default_params['batch_size'] == 0:
                agent.update()
                
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                self.logger.info(f"Pretraining Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
                
        # Save pretrained agent
        model_dir = os.path.join(self.config['output_dir'], 'pretrained_models', env_name)
        os.makedirs(model_dir, exist_ok=True)
        agent.save(os.path.join(model_dir, f'ppo_seed_{seed}.pth'))
        
        env.close()
        return agent
        
    def train_mask_network(self, env_name: str, pretrained_agent: PPOAgent, seed: int) -> MaskNetwork:
        """
        Phase 2: Train mask network using Algorithm 1.
        
        This implements Algorithm 1 from the paper for training the explanation
        module (mask network) that learns to identify important state features.
        
        Algorithm 1: Training Explanation Module
        1. Initialize mask network M_θ
        2. For each episode:
            a. Collect trajectory using pretrained policy
            b. Generate masks for each state
            c. Compute masked policy outputs
            d. Calculate explanation loss
            e. Update mask network parameters
            
        Args:
            env_name: Environment name
            pretrained_agent: Pretrained suboptimal agent
            seed: Random seed
            
        Returns:
            Trained mask network
        """
        self.logger.info(f"Starting mask network training for {env_name} with seed {seed}")
        
        env = self.create_environment(env_name)
        if env is None:
            return None
            
        # Initialize mask network
        state_dim = env.observation_space.shape[0]
        mask_network = MaskNetwork(
            state_dim=state_dim,
            hidden_dim=256,
            lr=self.default_params['learning_rates']['mask'],
            device=self.device
        )
        
        # Training loop for mask network (Algorithm 1)
        for episode in range(self.default_params['mask_training_episodes']):
            state = env.reset()
            trajectory = []
            done = False
            
            # Collect trajectory
            while not done:
                action = pretrained_agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                trajectory.append({
                    'state': state.copy(),
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done
                })
                state = next_state
                
            # Train mask network on collected trajectory
            for step_data in trajectory:
                state = torch.FloatTensor(step_data['state']).unsqueeze(0).to(self.device)
                
                # Generate mask
                mask = mask_network.generate_mask(state)
                
                # Apply mask to state
                masked_state = state * mask
                
                # Get original and masked policy outputs
                with torch.no_grad():
                    original_action_prob = pretrained_agent.get_action_probability(state)
                    masked_action_prob = pretrained_agent.get_action_probability(masked_state)
                
                # Compute explanation loss (KL divergence + sparsity regularization)
                kl_loss = torch.nn.functional.kl_div(
                    torch.log(masked_action_prob + 1e-8),
                    original_action_prob,
                    reduction='batchmean'
                )
                
                sparsity_loss = torch.mean(mask)  # Encourage sparse masks
                
                total_loss = kl_loss + 0.1 * sparsity_loss
                
                # Update mask network
                mask_network.optimizer.zero_grad()
                total_loss.backward()
                mask_network.optimizer.step()
                
            if (episode + 1) % 50 == 0:
                self.logger.info(f"Mask training Episode {episode + 1}, Loss: {total_loss.item():.4f}")
                
        # Save trained mask network
        model_dir = os.path.join(self.config['output_dir'], 'mask_networks', env_name)
        os.makedirs(model_dir, exist_ok=True)
        mask_network.save(os.path.join(model_dir, f'mask_seed_{seed}.pth'))
        
        env.close()
        return mask_network
        
    def rice_refinement(self, env_name: str, pretrained_agent: PPOAgent, 
                       mask_network: MaskNetwork, seed: int, p: float = 0.5, 
                       lambda_reg: float = 0.1) -> RICEAgent:
        """
        Phase 3: RICE policy refinement using Algorithm 2.
        
        This implements Algorithm 2 from the paper for RICE (Reinforcement learning
        with Interpretable Counterfactual Explanations) policy refinement.
        
        Algorithm 2: RICE Policy Refinement
        1. Initialize RICE agent with pretrained policy
        2. For each episode:
            a. Collect trajectory with current policy
            b. Generate counterfactual explanations using mask network
            c. Compute RICE loss (policy loss + explanation consistency)
            d. Update policy parameters
            
        Args:
            env_name: Environment name
            pretrained_agent: Pretrained suboptimal agent
            mask_network: Trained mask network
            seed: Random seed
            p: Masking probability
            lambda_reg: Regularization weight
            
        Returns:
            Refined RICE agent
        """
        self.logger.info(f"Starting RICE refinement for {env_name} with seed {seed}, p={p}, λ={lambda_reg}")
        
        env = self.create_environment(env_name)
        if env is None:
            return None
            
        # Initialize RICE agent
        rice_agent = RICEAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n,
            lr=self.default_params['learning_rates']['rice'],
            gamma=self.default_params['gamma'],
            mask_network=mask_network,
            lambda_reg=lambda_reg,
            device=self.device
        )
        
        # Initialize with pretrained weights
        rice_agent.load_pretrained_weights(pretrained_agent)
        
        # Training loop for RICE refinement (Algorithm 2)
        episode_rewards = []
        for episode in range(self.default_params['refinement_episodes']):
            state = env.reset()
            episode_reward = 0
            trajectory = []
            done = False
            
            # Collect trajectory
            while not done:
                action = rice_agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                trajectory.append({
                    'state': state.copy(),
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done
                })
                
                state = next_state
                episode_reward += reward
                
            episode_rewards.append(episode_reward)
            
            # RICE update with counterfactual explanations
            for step_data in trajectory:
                state_tensor = torch.FloatTensor(step_data['state']).unsqueeze(0).to(self.device)
                action_tensor = torch.FloatTensor([step_data['action']]).to(self.device)
                reward_tensor = torch.FloatTensor([step_data['reward']]).to(self.device)
                
                # Generate mask with probability p
                if np.random.random() < p:
                    mask = mask_network.generate_mask(state_tensor)
                    masked_state = state_tensor * mask
                    
                    # Compute counterfactual action
                    counterfactual_action = rice_agent.select_action(masked_state.cpu().numpy().squeeze())
                    
                    # Store transition with counterfactual information
                    rice_agent.store_transition(
                        step_data['state'], 
                        step_data['action'], 
                        step_data['reward'],
                        step_data['next_state'], 
                        step_data['done'],
                        counterfactual_action=counterfactual_action,
                        mask=mask.cpu().numpy().squeeze()
                    )
                else:
                    # Store normal transition
                    rice_agent.store_transition(
                        step_data['state'], 
                        step_data['action'], 
                        step_data['reward'],
                        step_data['next_state'], 
                        step_data['done']
                    )
                    
            # Update RICE agent
            if (episode + 1) % self.default_params['batch_size'] == 0:
                rice_agent.update()
                
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                self.logger.info(f"RICE Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
                
        # Save refined agent
        model_dir = os.path.join(self.config['output_dir'], 'rice_models', env_name)
        os.makedirs(model_dir, exist_ok=True)
        rice_agent.save(os.path.join(model_dir, f'rice_seed_{seed}_p_{p}_lambda_{lambda_reg}.pth'))
        
        env.close()
        return rice_agent
        
    def run_baseline_experiments(self, env_name: str, pretrained_agent: PPOAgent, seed: int):
        """
        Run baseline method comparisons.
        
        This implements the baseline methods mentioned in the paper:
        - PPO fine-tuning: Direct fine-tuning of pretrained PPO
        - StateMask-R: State masking with random masks
        - JSRL: Joint learning of policy and explanations
        
        Args:
            env_name: Environment name
            pretrained_agent: Pretrained suboptimal agent
            seed: Random seed
        """
        self.logger.info(f"Running baseline experiments for {env_name} with seed {seed}")
        
        baselines = {
            'ppo_finetune': self.run_ppo_finetuning,
            'statemask_r': self.run_statemask_r,
            'jsrl': self.run_jsrl
        }
        
        baseline_results = {}
        for baseline_name, baseline_func in baselines.items():
            try:
                result = baseline_func(env_name, pretrained_agent, seed)
                baseline_results[baseline_name] = result
                self.logger.info(f"Completed {baseline_name} for {env_name}")
            except Exception as e:
                self.logger.error(f"Failed to run {baseline_name} for {env_name}: {e}")
                baseline_results[baseline_name] = None
                
        return baseline_results
        
    def run_ppo_finetuning(self, env_name: str, pretrained_agent: PPOAgent, seed: int):
        """Run PPO fine-tuning baseline."""
        env = self.create_environment(env_name)
        if env is None:
            return None
            
        # Clone pretrained agent for fine-tuning
        finetuned_agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n,
            lr=self.default_params['learning_rates']['ppo'] * 0.1,  # Lower learning rate for fine-tuning
            gamma=self.default_params['gamma'],
            device=self.device
        )
        finetuned_agent.load_state_dict(pretrained_agent.state_dict())
        
        # Fine-tuning loop
        episode_rewards = []
        for episode in range(self.default_params['refinement_episodes']):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = finetuned_agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                finetuned_agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % self.default_params['batch_size'] == 0:
                finetuned_agent.update()
                
        env.close()
        return {'final_reward': np.mean(episode_rewards[-10:]), 'all_rewards': episode_rewards}
        
    def run_statemask_r(self, env_name: str, pretrained_agent: PPOAgent, seed: int):
        """Run StateMask-R baseline with random masking."""
        env = self.create_environment(env_name)
        if env is None:
            return None
            
        statemask_agent = StateMaskR(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n,
            lr=self.default_params['learning_rates']['rice'],
            device=self.device
        )
        statemask_agent.load_pretrained_weights(pretrained_agent)
        
        # Training with random masking
        episode_rewards = []
        for episode in range(self.default_params['refinement_episodes']):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Apply random mask
                mask = np.random.binomial(1, 0.5, size=state.shape)
                masked_state = state * mask
                
                action = statemask_agent.select_action(masked_state)
                next_state, reward, done, _ = env.step(action)
                statemask_agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % self.default_params['batch_size'] == 0:
                statemask_agent.update()
                
        env.close()
        return {'final_reward': np.mean(episode_rewards[-10:]), 'all_rewards': episode_rewards}
        
    def run_jsrl(self, env_name: str, pretrained_agent: PPOAgent, seed: int):
        """Run JSRL baseline with joint learning."""
        env = self.create_environment(env_name)
        if env is None:
            return None
            
        jsrl_agent = JSRL(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n,
            lr=self.default_params['learning_rates']['rice'],
            device=self.device
        )
        jsrl_agent.load_pretrained_weights(pretrained_agent)
        
        # Joint training of policy and explanations
        episode_rewards = []
        for episode in range(self.default_params['refinement_episodes']):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = jsrl_agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                jsrl_agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % self.default_params['batch_size'] == 0:
                jsrl_agent.update()
                
        env.close()
        return {'final_reward': np.mean(episode_rewards[-10:]), 'all_rewards': episode_rewards}
        
    def run_ablation_studies(self, env_name: str, pretrained_agent: PPOAgent, 
                           mask_network: MaskNetwork, seed: int):
        """
        Run ablation studies with different hyperparameters.
        
        This tests the effect of different p values (masking probabilities)
        and λ values (regularization weights) on RICE performance.
        
        Args:
            env_name: Environment name
            pretrained_agent: Pretrained suboptimal agent
            mask_network: Trained mask network
            seed: Random seed
        """
        self.logger.info(f"Running ablation studies for {env_name} with seed {seed}")
        
        ablation_results = {}
        
        # Test different p values (masking probabilities)
        for p in self.default_params['p_values']:
            self.logger.info(f"Testing p={p}")
            rice_agent = self.rice_refinement(env_name, pretrained_agent, mask_network, seed, p=p)
            if rice_agent is not None:
                # Evaluate performance
                eval_reward = self.evaluate_agent(env_name, rice_agent, num_episodes=10)
                ablation_results[f'p_{p}'] = eval_reward
                
        # Test different λ values (regularization weights)
        for lambda_val in self.default_params['lambda_values']:
            self.logger.info(f"Testing λ={lambda_val}")
            rice_agent = self.rice_refinement(env_name, pretrained_agent, mask_network, seed, lambda_reg=lambda_val)
            if rice_agent is not None:
                # Evaluate performance
                eval_reward = self.evaluate_agent(env_name, rice_agent, num_episodes=10)
                ablation_results[f'lambda_{lambda_val}'] = eval_reward
                
        return ablation_results
        
    def evaluate_agent(self, env_name: str, agent, num_episodes: int = 10):
        """
        Evaluate agent performance over multiple episodes.
        
        Args:
            env_name: Environment name
            agent: Agent to evaluate
            num_episodes: Number of evaluation episodes
            
        Returns:
            Average reward over evaluation episodes
        """
        env = self.create_environment(env_name)
        if env is None:
            return 0.0
            
        total_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state, deterministic=True)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                
            total_rewards.append(episode_reward)
            
        env.close()
        return np.mean(total_rewards)
        
    def run_single_environment_experiment(self, env_name: str, seeds: List[int]):
        """
        Run complete experiment pipeline for a single environment.
        
        Args:
            env_name: Environment name
            seeds: List of random seeds to use
        """
        self.logger.info(f"Starting experiments for environment: {env_name}")
        
        env_results = {
            'pretraining': [],
            'rice': [],
            'baselines': [],
            'ablation': []
        }
        
        for seed in seeds:
            self.logger.info(f"Running seed {seed} for {env_name}")
            self.set_random_seeds(seed)
            
            try:
                # Phase 1: Pretraining
                pretrained_agent = self.pretrain_suboptimal_agent(env_name, seed)
                if pretrained_agent is None:
                    continue
                    
                # Evaluate pretrained agent
                pretrain_reward = self.evaluate_agent(env_name, pretrained_agent)
                env_results['pretraining'].append(pretrain_reward)
                
                # Phase 2: Mask network training
                mask_network = self.train_mask_network(env_name, pretrained_agent, seed)
                if mask_network is None:
                    continue
                    
                # Phase 3: RICE refinement
                rice_agent = self.rice_refinement(env_name, pretrained_agent, mask_network, seed)
                if rice_agent is None:
                    continue
                    
                # Evaluate RICE agent
                rice_reward = self.evaluate_agent(env_name, rice_agent)
                env_results['rice'].append(rice_reward)
                
                # Phase 4: Baseline comparisons
                baseline_results = self.run_baseline_experiments(env_name, pretrained_agent, seed)
                env_results['baselines'].append(baseline_results)
                
                # Phase 5: Ablation studies
                ablation_results = self.run_ablation_studies(env_name, pretrained_agent, mask_network, seed)
                env_results['ablation'].append(ablation_results)
                
                self.logger.info(f"Completed seed {seed} for {env_name}")
                
            except Exception as e:
                self.logger.error(f"Error in seed {seed} for {env_name}: {e}")
                continue
                
        # Save environment results
        results_file = os.path.join(self.config['output_dir'], 'results', f'{env_name}_results.json')
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(env_results, f, indent=2)
            
        self.results[env_name] = env_results
        self.logger.info(f"Completed all experiments for {env_name}")
        
    def run_all_experiments(self):
        """Run experiments for all environments."""
        environments = self.config.get('environments', list(self.env_configs.keys()))
        seeds = self.config.get('seeds', [42, 123, 456])
        
        self.logger.info(f"Starting experiments for {len(environments)} environments with {len(seeds)} seeds each")
        
        for env_name in environments:
            if env_name in self.env_configs:
                self.run_single_environment_experiment(env_name, seeds)
            else:
                self.logger.warning(f"Unknown environment: {env_name}")
                
        # Generate final report
        self.generate_final_report()
        
    def generate_final_report(self):
        """Generate comprehensive experiment report."""
        self.logger.info("Generating final experiment report")
        
        report = {
            'experiment_config': self.config,
            'results_summary': {},
            'statistical_analysis': {}
        }
        
        # Summarize results for each environment
        for env_name, env_results in self.results.items():
            if not env_results['rice']:
                continue
                
            # Calculate statistics
            pretrain_mean = np.mean(env_results['pretraining'])
            pretrain_std = np.std(env_results['pretraining'])
            
            rice_mean = np.mean(env_results['rice'])
            rice_std = np.std(env_results['rice'])
            
            improvement = ((rice_mean - pretrain_mean) / pretrain_mean) * 100
            
            report['results_summary'][env_name] = {
                'pretrain_reward': {'mean': pretrain_mean, 'std': pretrain_std},
                'rice_reward': {'mean': rice_mean, 'std': rice_std},
                'improvement_percent': improvement,
                'num_seeds': len(env_results['rice'])
            }
            
            # Baseline comparisons
            if env_results['baselines']:
                baseline_summary = {}
                for baseline_name in ['ppo_finetune', 'statemask_r', 'jsrl']:
                    baseline_rewards = [b[baseline_name]['final_reward'] for b in env_results['baselines'] 
                                      if b[baseline_name] is not None]
                    if baseline_rewards:
                        baseline_summary[baseline_name] = {
                            'mean': np.mean(baseline_rewards),
                            'std': np.std(baseline_rewards)
                        }
                        
                report['results_summary'][env_name]['baselines'] = baseline_summary
                
        # Save final report
        report_file = os.path.join(self.config['output_dir'], 'final_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print summary to console
        self.print_results_summary(report)
        
    def print_results_summary(self, report: Dict):
        """Print results summary to console."""
        print("\n" + "="*80)
        print("RICE EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        for env_name, results in report['results_summary'].items():
            print(f"\nEnvironment: {env_name}")
            print("-" * 40)
            print(f"Pretrain Reward: {results['pretrain_reward']['mean']:.2f} ± {results['pretrain_reward']['std']:.2f}")
            print(f"RICE Reward:     {results['rice_reward']['mean']:.2f} ± {results['rice_reward']['std']:.2f}")
            print(f"Improvement:     {results['improvement_percent']:.1f}%")
            print(f"Number of seeds: {results['num_seeds']}")
            
            if 'baselines' in results:
                print("\nBaseline Comparisons:")
                for baseline, stats in results['baselines'].items():
                    print(f"  {baseline}: {stats['mean']:.2f} ± {stats['std']:.2f}")
                    
        print("\n" + "="*80)


def main():
    """Main function to run experiments based on command line arguments."""
    parser = argparse.ArgumentParser(description='RICE Paper Reproduction Experiments')
    
    # Environment selection
    parser.add_argument('--env', type=str, choices=[
        'Hopper-v3', 'Walker2d-v3', 'Reacher-v2', 'HalfCheetah-v3',
        'SelfishMining', 'CageChallenge2', 'AutoDriving', 'MalwareMutation', 'all'
    ], default='all', help='Environment to run experiments on')
    
    # Experiment configuration
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                       help='Random seeds for experiments')
    parser.add_argument('--output_dir', type=str, default='./rice_experiments',
                       help='Output directory for results')
    
    # Experiment phases
    parser.add_argument('--skip_pretraining', action='store_true',
                       help='Skip pretraining phase (use existing models)')
    parser.add_argument('--skip_baselines', action='store_true',
                       help='Skip baseline experiments')
    parser.add_argument('--skip_ablation', action='store_true',
                       help='Skip ablation studies')
    
    # Hyperparameter overrides
    parser.add_argument('--pretraining_episodes', type=int, default=1000,
                       help='Number of pretraining episodes')
    parser.add_argument('--refinement_episodes', type=int, default=200,
                       help='Number of refinement episodes')
    parser.add_argument('--p_values', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9],
                       help='Masking probability values for ablation')
    parser.add_argument('--lambda_values', nargs='+', type=float, default=[0.01, 0.1, 1.0, 10.0],
                       help='Regularization weight values for ablation')
    
    args = parser.parse_args()
    
    # Create experiment configuration
    config = {
        'environments': [args.env] if args.env != 'all' else [
            'Hopper-v3', 'Walker2d-v3', 'Reacher-v2', 'HalfCheetah-v3',
            'SelfishMining', 'CageChallenge2', 'AutoDriving', 'MalwareMutation'
        ],
        'seeds': args.seeds,
        'output_dir': args.output_dir,
        'skip_pretraining': args.skip_pretraining,
        'skip_baselines': args.skip_baselines,
        'skip_ablation': args.skip_ablation,
        'pretraining_episodes': args.pretraining_episodes,
        'refinement_episodes': args.refinement_episodes,
        'p_values': args.p_values,
        'lambda_values': args.lambda_values
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize and run experiments
    runner = ExperimentRunner(config)
    
    print(f"Starting RICE experiments with configuration:")
    print(f"  Environments: {config['environments']}")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Output directory: {config['output_dir']}")
    
    start_time = time.time()
    runner.run_all_experiments()
    end_time = time.time()
    
    print(f"\nAll experiments completed in {(end_time - start_time) / 3600:.2f} hours")
    print(f"Results saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()