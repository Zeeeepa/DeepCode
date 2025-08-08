import numpy as np
import torch
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import warnings
warnings.filterwarnings('ignore')

class StandardizedEvaluationProtocol:
    """
    Standardized evaluation protocol following the paper's methodology.
    
    This class implements the exact evaluation protocol described in the paper:
    - Evaluation every 10,000 training steps
    - 500 independent episodes for final performance evaluation
    - Multiple random seeds (minimum 3) for statistical significance
    - Mean and standard deviation calculation
    - Deterministic policy evaluation (no exploration)
    - Table 1 consistent metric calculation
    """
    
    def __init__(self, 
                 env_name: str,
                 algorithm: str = 'SAC',
                 n_seeds: int = 3,
                 eval_freq: int = 10000,
                 n_eval_episodes: int = 500,
                 total_timesteps: int = 1000000,
                 log_dir: str = './evaluation_logs',
                 deterministic: bool = True):
        """
        Initialize the standardized evaluation protocol.
        
        Args:
            env_name: Environment name (e.g., 'HalfCheetah-v4')
            algorithm: RL algorithm to use ('SAC' by default)
            n_seeds: Number of random seeds for statistical significance
            eval_freq: Frequency of evaluation during training (steps)
            n_eval_episodes: Number of episodes for final evaluation
            total_timesteps: Total training timesteps
            log_dir: Directory for logging results
            deterministic: Whether to use deterministic policy during evaluation
        """
        self.env_name = env_name
        self.algorithm = algorithm
        self.n_seeds = n_seeds
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.total_timesteps = total_timesteps
        self.log_dir = Path(log_dir)
        self.deterministic = deterministic
        
        # Create logging directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = defaultdict(list)
        self.training_curves = defaultdict(list)
        
    def create_environment(self, seed: int) -> gym.Env:
        """
        Create and configure the environment with proper seeding.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Configured environment
        """
        env = gym.make(self.env_name)
        env = Monitor(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        
        return env
    
    def create_model(self, env: gym.Env, seed: int) -> Any:
        """
        Create the RL model with standardized configuration.
        
        Args:
            env: Training environment
            seed: Random seed
            
        Returns:
            Configured RL model
        """
        if self.algorithm == 'SAC':
            # Use default MlpPolicy as mentioned in the addendum
            model = SAC(
                'MlpPolicy',
                env,
                verbose=0,
                seed=seed,
                device='auto'
            )
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")
        
        return model
    
    class EvaluationCallback(BaseCallback):
        """
        Custom callback for periodic evaluation during training.
        """
        
        def __init__(self, eval_env: gym.Env, eval_freq: int, n_eval_episodes: int,
                     deterministic: bool, results_storage: dict, seed: int):
            super().__init__()
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.deterministic = deterministic
            self.results_storage = results_storage
            self.seed = seed
            
        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                # Perform evaluation
                mean_reward, std_reward = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=self.deterministic,
                    return_episode_rewards=False
                )
                
                # Store results
                self.results_storage[f'seed_{self.seed}'].append({
                    'timestep': self.n_calls,
                    'mean_reward': mean_reward,
                    'std_reward': std_reward
                })
                
                print(f"Seed {self.seed}, Step {self.n_calls}: "
                      f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            return True
    
    def run_single_seed_experiment(self, seed: int) -> Dict[str, Any]:
        """
        Run a complete experiment for a single seed.
        
        Args:
            seed: Random seed for this experiment
            
        Returns:
            Dictionary containing all results for this seed
        """
        self.logger.info(f"Starting experiment with seed {seed}")
        
        # Set all random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Create environments
        train_env = self.create_environment(seed)
        eval_env = self.create_environment(seed + 1000)  # Different seed for evaluation
        
        # Create model
        model = self.create_model(train_env, seed)
        
        # Setup evaluation callback
        eval_callback = self.EvaluationCallback(
            eval_env=eval_env,
            eval_freq=self.eval_freq,
            n_eval_episodes=100,  # Fewer episodes during training
            deterministic=self.deterministic,
            results_storage=self.training_curves,
            seed=seed
        )
        
        # Train the model
        start_time = time.time()
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        training_time = time.time() - start_time
        
        # Final evaluation with full episode count
        self.logger.info(f"Performing final evaluation for seed {seed}")
        final_mean_reward, final_std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=False
        )
        
        # Store final results
        seed_results = {
            'seed': seed,
            'final_mean_reward': final_mean_reward,
            'final_std_reward': final_std_reward,
            'training_time': training_time,
            'training_curve': self.training_curves[f'seed_{seed}']
        }
        
        self.logger.info(f"Seed {seed} completed. Final reward: "
                        f"{final_mean_reward:.2f} ± {final_std_reward:.2f}")
        
        # Clean up
        train_env.close()
        eval_env.close()
        
        return seed_results
    
    def run_multi_seed_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete multi-seed evaluation protocol.
        
        Returns:
            Comprehensive results dictionary with statistics
        """
        self.logger.info(f"Starting multi-seed evaluation with {self.n_seeds} seeds")
        self.logger.info(f"Environment: {self.env_name}")
        self.logger.info(f"Algorithm: {self.algorithm}")
        self.logger.info(f"Evaluation frequency: {self.eval_freq} steps")
        self.logger.info(f"Final evaluation episodes: {self.n_eval_episodes}")
        
        all_seed_results = []
        
        # Run experiments for each seed
        for seed in range(self.n_seeds):
            try:
                seed_results = self.run_single_seed_experiment(seed)
                all_seed_results.append(seed_results)
                self.results[f'seed_{seed}'] = seed_results
            except Exception as e:
                self.logger.error(f"Error in seed {seed}: {str(e)}")
                continue
        
        # Calculate aggregate statistics
        final_rewards = [r['final_mean_reward'] for r in all_seed_results]
        
        aggregate_results = {
            'environment': self.env_name,
            'algorithm': self.algorithm,
            'n_seeds': len(all_seed_results),
            'eval_episodes': self.n_eval_episodes,
            'mean_reward': np.mean(final_rewards),
            'std_reward': np.std(final_rewards),
            'min_reward': np.min(final_rewards),
            'max_reward': np.max(final_rewards),
            'individual_results': all_seed_results
        }
        
        self.logger.info("=== FINAL RESULTS ===")
        self.logger.info(f"Environment: {self.env_name}")
        self.logger.info(f"Mean Reward: {aggregate_results['mean_reward']:.2f} ± "
                        f"{aggregate_results['std_reward']:.2f}")
        self.logger.info(f"Range: [{aggregate_results['min_reward']:.2f}, "
                        f"{aggregate_results['max_reward']:.2f}]")
        
        return aggregate_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary to save
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"{self.env_name}_{self.algorithm}_results.json"
        
        filepath = self.log_dir / filename
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Deep convert all numpy types
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = deep_convert(results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def plot_training_curves(self, results: Dict[str, Any], save_plot: bool = True) -> None:
        """
        Plot training curves for all seeds.
        
        Args:
            results: Results dictionary containing training curves
            save_plot: Whether to save the plot to file
        """
        plt.figure(figsize=(12, 8))
        
        # Plot individual seed curves
        for seed_result in results['individual_results']:
            curve = seed_result['training_curve']
            if curve:
                timesteps = [point['timestep'] for point in curve]
                rewards = [point['mean_reward'] for point in curve]
                plt.plot(timesteps, rewards, alpha=0.6, 
                        label=f"Seed {seed_result['seed']}")
        
        # Calculate and plot mean curve
        if results['individual_results']:
            # Align all curves to same timesteps
            all_timesteps = set()
            for seed_result in results['individual_results']:
                curve = seed_result['training_curve']
                all_timesteps.update([point['timestep'] for point in curve])
            
            all_timesteps = sorted(list(all_timesteps))
            
            mean_rewards = []
            std_rewards = []
            
            for timestep in all_timesteps:
                rewards_at_timestep = []
                for seed_result in results['individual_results']:
                    curve = seed_result['training_curve']
                    for point in curve:
                        if point['timestep'] == timestep:
                            rewards_at_timestep.append(point['mean_reward'])
                            break
                
                if rewards_at_timestep:
                    mean_rewards.append(np.mean(rewards_at_timestep))
                    std_rewards.append(np.std(rewards_at_timestep))
            
            if mean_rewards:
                mean_rewards = np.array(mean_rewards)
                std_rewards = np.array(std_rewards)
                
                plt.plot(all_timesteps[:len(mean_rewards)], mean_rewards, 
                        'k-', linewidth=2, label='Mean')
                plt.fill_between(all_timesteps[:len(mean_rewards)], 
                               mean_rewards - std_rewards,
                               mean_rewards + std_rewards,
                               alpha=0.2, color='black')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Mean Episode Reward')
        plt.title(f'Training Curves - {self.env_name} ({self.algorithm})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plot_path = self.log_dir / f"{self.env_name}_{self.algorithm}_training_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training curves saved to {plot_path}")
        
        plt.show()
    
    def generate_table1_format(self, results: Dict[str, Any]) -> str:
        """
        Generate results in Table 1 format from the paper.
        
        Args:
            results: Results dictionary
            
        Returns:
            Formatted string matching Table 1 style
        """
        mean_reward = results['mean_reward']
        std_reward = results['std_reward']
        
        # Format to match paper's precision
        formatted_result = f"{mean_reward:.1f} ± {std_reward:.1f}"
        
        table_entry = f"""
Environment: {self.env_name}
Algorithm: {self.algorithm}
Result: {formatted_result}
Episodes: {self.n_eval_episodes}
Seeds: {results['n_seeds']}
        """.strip()
        
        return table_entry
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete standardized evaluation protocol.
        
        Returns:
            Complete results dictionary
        """
        # Run multi-seed evaluation
        results = self.run_multi_seed_evaluation()
        
        # Save results
        self.save_results(results)
        
        # Generate plots
        self.plot_training_curves(results)
        
        # Print Table 1 format
        table_format = self.generate_table1_format(results)
        print("\n" + "="*50)
        print("TABLE 1 FORMAT RESULTS")
        print("="*50)
        print(table_format)
        print("="*50)
        
        return results


def run_paper_evaluation_protocol(env_names: List[str], 
                                 algorithm: str = 'SAC',
                                 n_seeds: int = 3,
                                 total_timesteps: int = 1000000) -> Dict[str, Any]:
    """
    Run the complete paper evaluation protocol for multiple environments.
    
    This function implements the exact evaluation methodology described in the paper
    to ensure results are comparable to Table 1.
    
    Args:
        env_names: List of environment names to evaluate
        algorithm: RL algorithm to use
        n_seeds: Number of random seeds for statistical significance
        total_timesteps: Total training timesteps per environment
        
    Returns:
        Dictionary containing results for all environments
    """
    all_results = {}
    
    for env_name in env_names:
        print(f"\n{'='*60}")
        print(f"EVALUATING ENVIRONMENT: {env_name}")
        print(f"{'='*60}")
        
        # Create evaluation protocol
        evaluator = StandardizedEvaluationProtocol(
            env_name=env_name,
            algorithm=algorithm,
            n_seeds=n_seeds,
            total_timesteps=total_timesteps,
            log_dir=f'./evaluation_logs/{env_name}'
        )
        
        # Run complete evaluation
        results = evaluator.run_complete_evaluation()
        all_results[env_name] = results
    
    return all_results


# Example usage for paper reproduction
if __name__ == "__main__":
    # Define environments from the paper
    mujoco_envs = [
        'HalfCheetah-v4',
        'Hopper-v4', 
        'Walker2d-v4',
        'Ant-v4'
    ]
    
    # Run standardized evaluation protocol
    print("Starting Paper Reproduction Evaluation Protocol")
    print("=" * 60)
    
    results = run_paper_evaluation_protocol(
        env_names=mujoco_envs,
        algorithm='SAC',
        n_seeds=3,  # Minimum 3 seeds as specified
        total_timesteps=1000000
    )
    
    # Print summary of all results
    print("\n" + "="*80)
    print("COMPLETE EVALUATION SUMMARY")
    print("="*80)
    
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        print(f"  Mean Reward: {env_results['mean_reward']:.1f} ± {env_results['std_reward']:.1f}")
        print(f"  Range: [{env_results['min_reward']:.1f}, {env_results['max_reward']:.1f}]")
        print(f"  Seeds: {env_results['n_seeds']}")