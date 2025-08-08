import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import json
import pickle
from pathlib import Path

# Import environment and agent modules
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Import RICE algorithm components
from rice_algorithm import RICEExplainer, RICERefinement
from mask_network import MaskNetwork
from utils import setup_logging, save_results, load_pretrained_agent
from environments import create_environment, SUPPORTED_ENVIRONMENTS

def parse_arguments():
    """Parse command line arguments for RICE algorithm reproduction."""
    parser = argparse.ArgumentParser(
        description="RICE Algorithm Reproduction - Reinforcement Learning with Interpretable Counterfactual Explanations"
    )
    
    # Environment settings
    parser.add_argument(
        '--env', 
        type=str, 
        choices=SUPPORTED_ENVIRONMENTS,
        default='Hopper-v4',
        help='Environment to run experiments on'
    )
    
    # Algorithm settings
    parser.add_argument(
        '--mode',
        type=str,
        choices=['explanation', 'refinement', 'both'],
        default='both',
        help='Run explanation only, refinement only, or both'
    )
    
    # Training parameters
    parser.add_argument(
        '--mask_epochs',
        type=int,
        default=1000,
        help='Number of epochs for mask network training (Algorithm 1)'
    )
    
    parser.add_argument(
        '--refinement_epochs',
        type=int,
        default=500,
        help='Number of epochs for policy refinement (Algorithm 2)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3e-4,
        help='Learning rate for optimization'
    )
    
    # Explanation parameters
    parser.add_argument(
        '--lambda_sparsity',
        type=float,
        default=0.01,
        help='Sparsity regularization coefficient'
    )
    
    parser.add_argument(
        '--lambda_fidelity',
        type=float,
        default=1.0,
        help='Fidelity loss coefficient'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--eval_episodes',
        type=int,
        default=100,
        help='Number of episodes for evaluation'
    )
    
    parser.add_argument(
        '--num_seeds',
        type=int,
        default=5,
        help='Number of random seeds for experiments'
    )
    
    # I/O settings
    parser.add_argument(
        '--pretrained_path',
        type=str,
        default='pretrained_models',
        help='Path to pretrained agent models'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    # Device settings
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (cpu, cuda, auto)'
    )
    
    # Additional training parameters for better control
    parser.add_argument(
        '--pretrain_timesteps',
        type=int,
        default=1000000,
        help='Number of timesteps for pretraining the base agent'
    )
    
    parser.add_argument(
        '--save_frequency',
        type=int,
        default=100,
        help='Save model every N epochs during training'
    )
    
    parser.add_argument(
        '--eval_frequency',
        type=int,
        default=50,
        help='Evaluate model every N epochs during training'
    )
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """Setup computation device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logging.info(f"Using device: {device}")
    return device

def load_or_train_pretrained_agent(env_name: str, pretrained_path: str, device: torch.device, timesteps: int = 1000000) -> SAC:
    """
    Load pretrained agent or train a new one if not available.
    
    This implements the base policy training phase as described in the paper.
    The paper uses SAC with default Stable Baselines3 MlpPolicy architecture.
    """
    model_path = os.path.join(pretrained_path, f"{env_name}_sac_model.zip")
    
    if os.path.exists(model_path):
        logging.info(f"Loading pretrained agent from {model_path}")
        try:
            agent = SAC.load(model_path, device=device)
            
            # Verify the loaded agent works
            env = create_environment(env_name)
            test_reward, _ = evaluate_policy(agent, env, n_eval_episodes=5, deterministic=True)
            logging.info(f"Loaded agent performance: {test_reward:.2f}")
            env.close()
            
            return agent
        except Exception as e:
            logging.warning(f"Failed to load pretrained model: {e}")
            logging.info("Training new agent...")
    
    # Train new agent using Stable Baselines3 default settings
    logging.info(f"Training new SAC agent for {env_name}")
    env = create_environment(env_name)
    
    # Use default MlpPolicy as mentioned in the addendum
    # This ensures compatibility with the black-box assumption in RICE
    agent = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        # Use default network architecture from SB3
        policy_kwargs=dict(net_arch=[256, 256])
    )
    
    # Train for sufficient timesteps based on environment complexity
    if 'Hopper' in env_name or 'Walker2d' in env_name or 'HalfCheetah' in env_name:
        total_timesteps = max(timesteps, 1000000)
    else:
        total_timesteps = max(timesteps, 500000)
    
    logging.info(f"Training agent for {total_timesteps} timesteps")
    agent.learn(total_timesteps=total_timesteps)
    
    # Evaluate trained agent
    final_reward, final_std = evaluate_policy(agent, env, n_eval_episodes=100, deterministic=True)
    logging.info(f"Trained agent performance: {final_reward:.2f} ± {final_std:.2f}")
    
    # Save the trained model
    os.makedirs(pretrained_path, exist_ok=True)
    agent.save(model_path)
    logging.info(f"Saved trained agent to {model_path}")
    
    env.close()
    return agent

def run_explanation_phase(
    agent: SAC,
    env: gym.Env,
    args: argparse.Namespace,
    device: torch.device,
    save_dir: Optional[str] = None
) -> Tuple[MaskNetwork, Dict]:
    """
    Run the explanation phase (Algorithm 1) to train mask network.
    
    This implements the StateMask explanation method from the paper.
    Algorithm 1: Training the Mask Network for State Explanation
    
    The mask network learns to identify important state features by:
    1. Generating binary masks for state observations
    2. Optimizing fidelity (masked policy matches original policy)
    3. Optimizing sparsity (masks are as sparse as possible)
    """
    logging.info("=" * 60)
    logging.info("STARTING EXPLANATION PHASE (Algorithm 1)")
    logging.info("Training mask network for state feature importance")
    logging.info("=" * 60)
    
    # Initialize mask network
    obs_dim = env.observation_space.shape[0]
    mask_network = MaskNetwork(obs_dim).to(device)
    
    # Initialize RICE explainer
    explainer = RICEExplainer(
        agent=agent,
        mask_network=mask_network,
        lambda_sparsity=args.lambda_sparsity,
        lambda_fidelity=args.lambda_fidelity,
        device=device
    )
    
    # Create save directory for this phase
    if save_dir:
        explanation_save_dir = os.path.join(save_dir, "explanation_phase")
        os.makedirs(explanation_save_dir, exist_ok=True)
    else:
        explanation_save_dir = None
    
    # Training loop for mask network (Algorithm 1)
    logging.info(f"Training mask network for {args.mask_epochs} epochs")
    logging.info(f"Sparsity coefficient (λ_s): {args.lambda_sparsity}")
    logging.info(f"Fidelity coefficient (λ_f): {args.lambda_fidelity}")
    
    explanation_results = explainer.train(
        env=env,
        epochs=args.mask_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=explanation_save_dir,
        save_frequency=args.save_frequency,
        eval_frequency=args.eval_frequency
    )
    
    # Evaluate explanation quality
    logging.info("Evaluating explanation quality...")
    eval_results = explainer.evaluate(env, num_episodes=args.eval_episodes)
    explanation_results.update(eval_results)
    
    # Save final mask network
    if explanation_save_dir:
        mask_path = os.path.join(explanation_save_dir, "final_mask_network.pth")
        torch.save(mask_network.state_dict(), mask_path)
        logging.info(f"Saved final mask network to {mask_path}")
    
    logging.info("EXPLANATION PHASE COMPLETED")
    logging.info(f"Final sparsity: {explanation_results.get('final_sparsity', 'N/A'):.4f}")
    logging.info(f"Final fidelity: {explanation_results.get('final_fidelity', 'N/A'):.4f}")
    logging.info(f"Explanation loss: {explanation_results.get('final_loss', 'N/A'):.6f}")
    
    return mask_network, explanation_results

def run_refinement_phase(
    agent: SAC,
    mask_network: MaskNetwork,
    env: gym.Env,
    args: argparse.Namespace,
    device: torch.device,
    save_dir: Optional[str] = None
) -> Tuple[SAC, Dict]:
    """
    Run the refinement phase (Algorithm 2) to improve policy using explanations.
    
    This implements the StateMask-R refinement method from the paper.
    Algorithm 2: Policy Refinement using State Explanations
    
    The refinement process:
    1. Uses the trained mask network to identify important state features
    2. Trains a refined policy that focuses on these important features
    3. Improves policy performance through explanation-guided learning
    """
    logging.info("=" * 60)
    logging.info("STARTING REFINEMENT PHASE (Algorithm 2)")
    logging.info("Refining policy using state explanations")
    logging.info("=" * 60)
    
    # Evaluate original agent performance for comparison
    original_reward, original_std = evaluate_policy(
        agent, env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    logging.info(f"Original agent performance: {original_reward:.2f} ± {original_std:.2f}")
    
    # Initialize RICE refinement
    refinement = RICERefinement(
        agent=agent,
        mask_network=mask_network,
        device=device
    )
    
    # Create save directory for this phase
    if save_dir:
        refinement_save_dir = os.path.join(save_dir, "refinement_phase")
        os.makedirs(refinement_save_dir, exist_ok=True)
    else:
        refinement_save_dir = None
    
    # Training loop for policy refinement (Algorithm 2)
    logging.info(f"Refining policy for {args.refinement_epochs} epochs")
    
    refinement_results = refinement.train(
        env=env,
        epochs=args.refinement_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=refinement_save_dir,
        save_frequency=args.save_frequency,
        eval_frequency=args.eval_frequency
    )
    
    # Add original performance to results
    refinement_results.update({
        'original_reward': original_reward,
        'original_std': original_std
    })
    
    # Get refined agent
    refined_agent = refinement.get_refined_agent()
    
    # Evaluate refined policy
    logging.info("Evaluating refined policy...")
    refined_reward, refined_std = evaluate_policy(
        refined_agent,
        env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True
    )
    
    # Calculate improvement
    improvement = refined_reward - original_reward
    improvement_percentage = (improvement / abs(original_reward)) * 100 if original_reward != 0 else 0
    
    refinement_results.update({
        'refined_mean_reward': refined_reward,
        'refined_std_reward': refined_std,
        'improvement_absolute': improvement,
        'improvement_percentage': improvement_percentage
    })
    
    # Save refined agent
    if refinement_save_dir:
        refined_agent_path = os.path.join(refinement_save_dir, "refined_agent")
        refined_agent.save(refined_agent_path)
        logging.info(f"Saved refined agent to {refined_agent_path}")
    
    logging.info("REFINEMENT PHASE COMPLETED")
    logging.info(f"Original reward: {original_reward:.2f} ± {original_std:.2f}")
    logging.info(f"Refined reward: {refined_reward:.2f} ± {refined_std:.2f}")
    logging.info(f"Improvement: {improvement:.2f} ({improvement_percentage:.1f}%)")
    
    return refined_agent, refinement_results

def run_single_experiment(
    env_name: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int
) -> Dict:
    """
    Run a single experiment with given seed.
    
    This implements the complete RICE pipeline:
    1. Load/train base policy
    2. Train explanation mask network (Algorithm 1)
    3. Refine policy using explanations (Algorithm 2)
    4. Evaluate and compare results
    """
    logging.info("*" * 80)
    logging.info(f"RUNNING EXPERIMENT: {env_name} | Seed: {seed}")
    logging.info("*" * 80)
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Create environment with seed
    env = create_environment(env_name)
    env.reset(seed=seed)
    
    # Create experiment-specific save directory
    experiment_dir = os.path.join(args.output_dir, f"experiment_seed_{seed}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Phase 1: Load or train pretrained agent
    logging.info("Phase 1: Loading/Training base policy")
    agent = load_or_train_pretrained_agent(
        env_name, 
        args.pretrained_path, 
        device,
        args.pretrain_timesteps
    )
    
    # Evaluate original agent performance
    original_reward, original_std = evaluate_policy(
        agent, env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    
    logging.info(f"Base policy performance: {original_reward:.2f} ± {original_std:.2f}")
    
    results = {
        'environment': env_name,
        'seed': seed,
        'original_reward': original_reward,
        'original_std': original_std,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Phase 2: Run explanation phase if requested
    mask_network = None
    if args.mode in ['explanation', 'both']:
        logging.info("Phase 2: Running explanation phase")
        mask_network, explanation_results = run_explanation_phase(
            agent, env, args, device, experiment_dir
        )
        results.update({'explanation': explanation_results})
        
        # Save intermediate results
        intermediate_file = os.path.join(experiment_dir, "explanation_results.json")
        with open(intermediate_file, 'w') as f:
            json.dump({'explanation': explanation_results}, f, indent=2, default=str)
    else:
        # Create dummy mask network for refinement-only mode
        logging.info("Skipping explanation phase, creating dummy mask network")
        obs_dim = env.observation_space.shape[0]
        mask_network = MaskNetwork(obs_dim).to(device)
    
    # Phase 3: Run refinement phase if requested
    if args.mode in ['refinement', 'both']:
        logging.info("Phase 3: Running refinement phase")
        refined_agent, refinement_results = run_refinement_phase(
            agent, mask_network, env, args, device, experiment_dir
        )
        results.update({'refinement': refinement_results})
        
        # Save intermediate results
        intermediate_file = os.path.join(experiment_dir, "refinement_results.json")
        with open(intermediate_file, 'w') as f:
            json.dump({'refinement': refinement_results}, f, indent=2, default=str)
    
    # Save complete experiment results
    complete_results_file = os.path.join(experiment_dir, "complete_results.json")
    with open(complete_results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    env.close()
    
    logging.info("*" * 80)
    logging.info(f"EXPERIMENT COMPLETED: {env_name} | Seed: {seed}")
    logging.info("*" * 80)
    
    return results

def aggregate_results(all_results: List[Dict]) -> Dict:
    """
    Aggregate results across multiple seeds.
    
    Computes mean and standard deviation for all metrics across seeds.
    This provides statistical significance for the reported results.
    """
    if not all_results:
        return {}
    
    logging.info("Aggregating results across seeds...")
    
    # Group results by environment
    env_results = {}
    for result in all_results:
        env_name = result['environment']
        if env_name not in env_results:
            env_results[env_name] = []
        env_results[env_name].append(result)
    
    # Compute statistics for each environment
    aggregated = {}
    for env_name, results in env_results.items():
        env_stats = {
            'environment': env_name,
            'num_seeds': len(results),
            'original_reward_mean': np.mean([r['original_reward'] for r in results]),
            'original_reward_std': np.std([r['original_reward'] for r in results]),
        }
        
        # Aggregate explanation results if available
        if 'explanation' in results[0]:
            explanation_metrics = [
                'final_sparsity', 'final_fidelity', 'final_loss',
                'training_time', 'convergence_epoch'
            ]
            for metric in explanation_metrics:
                values = [r['explanation'].get(metric, 0) for r in results if 'explanation' in r and metric in r['explanation']]
                if values:
                    env_stats[f'explanation_{metric}_mean'] = np.mean(values)
                    env_stats[f'explanation_{metric}_std'] = np.std(values)
        
        # Aggregate refinement results if available
        if 'refinement' in results[0]:
            refinement_metrics = [
                'refined_mean_reward', 'improvement_absolute', 'improvement_percentage',
                'training_time', 'convergence_epoch'
            ]
            for metric in refinement_metrics:
                values = [r['refinement'].get(metric, 0) for r in results if 'refinement' in r and metric in r['refinement']]
                if values:
                    env_stats[f'refinement_{metric}_mean'] = np.mean(values)
                    env_stats[f'refinement_{metric}_std'] = np.std(values)
        
        aggregated[env_name] = env_stats
    
    return aggregated

def save_experiment_results(results: Dict, output_dir: str, args: argparse.Namespace):
    """
    Save experiment results to files.
    
    Creates comprehensive reports including:
    - Raw results in JSON format
    - Configuration used
    - Human-readable summary
    - Statistical analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"rice_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save configuration
    config_file = os.path.join(output_dir, f"config_{timestamp}.json")
    config = vars(args)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create detailed summary report
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("RICE Algorithm Reproduction Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Experiment Configuration:\n")
        f.write(f"- Mode: {args.mode}\n")
        f.write(f"- Number of seeds: {args.num_seeds}\n")
        f.write(f"- Mask epochs: {args.mask_epochs}\n")
        f.write(f"- Refinement epochs: {args.refinement_epochs}\n")
        f.write(f"- Lambda sparsity: {args.lambda_sparsity}\n")
        f.write(f"- Lambda fidelity: {args.lambda_fidelity}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Learning rate: {args.learning_rate}\n\n")
        
        for env_name, env_results in results.items():
            f.write(f"Environment: {env_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Original Reward: {env_results.get('original_reward_mean', 0):.2f} ± {env_results.get('original_reward_std', 0):.2f}\n")
            
            if 'explanation_final_sparsity_mean' in env_results:
                f.write(f"\nExplanation Results:\n")
                f.write(f"  Sparsity: {env_results['explanation_final_sparsity_mean']:.4f} ± {env_results.get('explanation_final_sparsity_std', 0):.4f}\n")
                f.write(f"  Fidelity: {env_results['explanation_final_fidelity_mean']:.4f} ± {env_results.get('explanation_final_fidelity_std', 0):.4f}\n")
                if 'explanation_training_time_mean' in env_results:
                    f.write(f"  Training Time: {env_results['explanation_training_time_mean']:.1f} ± {env_results.get('explanation_training_time_std', 0):.1f} seconds\n")
            
            if 'refinement_refined_mean_reward_mean' in env_results:
                f.write(f"\nRefinement Results:\n")
                f.write(f"  Refined Reward: {env_results['refinement_refined_mean_reward_mean']:.2f} ± {env_results.get('refinement_refined_mean_reward_std', 0):.2f}\n")
                f.write(f"  Absolute Improvement: {env_results.get('refinement_improvement_absolute_mean', 0):.2f} ± {env_results.get('refinement_improvement_absolute_std', 0):.2f}\n")
                f.write(f"  Percentage Improvement: {env_results.get('refinement_improvement_percentage_mean', 0):.1f}% ± {env_results.get('refinement_improvement_percentage_std', 0):.1f}%\n")
                if 'refinement_training_time_mean' in env_results:
                    f.write(f"  Training Time: {env_results['refinement_training_time_mean']:.1f} ± {env_results.get('refinement_training_time_std', 0):.1f} seconds\n")
            
            f.write("\n")
    
    # Create CSV summary for easy analysis
    csv_file = os.path.join(output_dir, f"results_summary_{timestamp}.csv")
    with open(csv_file, 'w') as f:
        f.write("Environment,Original_Reward_Mean,Original_Reward_Std,")
        if args.mode in ['explanation', 'both']:
            f.write("Sparsity_Mean,Sparsity_Std,Fidelity_Mean,Fidelity_Std,")
        if args.mode in ['refinement', 'both']:
            f.write("Refined_Reward_Mean,Refined_Reward_Std,Improvement_Mean,Improvement_Std,Improvement_Pct_Mean,Improvement_Pct_Std")
        f.write("\n")
        
        for env_name, env_results in results.items():
            f.write(f"{env_name},{env_results.get('original_reward_mean', 0):.2f},{env_results.get('original_reward_std', 0):.2f},")
            if args.mode in ['explanation', 'both']:
                f.write(f"{env_results.get('explanation_final_sparsity_mean', 0):.4f},{env_results.get('explanation_final_sparsity_std', 0):.4f},")
                f.write(f"{env_results.get('explanation_final_fidelity_mean', 0):.4f},{env_results.get('explanation_final_fidelity_std', 0):.4f},")
            if args.mode in ['refinement', 'both']:
                f.write(f"{env_results.get('refinement_refined_mean_reward_mean', 0):.2f},{env_results.get('refinement_refined_mean_reward_std', 0):.2f},")
                f.write(f"{env_results.get('refinement_improvement_absolute_mean', 0):.2f},{env_results.get('refinement_improvement_absolute_std', 0):.2f},")
                f.write(f"{env_results.get('refinement_improvement_percentage_mean', 0):.1f},{env_results.get('refinement_improvement_percentage_std', 0):.1f}")
            f.write("\n")
    
    logging.info(f"Results saved to {output_dir}")
    logging.info(f"Summary available at {summary_file}")
    logging.info(f"CSV summary available at {csv_file}")

def main():
    """
    Main entry point for RICE algorithm reproduction.
    
    This implements the complete experimental pipeline described in the paper:
    1. Parse configuration and setup environment
    2. Run experiments across multiple seeds for statistical significance
    3. Aggregate results and generate comprehensive reports
    
    The implementation follows the paper's experimental methodology to ensure
    reproducible results that match the paper's findings.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logging.info("=" * 80)
    logging.info("RICE ALGORITHM REPRODUCTION")
    logging.info("Reinforcement Learning with Interpretable Counterfactual Explanations")
    logging.info("=" * 80)
    logging.info(f"Configuration: {vars(args)}")
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration for reproducibility
    config_file = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    try:
        # Run experiments across multiple seeds for statistical significance
        all_results = []
        failed_seeds = []
        
        logging.info(f"Running {args.num_seeds} experiments with different random seeds")
        
        for seed in range(args.num_seeds):
            logging.info(f"\n{'='*60}")
            logging.info(f"EXPERIMENT {seed + 1}/{args.num_seeds} (Seed: {seed})")
            logging.info(f"{'='*60}")
            
            try:
                result = run_single_experiment(args.env, args, device, seed)
                all_results.append(result)
                
                # Save intermediate results after each successful experiment
                intermediate_file = os.path.join(args.output_dir, f"intermediate_seed_{seed}.json")
                with open(intermediate_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                logging.info(f"Experiment {seed + 1} completed successfully")
                    
            except Exception as e:
                logging.error(f"Experiment with seed {seed} failed: {e}")
                failed_seeds.append(seed)
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        # Check if we have any successful results
        if not all_results:
            logging.error("All experiments failed!")
            sys.exit(1)
        
        if failed_seeds:
            logging.warning(f"Failed seeds: {failed_seeds}")
            logging.warning(f"Successful experiments: {len(all_results)}/{args.num_seeds}")
        
        # Aggregate results across seeds
        logging.info("\n" + "="*60)
        logging.info("AGGREGATING RESULTS ACROSS SEEDS")
        logging.info("="*60)
        
        aggregated_results = aggregate_results(all_results)
        
        # Save final results
        save_experiment_results(aggregated_results, args.output_dir, args)
        
        # Print final summary
        logging.info("\n" + "="*80)
        logging.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logging.info("="*80)
        
        logging.info("Summary of results:")
        for env_name, env_results in aggregated_results.items():
            original_reward = env_results.get('original_reward_mean', 0)
            improvement = env_results.get('refinement_improvement_absolute_mean', 0)
            improvement_pct = env_results.get('refinement_improvement_percentage_mean', 0)
            
            logging.info(f"{env_name}:")
            logging.info(f"  Original: {original_reward:.2f}")
            if args.mode in ['refinement', 'both']:
                logging.info(f"  Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
            if args.mode in ['explanation', 'both']:
                sparsity = env_results.get('explanation_final_sparsity_mean', 0)
                fidelity = env_results.get('explanation_final_fidelity_mean', 0)
                logging.info(f"  Sparsity: {sparsity:.4f}, Fidelity: {fidelity:.4f}")
        
        logging.info(f"\nDetailed results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        logging.info("Experiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()