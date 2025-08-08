import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import logging
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

class TrainingMonitor:
    """
    Training monitor for StateMask paper reproduction.
    
    Monitors key metrics mentioned in the paper:
    - Average episode return
    - Masked network return 
    - RND loss (for curiosity-driven exploration)
    - Explanation quality metrics
    - Training stability indicators
    
    Based on paper's experimental setup focusing on overall trends
    rather than exact numerical reproduction.
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 experiment_name: str = "statemask_experiment",
                 save_frequency: int = 1000,
                 window_size: int = 100,
                 enable_tensorboard: bool = True,
                 enable_early_stopping: bool = True,
                 patience: int = 50000,
                 min_delta: float = 0.01):
        """
        Initialize training monitor.
        
        Args:
            log_dir: Directory for saving logs and checkpoints
            experiment_name: Name of the experiment
            save_frequency: Frequency of saving metrics (in steps)
            window_size: Window size for moving averages
            enable_tensorboard: Whether to use TensorBoard logging
            enable_early_stopping: Whether to enable early stopping
            patience: Early stopping patience (in steps)
            min_delta: Minimum improvement for early stopping
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_frequency = save_frequency
        self.window_size = window_size
        self.enable_early_stopping = enable_early_stopping
        self.patience = patience
        self.min_delta = min_delta
        
        # Initialize metrics storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.step_metrics = defaultdict(list)
        
        # Moving averages for stability monitoring
        self.moving_averages = defaultdict(lambda: deque(maxlen=window_size))
        
        # Training state
        self.current_step = 0
        self.current_episode = 0
        self.start_time = time.time()
        self.last_save_time = time.time()
        
        # Early stopping state
        self.best_metric = -np.inf
        self.steps_without_improvement = 0
        self.should_stop = False
        
        # TensorBoard writer
        self.writer = None
        if enable_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        
        # Setup logging
        self._setup_logging()
        
        # Metrics specific to StateMask paper
        self.paper_metrics = {
            'episode_return': [],
            'masked_return': [],
            'rnd_loss': [],
            'explanation_quality': [],
            'mask_sparsity': [],
            'training_stability': [],
            'exploration_bonus': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }
        
        self.logger.info(f"Training monitor initialized for experiment: {experiment_name}")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"TensorBoard enabled: {self.writer is not None}")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.log_dir / "training.log"
        
        # Create logger
        self.logger = logging.getLogger(f"TrainingMonitor_{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_step_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for a single training step.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step number (auto-incremented if None)
        """
        if step is None:
            step = self.current_step
            self.current_step += 1
        
        # Store metrics
        for name, value in metrics.items():
            self.step_metrics[name].append((step, value))
            self.moving_averages[name].append(value)
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar(f"step/{name}", value, step)
        
        # Log paper-specific metrics
        self._log_paper_metrics(metrics, step)
        
        # Check for early stopping
        if self.enable_early_stopping:
            self._check_early_stopping(metrics, step)
        
        # Periodic saving
        if step % self.save_frequency == 0:
            self._save_metrics()
            self._log_training_progress(step)
    
    def log_episode_metrics(self, metrics: Dict[str, float], episode: Optional[int] = None):
        """
        Log metrics for a completed episode.
        
        Args:
            metrics: Dictionary of metric name -> value
            episode: Episode number (auto-incremented if None)
        """
        if episode is None:
            episode = self.current_episode
            self.current_episode += 1
        
        # Store metrics
        for name, value in metrics.items():
            self.episode_metrics[name].append((episode, value))
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar(f"episode/{name}", value, episode)
        
        # Log episode summary
        if 'episode_return' in metrics:
            avg_return = np.mean([v for _, v in self.episode_metrics['episode_return'][-self.window_size:]])
            self.logger.info(f"Episode {episode}: Return = {metrics['episode_return']:.2f}, "
                           f"Avg Return (last {self.window_size}) = {avg_return:.2f}")
    
    def _log_paper_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics specific to StateMask paper reproduction."""
        
        # Track key metrics mentioned in the paper
        paper_metric_mapping = {
            'episode_return': 'episode_return',
            'masked_return': 'masked_network_return', 
            'rnd_loss': 'rnd_loss',
            'explanation_quality': 'explanation_score',
            'mask_sparsity': 'mask_sparsity',
            'exploration_bonus': 'intrinsic_reward',
            'policy_loss': 'policy_loss',
            'value_loss': 'value_loss',
            'entropy': 'policy_entropy'
        }
        
        for paper_key, metric_key in paper_metric_mapping.items():
            if metric_key in metrics:
                self.paper_metrics[paper_key].append((step, metrics[metric_key]))
                
                # Log moving average to TensorBoard
                if self.writer is not None and len(self.moving_averages[metric_key]) > 0:
                    avg_value = np.mean(list(self.moving_averages[metric_key]))
                    self.writer.add_scalar(f"paper_metrics/{paper_key}_avg", avg_value, step)
        
        # Calculate training stability (variance in recent performance)
        if 'episode_return' in metrics and len(self.moving_averages['episode_return']) >= 10:
            recent_returns = list(self.moving_averages['episode_return'])
            stability = 1.0 / (1.0 + np.var(recent_returns))  # Higher is more stable
            self.paper_metrics['training_stability'].append((step, stability))
            
            if self.writer is not None:
                self.writer.add_scalar("paper_metrics/training_stability", stability, step)
    
    def _check_early_stopping(self, metrics: Dict[str, float], step: int):
        """Check early stopping condition based on episode return."""
        if 'episode_return' in metrics:
            current_metric = metrics['episode_return']
            
            if current_metric > self.best_metric + self.min_delta:
                self.best_metric = current_metric
                self.steps_without_improvement = 0
                self.logger.info(f"New best metric: {self.best_metric:.4f} at step {step}")
            else:
                self.steps_without_improvement += 1
            
            if self.steps_without_improvement >= self.patience:
                self.should_stop = True
                self.logger.warning(f"Early stopping triggered at step {step}. "
                                  f"No improvement for {self.patience} steps.")
    
    def _save_metrics(self):
        """Save metrics to disk."""
        metrics_file = self.log_dir / "metrics.json"
        
        # Prepare data for JSON serialization
        save_data = {
            'step_metrics': {k: v for k, v in self.step_metrics.items()},
            'episode_metrics': {k: v for k, v in self.episode_metrics.items()},
            'paper_metrics': {k: v for k, v in self.paper_metrics.items()},
            'training_info': {
                'current_step': self.current_step,
                'current_episode': self.current_episode,
                'start_time': self.start_time,
                'best_metric': self.best_metric,
                'steps_without_improvement': self.steps_without_improvement
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.last_save_time = time.time()
    
    def _log_training_progress(self, step: int):
        """Log training progress summary."""
        elapsed_time = time.time() - self.start_time
        steps_per_sec = step / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate recent performance
        recent_metrics = {}
        for metric_name, values in self.moving_averages.items():
            if len(values) > 0:
                recent_metrics[metric_name] = np.mean(list(values))
        
        self.logger.info(f"Step {step} - Elapsed: {elapsed_time:.1f}s - "
                        f"Steps/sec: {steps_per_sec:.2f}")
        
        if recent_metrics:
            metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in recent_metrics.items()])
            self.logger.info(f"Recent metrics: {metric_str}")
    
    def plot_training_curves(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot training curves for key metrics.
        
        Args:
            save_path: Path to save the plot (optional)
            show: Whether to display the plot
        """
        if not self.paper_metrics['episode_return']:
            self.logger.warning("No episode return data to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('StateMask Training Progress', fontsize=16)
        
        # Plot configurations for paper metrics
        plot_configs = [
            ('episode_return', 'Episode Return', 'Steps', 'Return'),
            ('masked_return', 'Masked Network Return', 'Steps', 'Return'),
            ('rnd_loss', 'RND Loss', 'Steps', 'Loss'),
            ('explanation_quality', 'Explanation Quality', 'Steps', 'Score'),
            ('mask_sparsity', 'Mask Sparsity', 'Steps', 'Sparsity'),
            ('training_stability', 'Training Stability', 'Steps', 'Stability')
        ]
        
        for idx, (metric_name, title, xlabel, ylabel) in enumerate(plot_configs):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            if metric_name in self.paper_metrics and self.paper_metrics[metric_name]:
                steps, values = zip(*self.paper_metrics[metric_name])
                ax.plot(steps, values, alpha=0.7, label='Raw')
                
                # Add moving average
                if len(values) > 10:
                    window = min(len(values) // 10, 100)
                    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                    smoothed_steps = steps[window-1:]
                    ax.plot(smoothed_steps, smoothed, linewidth=2, label='Smoothed')
                
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No {metric_name} data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training curves saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of training progress."""
        summary = {
            'training_info': {
                'total_steps': self.current_step,
                'total_episodes': self.current_episode,
                'elapsed_time': time.time() - self.start_time,
                'should_stop': self.should_stop,
                'best_metric': self.best_metric
            },
            'performance_metrics': {}
        }
        
        # Calculate performance statistics
        for metric_name, values in self.paper_metrics.items():
            if values:
                _, metric_values = zip(*values)
                summary['performance_metrics'][metric_name] = {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values),
                    'final': metric_values[-1],
                    'count': len(metric_values)
                }
                
                # Recent performance (last 20% of data)
                recent_count = max(1, len(metric_values) // 5)
                recent_values = metric_values[-recent_count:]
                summary['performance_metrics'][metric_name]['recent_mean'] = np.mean(recent_values)
        
        return summary
    
    def save_final_report(self):
        """Save final training report."""
        summary = self.get_summary_statistics()
        
        # Save summary
        summary_file = self.log_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate and save plots
        plot_file = self.log_dir / "training_curves.png"
        self.plot_training_curves(save_path=str(plot_file), show=False)
        
        # Log final summary
        self.logger.info("=== Training Summary ===")
        self.logger.info(f"Total steps: {summary['training_info']['total_steps']}")
        self.logger.info(f"Total episodes: {summary['training_info']['total_episodes']}")
        self.logger.info(f"Elapsed time: {summary['training_info']['elapsed_time']:.1f}s")
        self.logger.info(f"Best metric: {summary['training_info']['best_metric']:.4f}")
        
        if 'episode_return' in summary['performance_metrics']:
            ep_stats = summary['performance_metrics']['episode_return']
            self.logger.info(f"Episode return - Mean: {ep_stats['mean']:.2f}, "
                           f"Std: {ep_stats['std']:.2f}, Final: {ep_stats['final']:.2f}")
        
        self.logger.info(f"Final report saved to {self.log_dir}")
    
    def close(self):
        """Close the monitor and cleanup resources."""
        if self.writer is not None:
            self.writer.close()
        
        self.save_final_report()
        self.logger.info("Training monitor closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class EarlyStoppingCallback:
    """
    Early stopping callback for StateMask training.
    
    Implements early stopping based on validation performance
    with configurable patience and minimum improvement thresholds.
    """
    
    def __init__(self, 
                 monitor_metric: str = 'episode_return',
                 patience: int = 50000,
                 min_delta: float = 0.01,
                 mode: str = 'max',
                 restore_best_weights: bool = True):
        """
        Initialize early stopping callback.
        
        Args:
            monitor_metric: Metric to monitor for early stopping
            patience: Number of steps with no improvement after which training will be stopped
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for maximizing metric, 'min' for minimizing
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_metric = -np.inf if mode == 'max' else np.inf
        self.steps_without_improvement = 0
        self.best_weights = None
        self.should_stop = False
        
        self.compare_fn = np.greater if mode == 'max' else np.less
        
    def __call__(self, current_metric: float, current_weights: Any = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_metric: Current value of the monitored metric
            current_weights: Current model weights (for restoration)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.compare_fn(current_metric - self.min_delta, self.best_metric):
            self.best_metric = current_metric
            self.steps_without_improvement = 0
            if current_weights is not None:
                self.best_weights = current_weights
        else:
            self.steps_without_improvement += 1
        
        if self.steps_without_improvement >= self.patience:
            self.should_stop = True
            return True
        
        return False
    
    def get_best_weights(self):
        """Get the best weights saved during training."""
        return self.best_weights


class PerformanceEvaluator:
    """
    Performance evaluator for StateMask experiments.
    
    Evaluates trained agents on test environments and computes
    metrics relevant to the paper's experimental setup.
    """
    
    def __init__(self, 
                 num_eval_episodes: int = 100,
                 eval_frequency: int = 10000,
                 render: bool = False):
        """
        Initialize performance evaluator.
        
        Args:
            num_eval_episodes: Number of episodes for evaluation
            eval_frequency: Frequency of evaluation (in training steps)
            render: Whether to render during evaluation
        """
        self.num_eval_episodes = num_eval_episodes
        self.eval_frequency = eval_frequency
        self.render = render
        
        self.eval_history = []
        
    def evaluate_agent(self, 
                      agent, 
                      env, 
                      step: int,
                      use_mask: bool = False,
                      mask_network = None) -> Dict[str, float]:
        """
        Evaluate agent performance.
        
        Args:
            agent: The agent to evaluate
            env: Environment for evaluation
            step: Current training step
            use_mask: Whether to use state masking during evaluation
            mask_network: Mask network for state masking
            
        Returns:
            Dictionary of evaluation metrics
        """
        episode_returns = []
        episode_lengths = []
        mask_sparsities = []
        
        for episode in range(self.num_eval_episodes):
            obs = env.reset()
            episode_return = 0
            episode_length = 0
            episode_sparsity = []
            
            done = False
            while not done:
                if use_mask and mask_network is not None:
                    # Apply state masking
                    mask = mask_network.get_mask(obs)
                    masked_obs = obs * mask
                    episode_sparsity.append(np.mean(mask))
                    action = agent.predict(masked_obs, deterministic=True)[0]
                else:
                    action = agent.predict(obs, deterministic=True)[0]
                
                obs, reward, done, info = env.step(action)
                episode_return += reward
                episode_length += 1
                
                if self.render:
                    env.render()
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            if episode_sparsity:
                mask_sparsities.append(np.mean(episode_sparsity))
        
        # Compute evaluation metrics
        eval_metrics = {
            'eval_mean_return': np.mean(episode_returns),
            'eval_std_return': np.std(episode_returns),
            'eval_min_return': np.min(episode_returns),
            'eval_max_return': np.max(episode_returns),
            'eval_mean_length': np.mean(episode_lengths),
            'eval_std_length': np.std(episode_lengths)
        }
        
        if mask_sparsities:
            eval_metrics.update({
                'eval_mean_mask_sparsity': np.mean(mask_sparsities),
                'eval_std_mask_sparsity': np.std(mask_sparsities)
            })
        
        # Store evaluation result
        eval_result = {
            'step': step,
            'metrics': eval_metrics,
            'returns': episode_returns,
            'lengths': episode_lengths
        }
        
        if mask_sparsities:
            eval_result['mask_sparsities'] = mask_sparsities
        
        self.eval_history.append(eval_result)
        
        return eval_metrics
    
    def should_evaluate(self, step: int) -> bool:
        """Check if evaluation should be performed at current step."""
        return step % self.eval_frequency == 0
    
    def get_best_performance(self) -> Dict[str, Any]:
        """Get the best performance achieved during training."""
        if not self.eval_history:
            return {}
        
        best_eval = max(self.eval_history, key=lambda x: x['metrics']['eval_mean_return'])
        return {
            'best_step': best_eval['step'],
            'best_metrics': best_eval['metrics'],
            'best_return': best_eval['metrics']['eval_mean_return']
        }
    
    def plot_evaluation_curves(self, save_path: Optional[str] = None, show: bool = True):
        """Plot evaluation performance over training."""
        if not self.eval_history:
            print("No evaluation data to plot")
            return
        
        steps = [eval_result['step'] for eval_result in self.eval_history]
        mean_returns = [eval_result['metrics']['eval_mean_return'] for eval_result in self.eval_history]
        std_returns = [eval_result['metrics']['eval_std_return'] for eval_result in self.eval_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, mean_returns, 'b-', linewidth=2, label='Mean Return')
        plt.fill_between(steps, 
                        np.array(mean_returns) - np.array(std_returns),
                        np.array(mean_returns) + np.array(std_returns),
                        alpha=0.3, color='blue', label='±1 Std')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Evaluation Return')
        plt.title('Evaluation Performance During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()