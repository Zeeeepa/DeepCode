import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results with statistical information."""
    mean: float
    std: float
    values: List[float]
    confidence_interval: Tuple[float, float]
    sample_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mean': self.mean,
            'std': self.std,
            'values': self.values,
            'confidence_interval': self.confidence_interval,
            'sample_size': self.sample_size
        }

class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        """Compute the metric value."""
        pass
    
    def reset(self):
        """Reset accumulated results."""
        self.results = []
    
    def add_result(self, value: float):
        """Add a single result value."""
        self.results.append(value)
    
    def get_statistics(self, confidence_level: float = 0.95) -> EvaluationResult:
        """
        Compute statistical summary of accumulated results.
        
        Args:
            confidence_level: Confidence level for interval estimation
            
        Returns:
            EvaluationResult with statistical summary
        """
        if not self.results:
            raise ValueError(f"No results available for metric {self.name}")
        
        values = np.array(self.results)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        
        # Compute confidence interval using t-distribution
        if len(values) > 1:
            alpha = 1 - confidence_level
            df = len(values) - 1
            t_critical = stats.t.ppf(1 - alpha/2, df)
            margin_error = t_critical * std / np.sqrt(len(values))
            ci = (mean - margin_error, mean + margin_error)
        else:
            ci = (mean, mean)
        
        return EvaluationResult(
            mean=mean,
            std=std,
            values=self.results.copy(),
            confidence_interval=ci,
            sample_size=len(values)
        )

class AverageReturnMetric(BaseMetric):
    """
    Average Total Return Metric.
    
    Computes the average cumulative reward over multiple test episodes.
    This is the primary evaluation metric as specified in the paper.
    """
    
    def __init__(self):
        super().__init__("Average Total Return")
        self.episode_returns = []
    
    def compute(self, episode_rewards: List[float]) -> float:
        """
        Compute average return for a single evaluation run.
        
        Args:
            episode_rewards: List of cumulative rewards for each episode
            
        Returns:
            Average return across all episodes
        """
        if not episode_rewards:
            return 0.0
        
        avg_return = np.mean(episode_rewards)
        self.episode_returns.extend(episode_rewards)
        return avg_return
    
    def compute_batch(self, batch_rewards: List[List[float]]) -> List[float]:
        """
        Compute average returns for multiple evaluation runs.
        
        Args:
            batch_rewards: List of episode reward lists for each run
            
        Returns:
            List of average returns for each run
        """
        avg_returns = []
        for episode_rewards in batch_rewards:
            avg_return = self.compute(episode_rewards)
            avg_returns.append(avg_return)
            self.add_result(avg_return)
        
        return avg_returns
    
    def get_episode_statistics(self) -> EvaluationResult:
        """Get statistics across all individual episodes."""
        if not self.episode_returns:
            raise ValueError("No episode data available")
        
        values = np.array(self.episode_returns)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        
        # Confidence interval for episodes
        if len(values) > 1:
            alpha = 0.05
            df = len(values) - 1
            t_critical = stats.t.ppf(1 - alpha/2, df)
            margin_error = t_critical * std / np.sqrt(len(values))
            ci = (mean - margin_error, mean + margin_error)
        else:
            ci = (mean, mean)
        
        return EvaluationResult(
            mean=mean,
            std=std,
            values=self.episode_returns.copy(),
            confidence_interval=ci,
            sample_size=len(values)
        )

class EvasionProbabilityMetric(BaseMetric):
    """
    Evasion Detection Probability Metric for Malware Variant Tasks.
    
    Computes the probability that the agent successfully evades detection
    in malware-related environments.
    """
    
    def __init__(self):
        super().__init__("Evasion Probability")
        self.detection_results = []
    
    def compute(self, detection_outcomes: List[bool]) -> float:
        """
        Compute evasion probability for a batch of episodes.
        
        Args:
            detection_outcomes: List of boolean values where True means detected
            
        Returns:
            Probability of evasion (1 - detection_rate)
        """
        if not detection_outcomes:
            return 0.0
        
        detection_rate = np.mean(detection_outcomes)
        evasion_prob = 1.0 - detection_rate
        
        self.detection_results.extend(detection_outcomes)
        return evasion_prob
    
    def compute_from_rewards(self, episode_rewards: List[float], 
                           success_threshold: float = 0.0) -> float:
        """
        Compute evasion probability from episode rewards.
        
        Args:
            episode_rewards: List of cumulative episode rewards
            success_threshold: Threshold above which evasion is considered successful
            
        Returns:
            Probability of successful evasion
        """
        if not episode_rewards:
            return 0.0
        
        successful_evasions = [r > success_threshold for r in episode_rewards]
        return self.compute(successful_evasions)

class FidelityScoreMetric(BaseMetric):
    """
    Fidelity Score Metric.
    
    Computes fidelity score as: log(d/d_max) - log(l/L)
    where:
    - d: reward change after random action at critical step
    - d_max: maximum possible reward change
    - l: critical step window width
    - L: total trajectory length
    """
    
    def __init__(self, d_max: float = 1.0):
        super().__init__("Fidelity Score")
        self.d_max = d_max
        self.trajectory_data = []
    
    def compute(self, reward_change: float, window_width: int, 
                trajectory_length: int) -> float:
        """
        Compute fidelity score for a single trajectory.
        
        Args:
            reward_change: Absolute reward change after random action
            window_width: Width of critical step window
            trajectory_length: Total length of trajectory
            
        Returns:
            Fidelity score
        """
        if reward_change <= 0 or window_width <= 0 or trajectory_length <= 0:
            logger.warning("Invalid parameters for fidelity score computation")
            return float('-inf')
        
        if reward_change > self.d_max:
            logger.warning(f"Reward change {reward_change} exceeds d_max {self.d_max}")
            reward_change = self.d_max
        
        # Compute fidelity score according to paper formula
        d_term = np.log(reward_change / self.d_max)
        l_term = np.log(window_width / trajectory_length)
        
        fidelity_score = d_term - l_term
        
        # Store trajectory data for analysis
        self.trajectory_data.append({
            'reward_change': reward_change,
            'window_width': window_width,
            'trajectory_length': trajectory_length,
            'fidelity_score': fidelity_score
        })
        
        return fidelity_score
    
    def compute_batch(self, trajectory_data: List[Dict[str, Union[float, int]]]) -> List[float]:
        """
        Compute fidelity scores for multiple trajectories.
        
        Args:
            trajectory_data: List of dictionaries with trajectory information
            
        Returns:
            List of fidelity scores
        """
        scores = []
        for data in trajectory_data:
            score = self.compute(
                data['reward_change'],
                data['window_width'],
                data['trajectory_length']
            )
            scores.append(score)
            self.add_result(score)
        
        return scores
    
    def get_trajectory_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of trajectory data."""
        if not self.trajectory_data:
            return {}
        
        data = self.trajectory_data
        analysis = {
            'num_trajectories': len(data),
            'avg_reward_change': np.mean([d['reward_change'] for d in data]),
            'avg_window_width': np.mean([d['window_width'] for d in data]),
            'avg_trajectory_length': np.mean([d['trajectory_length'] for d in data]),
            'reward_change_distribution': {
                'min': min(d['reward_change'] for d in data),
                'max': max(d['reward_change'] for d in data),
                'std': np.std([d['reward_change'] for d in data])
            }
        }
        
        return analysis

class StatisticalSignificanceTest:
    """
    Statistical Significance Testing for Evaluation Results.
    
    Provides various statistical tests to compare different methods
    and determine significance of improvements.
    """
    
    @staticmethod
    def paired_t_test(baseline_results: List[float], 
                     treatment_results: List[float],
                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform paired t-test between baseline and treatment results.
        
        Args:
            baseline_results: Results from baseline method
            treatment_results: Results from treatment method
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        if len(baseline_results) != len(treatment_results):
            raise ValueError("Baseline and treatment must have same number of results")
        
        if len(baseline_results) < 2:
            raise ValueError("Need at least 2 paired observations")
        
        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(treatment_results, baseline_results)
        
        # Effect size (Cohen's d for paired samples)
        differences = np.array(treatment_results) - np.array(baseline_results)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        result = {
            'test_type': 'paired_t_test',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'alpha': alpha,
            'effect_size': effect_size,
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences, ddof=1),
            'sample_size': len(baseline_results)
        }
        
        return result
    
    @staticmethod
    def independent_t_test(group1_results: List[float],
                          group2_results: List[float],
                          alpha: float = 0.05,
                          equal_var: bool = False) -> Dict[str, Any]:
        """
        Perform independent samples t-test.
        
        Args:
            group1_results: Results from first group
            group2_results: Results from second group
            alpha: Significance level
            equal_var: Whether to assume equal variances
            
        Returns:
            Dictionary with test results
        """
        if len(group1_results) < 2 or len(group2_results) < 2:
            raise ValueError("Need at least 2 observations per group")
        
        # Perform independent t-test
        statistic, p_value = stats.ttest_ind(
            group1_results, group2_results, equal_var=equal_var
        )
        
        # Effect size (Cohen's d for independent samples)
        mean1, mean2 = np.mean(group1_results), np.mean(group2_results)
        std1, std2 = np.std(group1_results, ddof=1), np.std(group2_results, ddof=1)
        
        if equal_var:
            pooled_std = np.sqrt(((len(group1_results) - 1) * std1**2 + 
                                 (len(group2_results) - 1) * std2**2) / 
                                (len(group1_results) + len(group2_results) - 2))
            effect_size = (mean1 - mean2) / pooled_std
        else:
            effect_size = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)
        
        result = {
            'test_type': 'independent_t_test',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'alpha': alpha,
            'effect_size': effect_size,
            'mean_difference': mean1 - mean2,
            'equal_var_assumed': equal_var,
            'group1_size': len(group1_results),
            'group2_size': len(group2_results)
        }
        
        return result
    
    @staticmethod
    def wilcoxon_signed_rank_test(baseline_results: List[float],
                                 treatment_results: List[float],
                                 alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
        
        Args:
            baseline_results: Results from baseline method
            treatment_results: Results from treatment method
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        if len(baseline_results) != len(treatment_results):
            raise ValueError("Baseline and treatment must have same number of results")
        
        differences = np.array(treatment_results) - np.array(baseline_results)
        
        # Remove zero differences
        non_zero_diffs = differences[differences != 0]
        
        if len(non_zero_diffs) < 1:
            return {
                'test_type': 'wilcoxon_signed_rank',
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'alpha': alpha,
                'note': 'No non-zero differences found'
            }
        
        statistic, p_value = stats.wilcoxon(non_zero_diffs)
        
        result = {
            'test_type': 'wilcoxon_signed_rank',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'alpha': alpha,
            'median_difference': np.median(differences),
            'sample_size': len(baseline_results),
            'non_zero_differences': len(non_zero_diffs)
        }
        
        return result

class PerformanceImprovementCalculator:
    """
    Performance Improvement Rate Calculator.
    
    Computes various measures of performance improvement relative to baseline.
    """
    
    @staticmethod
    def relative_improvement(baseline_mean: float, 
                           treatment_mean: float) -> float:
        """
        Calculate relative improvement percentage.
        
        Args:
            baseline_mean: Mean performance of baseline method
            treatment_mean: Mean performance of treatment method
            
        Returns:
            Relative improvement as percentage
        """
        if baseline_mean == 0:
            return float('inf') if treatment_mean > 0 else 0.0
        
        return ((treatment_mean - baseline_mean) / abs(baseline_mean)) * 100
    
    @staticmethod
    def absolute_improvement(baseline_mean: float,
                           treatment_mean: float) -> float:
        """
        Calculate absolute improvement.
        
        Args:
            baseline_mean: Mean performance of baseline method
            treatment_mean: Mean performance of treatment method
            
        Returns:
            Absolute improvement
        """
        return treatment_mean - baseline_mean
    
    @staticmethod
    def normalized_improvement(baseline_results: List[float],
                             treatment_results: List[float]) -> Dict[str, float]:
        """
        Calculate normalized improvement metrics.
        
        Args:
            baseline_results: Baseline performance results
            treatment_results: Treatment performance results
            
        Returns:
            Dictionary with various improvement metrics
        """
        baseline_mean = np.mean(baseline_results)
        treatment_mean = np.mean(treatment_results)
        baseline_std = np.std(baseline_results, ddof=1)
        
        # Relative improvement
        rel_improvement = PerformanceImprovementCalculator.relative_improvement(
            baseline_mean, treatment_mean
        )
        
        # Absolute improvement
        abs_improvement = PerformanceImprovementCalculator.absolute_improvement(
            baseline_mean, treatment_mean
        )
        
        # Standardized improvement (improvement in standard deviations)
        std_improvement = abs_improvement / baseline_std if baseline_std > 0 else 0.0
        
        # Success rate improvement (for binary outcomes)
        baseline_success_rate = np.mean([r > 0 for r in baseline_results])
        treatment_success_rate = np.mean([r > 0 for r in treatment_results])
        success_rate_improvement = treatment_success_rate - baseline_success_rate
        
        return {
            'relative_improvement_percent': rel_improvement,
            'absolute_improvement': abs_improvement,
            'standardized_improvement': std_improvement,
            'success_rate_improvement': success_rate_improvement,
            'baseline_mean': baseline_mean,
            'treatment_mean': treatment_mean,
            'baseline_std': baseline_std
        }

class EvaluationMetricsManager:
    """
    Comprehensive Evaluation Metrics Manager.
    
    Manages all evaluation metrics and provides unified interface
    for computing and analyzing results.
    """
    
    def __init__(self, d_max: float = 1.0):
        self.metrics = {
            'average_return': AverageReturnMetric(),
            'evasion_probability': EvasionProbabilityMetric(),
            'fidelity_score': FidelityScoreMetric(d_max=d_max)
        }
        self.statistical_test = StatisticalSignificanceTest()
        self.improvement_calc = PerformanceImprovementCalculator()
        self.results_history = defaultdict(list)
    
    def add_custom_metric(self, name: str, metric: BaseMetric):
        """Add a custom metric to the manager."""
        self.metrics[name] = metric
    
    def evaluate_episode_batch(self, 
                              episode_rewards: List[float],
                              detection_outcomes: Optional[List[bool]] = None,
                              trajectory_data: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Evaluate a batch of episodes across all metrics.
        
        Args:
            episode_rewards: List of cumulative episode rewards
            detection_outcomes: Optional list of detection results for evasion metric
            trajectory_data: Optional trajectory data for fidelity score
            
        Returns:
            Dictionary with computed metric values
        """
        results = {}
        
        # Average return
        avg_return = self.metrics['average_return'].compute(episode_rewards)
        results['average_return'] = avg_return
        
        # Evasion probability
        if detection_outcomes is not None:
            evasion_prob = self.metrics['evasion_probability'].compute(detection_outcomes)
        else:
            # Compute from rewards if detection outcomes not provided
            evasion_prob = self.metrics['evasion_probability'].compute_from_rewards(episode_rewards)
        results['evasion_probability'] = evasion_prob
        
        # Fidelity score
        if trajectory_data is not None:
            fidelity_scores = self.metrics['fidelity_score'].compute_batch(trajectory_data)
            results['fidelity_score'] = np.mean(fidelity_scores) if fidelity_scores else 0.0
        
        # Store results for later analysis
        for metric_name, value in results.items():
            self.results_history[metric_name].append(value)
        
        return results
    
    def get_comprehensive_statistics(self) -> Dict[str, EvaluationResult]:
        """Get comprehensive statistics for all metrics."""
        statistics = {}
        
        for metric_name, metric in self.metrics.items():
            try:
                stats = metric.get_statistics()
                statistics[metric_name] = stats
            except ValueError as e:
                logger.warning(f"Could not compute statistics for {metric_name}: {e}")
        
        return statistics
    
    def compare_methods(self, 
                       baseline_results: Dict[str, List[float]],
                       treatment_results: Dict[str, List[float]],
                       alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """
        Compare two methods across all metrics with statistical testing.
        
        Args:
            baseline_results: Results from baseline method
            treatment_results: Results from treatment method
            alpha: Significance level for statistical tests
            
        Returns:
            Comprehensive comparison results
        """
        comparison_results = {}
        
        for metric_name in baseline_results.keys():
            if metric_name not in treatment_results:
                continue
            
            baseline_vals = baseline_results[metric_name]
            treatment_vals = treatment_results[metric_name]
            
            # Statistical tests
            try:
                # Paired t-test (if same number of observations)
                if len(baseline_vals) == len(treatment_vals):
                    paired_test = self.statistical_test.paired_t_test(
                        baseline_vals, treatment_vals, alpha
                    )
                else:
                    paired_test = None
                
                # Independent t-test
                independent_test = self.statistical_test.independent_t_test(
                    baseline_vals, treatment_vals, alpha
                )
                
                # Wilcoxon test (if paired)
                wilcoxon_test = None
                if len(baseline_vals) == len(treatment_vals):
                    wilcoxon_test = self.statistical_test.wilcoxon_signed_rank_test(
                        baseline_vals, treatment_vals, alpha
                    )
                
                # Improvement metrics
                improvement_metrics = self.improvement_calc.normalized_improvement(
                    baseline_vals, treatment_vals
                )
                
                comparison_results[metric_name] = {
                    'paired_t_test': paired_test,
                    'independent_t_test': independent_test,
                    'wilcoxon_test': wilcoxon_test,
                    'improvement_metrics': improvement_metrics,
                    'baseline_stats': {
                        'mean': np.mean(baseline_vals),
                        'std': np.std(baseline_vals, ddof=1),
                        'n': len(baseline_vals)
                    },
                    'treatment_stats': {
                        'mean': np.mean(treatment_vals),
                        'std': np.std(treatment_vals, ddof=1),
                        'n': len(treatment_vals)
                    }
                }
                
            except Exception as e:
                logger.error(f"Error comparing methods for {metric_name}: {e}")
                comparison_results[metric_name] = {'error': str(e)}
        
        return comparison_results
    
    def generate_evaluation_report(self, 
                                  output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path: Optional path to save report as JSON
            
        Returns:
            Complete evaluation report
        """
        report = {
            'timestamp': str(np.datetime64('now')),
            'metrics_statistics': {},
            'trajectory_analysis': {},
            'summary': {}
        }
        
        # Get statistics for all metrics
        try:
            stats = self.get_comprehensive_statistics()
            for metric_name, stat in stats.items():
                report['metrics_statistics'][metric_name] = stat.to_dict()
        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
        
        # Get trajectory analysis for fidelity score
        try:
            if 'fidelity_score' in self.metrics:
                trajectory_analysis = self.metrics['fidelity_score'].get_trajectory_analysis()
                report['trajectory_analysis'] = trajectory_analysis
        except Exception as e:
            logger.error(f"Error in trajectory analysis: {e}")
        
        # Generate summary
        try:
            summary = {}
            for metric_name, stat_dict in report['metrics_statistics'].items():
                summary[metric_name] = {
                    'mean': stat_dict['mean'],
                    'std': stat_dict['std'],
                    'sample_size': stat_dict['sample_size']
                }
            report['summary'] = summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        
        # Save report if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Evaluation report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report
    
    def reset_all_metrics(self):
        """Reset all metrics and clear history."""
        for metric in self.metrics.values():
            metric.reset()
        self.results_history.clear()
    
    def plot_metric_trends(self, 
                          metric_name: str, 
                          save_path: Optional[str] = None) -> None:
        """
        Plot trends for a specific metric over time.
        
        Args:
            metric_name: Name of metric to plot
            save_path: Optional path to save plot
        """
        if metric_name not in self.results_history:
            logger.warning(f"No history available for metric {metric_name}")
            return
        
        values = self.results_history[metric_name]
        
        plt.figure(figsize=(10, 6))
        plt.plot(values, marker='o', linewidth=2, markersize=4)
        plt.title(f'{metric_name.replace("_", " ").title()} Over Time')
        plt.xlabel('Evaluation Run')
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(values) > 1:
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "--", alpha=0.8, color='red', 
                    label=f'Trend (slope: {z[0]:.4f})')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()

# Unit tests for the evaluation metrics
class TestEvaluationMetrics:
    """Unit tests for evaluation metrics."""
    
    @staticmethod
    def test_average_return_metric():
        """Test average return metric computation."""
        metric = AverageReturnMetric()
        
        # Test single batch
        rewards = [10.0, 20.0, 15.0, 25.0, 18.0]
        avg_return = metric.compute(rewards)
        expected = np.mean(rewards)
        assert abs(avg_return - expected) < 1e-6, f"Expected {expected}, got {avg_return}"
        
        # Test statistics
        metric.add_result(avg_return)
        stats = metric.get_statistics()
        assert stats.sample_size == 1
        assert abs(stats.mean - avg_return) < 1e-6
        
        print("✓ Average return metric test passed")
    
    @staticmethod
    def test_evasion_probability_metric():
        """Test evasion probability metric computation."""
        metric = EvasionProbabilityMetric()
        
        # Test with detection outcomes
        detections = [True, False, False, True, False]  # 2/5 detected
        evasion_prob = metric.compute(detections)
        expected = 1.0 - (2/5)  # 3/5 evaded
        assert abs(evasion_prob - expected) < 1e-6, f"Expected {expected}, got {evasion_prob}"
        
        # Test with rewards
        rewards = [10.0, -5.0, 8.0, -2.0, 15.0]  # 3/5 positive
        evasion_prob_rewards = metric.compute_from_rewards(rewards)
        expected_rewards = 3/5
        assert abs(evasion_prob_rewards - expected_rewards) < 1e-6
        
        print("✓ Evasion probability metric test passed")
    
    @staticmethod
    def test_fidelity_score_metric():
        """Test fidelity score metric computation."""
        metric = FidelityScoreMetric(d_max=1.0)
        
        # Test single computation
        reward_change = 0.5
        window_width = 10
        trajectory_length = 100
        
        score = metric.compute(reward_change, window_width, trajectory_length)
        
        # Manual calculation
        d_term = np.log(reward_change / 1.0)
        l_term = np.log(window_width / trajectory_length)
        expected = d_term - l_term
        
        assert abs(score - expected) < 1e-6, f"Expected {expected}, got {score}"
        
        print("✓ Fidelity score metric test passed")
    
    @staticmethod
    def test_statistical_significance():
        """Test statistical significance testing."""
        # Generate test data
        np.random.seed(42)
        baseline = np.random.normal(10, 2, 20)
        treatment = np.random.normal(12, 2, 20)  # Higher mean
        
        # Paired t-test
        result = StatisticalSignificanceTest.paired_t_test(baseline.tolist(), treatment.tolist())
        assert 'p_value' in result
        assert 'significant' in result
        assert result['sample_size'] == 20
        
        # Independent t-test
        result_ind = StatisticalSignificanceTest.independent_t_test(baseline.tolist(), treatment.tolist())
        assert 'p_value' in result_ind
        assert 'effect_size' in result_ind
        
        print("✓ Statistical significance test passed")
    
    @staticmethod
    def test_performance_improvement():
        """Test performance improvement calculations."""
        baseline_mean = 10.0
        treatment_mean = 12.0
        
        # Relative improvement
        rel_imp = PerformanceImprovementCalculator.relative_improvement(baseline_mean, treatment_mean)
        expected_rel = ((12.0 - 10.0) / 10.0) * 100
        assert abs(rel_imp - expected_rel) < 1e-6
        
        # Absolute improvement
        abs_imp = PerformanceImprovementCalculator.absolute_improvement(baseline_mean, treatment_mean)
        expected_abs = 2.0
        assert abs(abs_imp - expected_abs) < 1e-6
        
        print("✓ Performance improvement calculation test passed")
    
    @staticmethod
    def run_all_tests():
        """Run all unit tests."""
        print("Running evaluation metrics unit tests...")
        TestEvaluationMetrics.test_average_return_metric()
        TestEvaluationMetrics.test_evasion_probability_metric()
        TestEvaluationMetrics.test_fidelity_score_metric()
        TestEvaluationMetrics.test_statistical_significance()
        TestEvaluationMetrics.test_performance_improvement()
        print("✓ All tests passed!")

if __name__ == "__main__":
    # Run unit tests
    TestEvaluationMetrics.run_all_tests()
    
    # Example usage
    print("\nExample usage:")
    
    # Initialize metrics manager
    manager = EvaluationMetricsManager(d_max=1.0)
    
    # Simulate evaluation data
    np.random.seed(42)
    
    # Simulate multiple evaluation runs
    for run in range(5):
        # Generate episode rewards
        episode_rewards = np.random.normal(15, 3, 100).tolist()
        
        # Generate detection outcomes
        detection_outcomes = (np.random.random(100) < 0.3).tolist()
        
        # Generate trajectory data
        trajectory_data = []
        for i in range(10):
            trajectory_data.append({
                'reward_change': np.random.uniform(0.1, 0.8),
                'window_width': np.random.randint(5, 20),
                'trajectory_length': np.random.randint(50, 150)
            })
        
        # Evaluate batch
        results = manager.evaluate_episode_batch(
            episode_rewards, detection_outcomes, trajectory_data
        )
        
        print(f"Run {run + 1} results: {results}")
    
    # Get comprehensive statistics
    stats = manager.get_comprehensive_statistics()
    print(f"\nFinal statistics:")
    for metric_name, stat in stats.items():
        print(f"{metric_name}: {stat.mean:.4f} ± {stat.std:.4f}")
    
    # Generate report
    report = manager.generate_evaluation_report("evaluation_report.json")
    print(f"\nReport generated with {len(report['metrics_statistics'])} metrics")