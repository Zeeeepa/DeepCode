import numpy as np
import torch
import gym
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from datetime import datetime
import logging

class EvaluationManager:
    """
    严格按照论文评估协议的评估管理器
    
    论文评估协议:
    - 中间评估: 每10000步进行一次，使用10-100个episodes
    - 最终评估: 训练结束后使用500个episodes
    - 多随机种子: 3-5次独立实验
    - 统计分析: 报告均值和标准差，进行显著性测试
    - 确定性评估: 关闭探索机制
    """
    
    def __init__(self, 
                 env_name: str,
                 eval_episodes_intermediate: int = 50,
                 eval_episodes_final: int = 500,
                 eval_interval: int = 10000,
                 num_seeds: int = 3,
                 save_dir: str = "evaluation_results",
                 paper_results: Optional[Dict] = None):
        """
        初始化评估管理器
        
        Args:
            env_name: 环境名称
            eval_episodes_intermediate: 中间评估的episode数量
            eval_episodes_final: 最终评估的episode数量
            eval_interval: 评估间隔步数
            num_seeds: 随机种子数量
            save_dir: 结果保存目录
            paper_results: 论文报告的结果用于对比
        """
        self.env_name = env_name
        self.eval_episodes_intermediate = eval_episodes_intermediate
        self.eval_episodes_final = eval_episodes_final
        self.eval_interval = eval_interval
        self.num_seeds = num_seeds
        self.save_dir = save_dir
        self.paper_results = paper_results or {}
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化结果存储
        self.results = defaultdict(list)  # {seed: [evaluation_results]}
        self.intermediate_results = defaultdict(list)  # {seed: [(step, result)]}
        self.final_results = {}  # {seed: final_result}
        
        # 设置日志
        self._setup_logging()
        
        # 论文Table 1的参考结果 (示例，需要根据实际论文调整)
        self.paper_table1_results = {
            'HalfCheetah-v2': {'mean': 4000.0, 'std': 200.0},
            'Walker2d-v2': {'mean': 3500.0, 'std': 150.0},
            'Hopper-v2': {'mean': 2500.0, 'std': 100.0},
            'Ant-v2': {'mean': 5000.0, 'std': 300.0},
        }
        
    def _setup_logging(self):
        """设置日志系统"""
        log_file = os.path.join(self.save_dir, f"evaluation_{self.env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_eval_env(self, seed: int = None):
        """
        创建独立的测试环境
        
        Args:
            seed: 随机种子
            
        Returns:
            gym.Env: 测试环境
        """
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)
            env.action_space.seed(seed)
        return env
        
    def evaluate_agent(self, 
                      agent, 
                      num_episodes: int, 
                      seed: int = None,
                      deterministic: bool = True,
                      render: bool = False) -> Dict[str, float]:
        """
        评估智能体性能
        
        Args:
            agent: 要评估的智能体
            num_episodes: 评估episode数量
            seed: 随机种子
            deterministic: 是否使用确定性策略
            render: 是否渲染环境
            
        Returns:
            Dict: 评估结果统计
        """
        env = self.create_eval_env(seed)
        
        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        
        self.logger.info(f"开始评估 - Episodes: {num_episodes}, Deterministic: {deterministic}")
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # 获取动作 (确定性策略用于评估)
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(obs, deterministic=deterministic)
                elif hasattr(agent, 'act'):
                    action = agent.act(obs, deterministic=deterministic)
                else:
                    raise AttributeError("Agent must have 'select_action' or 'act' method")
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if render:
                    env.render()
                    
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 检查是否成功 (某些环境有成功标准)
            if 'is_success' in info:
                success_rate += info['is_success']
                
            if (episode + 1) % max(1, num_episodes // 10) == 0:
                self.logger.info(f"Episode {episode + 1}/{num_episodes} completed")
                
        env.close()
        
        # 计算统计信息
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_rate / num_episodes if num_episodes > 0 else 0,
            'num_episodes': num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        self.logger.info(f"评估完成 - 平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        
        return results
        
    def intermediate_evaluation(self, 
                              agent, 
                              step: int, 
                              seed: int) -> Dict[str, float]:
        """
        中间评估 (每10000步进行一次)
        
        Args:
            agent: 智能体
            step: 当前训练步数
            seed: 随机种子
            
        Returns:
            Dict: 评估结果
        """
        self.logger.info(f"中间评估 - Step: {step}, Seed: {seed}")
        
        results = self.evaluate_agent(
            agent=agent,
            num_episodes=self.eval_episodes_intermediate,
            seed=seed,
            deterministic=True
        )
        
        # 添加步数信息
        results['step'] = step
        results['seed'] = seed
        results['evaluation_type'] = 'intermediate'
        
        # 存储结果
        self.intermediate_results[seed].append((step, results))
        
        return results
        
    def final_evaluation(self, 
                        agent, 
                        seed: int) -> Dict[str, float]:
        """
        最终评估 (500个episodes)
        
        Args:
            agent: 智能体
            seed: 随机种子
            
        Returns:
            Dict: 评估结果
        """
        self.logger.info(f"最终评估 - Seed: {seed}")
        
        results = self.evaluate_agent(
            agent=agent,
            num_episodes=self.eval_episodes_final,
            seed=seed,
            deterministic=True
        )
        
        # 添加种子信息
        results['seed'] = seed
        results['evaluation_type'] = 'final'
        
        # 存储结果
        self.final_results[seed] = results
        
        return results
        
    def multi_seed_evaluation(self, 
                            agent_factory, 
                            training_function,
                            seeds: List[int] = None) -> Dict[str, Any]:
        """
        多随机种子实验管理
        
        Args:
            agent_factory: 智能体工厂函数
            training_function: 训练函数
            seeds: 随机种子列表
            
        Returns:
            Dict: 汇总结果
        """
        if seeds is None:
            seeds = list(range(self.num_seeds))
            
        self.logger.info(f"开始多种子实验 - Seeds: {seeds}")
        
        all_final_results = []
        
        for seed in seeds:
            self.logger.info(f"开始种子 {seed} 的实验")
            
            # 设置随机种子
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                
            # 创建智能体
            agent = agent_factory(seed=seed)
            
            # 训练智能体 (包含中间评估)
            training_function(
                agent=agent,
                seed=seed,
                evaluation_callback=self.intermediate_evaluation
            )
            
            # 最终评估
            final_result = self.final_evaluation(agent, seed)
            all_final_results.append(final_result)
            
        # 计算汇总统计
        summary = self._compute_summary_statistics(all_final_results)
        
        # 保存结果
        self._save_results(summary)
        
        return summary
        
    def _compute_summary_statistics(self, 
                                  results: List[Dict]) -> Dict[str, Any]:
        """
        计算汇总统计信息
        
        Args:
            results: 各种子的结果列表
            
        Returns:
            Dict: 汇总统计
        """
        if not results:
            return {}
            
        # 提取关键指标
        mean_rewards = [r['mean_reward'] for r in results]
        std_rewards = [r['std_reward'] for r in results]
        
        # 计算跨种子统计
        summary = {
            'num_seeds': len(results),
            'mean_reward_across_seeds': np.mean(mean_rewards),
            'std_reward_across_seeds': np.std(mean_rewards),
            'min_reward_across_seeds': np.min(mean_rewards),
            'max_reward_across_seeds': np.max(mean_rewards),
            'median_reward_across_seeds': np.median(mean_rewards),
            'mean_std_within_seed': np.mean(std_rewards),
            'individual_results': results,
            'confidence_interval_95': self._compute_confidence_interval(mean_rewards, 0.95),
            'confidence_interval_99': self._compute_confidence_interval(mean_rewards, 0.99)
        }
        
        # 统计显著性测试
        if self.env_name in self.paper_table1_results:
            summary['statistical_tests'] = self._perform_statistical_tests(
                mean_rewards, 
                self.paper_table1_results[self.env_name]
            )
            
        return summary
        
    def _compute_confidence_interval(self, 
                                   data: List[float], 
                                   confidence: float) -> Tuple[float, float]:
        """
        计算置信区间
        
        Args:
            data: 数据列表
            confidence: 置信水平
            
        Returns:
            Tuple: (下界, 上界)
        """
        if len(data) < 2:
            return (np.mean(data), np.mean(data))
            
        mean = np.mean(data)
        sem = stats.sem(data)  # 标准误差
        h = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
        
        return (mean - h, mean + h)
        
    def _perform_statistical_tests(self, 
                                 our_results: List[float],
                                 paper_results: Dict[str, float]) -> Dict[str, Any]:
        """
        执行统计显著性测试
        
        Args:
            our_results: 我们的实验结果
            paper_results: 论文报告的结果
            
        Returns:
            Dict: 统计测试结果
        """
        tests = {}
        
        # 单样本t检验 (与论文均值比较)
        paper_mean = paper_results['mean']
        t_stat, p_value = stats.ttest_1samp(our_results, paper_mean)
        
        tests['one_sample_ttest'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'paper_mean': paper_mean,
            'our_mean': np.mean(our_results),
            'difference': np.mean(our_results) - paper_mean,
            'relative_difference_percent': ((np.mean(our_results) - paper_mean) / paper_mean) * 100
        }
        
        # 效应大小 (Cohen's d)
        our_std = np.std(our_results, ddof=1)
        cohens_d = (np.mean(our_results) - paper_mean) / our_std
        tests['effect_size'] = {
            'cohens_d': cohens_d,
            'interpretation': self._interpret_cohens_d(cohens_d)
        }
        
        return tests
        
    def _interpret_cohens_d(self, d: float) -> str:
        """解释Cohen's d效应大小"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
            
    def compare_with_paper_table1(self, 
                                summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        与论文Table 1结果自动对比
        
        Args:
            summary: 实验汇总结果
            
        Returns:
            Dict: 对比分析结果
        """
        if self.env_name not in self.paper_table1_results:
            self.logger.warning(f"没有找到环境 {self.env_name} 的论文参考结果")
            return {}
            
        paper_result = self.paper_table1_results[self.env_name]
        our_result = {
            'mean': summary['mean_reward_across_seeds'],
            'std': summary['std_reward_across_seeds']
        }
        
        comparison = {
            'environment': self.env_name,
            'paper_result': paper_result,
            'our_result': our_result,
            'absolute_difference': our_result['mean'] - paper_result['mean'],
            'relative_difference_percent': ((our_result['mean'] - paper_result['mean']) / paper_result['mean']) * 100,
            'performance_ratio': our_result['mean'] / paper_result['mean'],
            'std_comparison': {
                'paper_std': paper_result['std'],
                'our_std': our_result['std'],
                'std_ratio': our_result['std'] / paper_result['std']
            }
        }
        
        # 性能评级
        ratio = comparison['performance_ratio']
        if ratio >= 0.98:
            comparison['performance_grade'] = 'Excellent (≥98% of paper)'
        elif ratio >= 0.95:
            comparison['performance_grade'] = 'Very Good (≥95% of paper)'
        elif ratio >= 0.90:
            comparison['performance_grade'] = 'Good (≥90% of paper)'
        elif ratio >= 0.80:
            comparison['performance_grade'] = 'Acceptable (≥80% of paper)'
        else:
            comparison['performance_grade'] = 'Needs Improvement (<80% of paper)'
            
        return comparison
        
    def _save_results(self, summary: Dict[str, Any]):
        """保存评估结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON格式的详细结果
        results_file = os.path.join(self.save_dir, f"evaluation_results_{self.env_name}_{timestamp}.json")
        with open(results_file, 'w') as f:
            # 转换numpy类型为Python原生类型以便JSON序列化
            json_summary = self._convert_for_json(summary)
            json.dump(json_summary, f, indent=2)
            
        # 保存CSV格式的简化结果
        csv_file = os.path.join(self.save_dir, f"evaluation_summary_{self.env_name}_{timestamp}.csv")
        self._save_csv_summary(summary, csv_file)
        
        # 生成可视化图表
        self._generate_plots(summary, timestamp)
        
        self.logger.info(f"结果已保存到: {results_file}")
        
    def _convert_for_json(self, obj):
        """转换对象为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
            
    def _save_csv_summary(self, summary: Dict[str, Any], csv_file: str):
        """保存CSV格式的汇总结果"""
        data = []
        for i, result in enumerate(summary.get('individual_results', [])):
            data.append({
                'seed': result.get('seed', i),
                'mean_reward': result['mean_reward'],
                'std_reward': result['std_reward'],
                'min_reward': result['min_reward'],
                'max_reward': result['max_reward'],
                'median_reward': result['median_reward'],
                'success_rate': result.get('success_rate', 0),
                'num_episodes': result['num_episodes']
            })
            
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        
    def _generate_plots(self, summary: Dict[str, Any], timestamp: str):
        """生成可视化图表"""
        try:
            # 设置图表样式
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Evaluation Results - {self.env_name}', fontsize=16)
            
            # 1. 各种子性能对比
            seeds = [r.get('seed', i) for i, r in enumerate(summary.get('individual_results', []))]
            mean_rewards = [r['mean_reward'] for r in summary.get('individual_results', [])]
            std_rewards = [r['std_reward'] for r in summary.get('individual_results', [])]
            
            axes[0, 0].bar(seeds, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
            axes[0, 0].set_title('Mean Reward by Seed')
            axes[0, 0].set_xlabel('Seed')
            axes[0, 0].set_ylabel('Mean Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 奖励分布
            all_rewards = []
            for result in summary.get('individual_results', []):
                all_rewards.extend(result.get('episode_rewards', []))
                
            if all_rewards:
                axes[0, 1].hist(all_rewards, bins=50, alpha=0.7, density=True)
                axes[0, 1].axvline(np.mean(all_rewards), color='red', linestyle='--', 
                                 label=f'Mean: {np.mean(all_rewards):.2f}')
                axes[0, 1].set_title('Reward Distribution')
                axes[0, 1].set_xlabel('Reward')
                axes[0, 1].set_ylabel('Density')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 学习曲线 (如果有中间评估结果)
            if self.intermediate_results:
                for seed in self.intermediate_results:
                    steps = [step for step, _ in self.intermediate_results[seed]]
                    rewards = [result['mean_reward'] for _, result in self.intermediate_results[seed]]
                    axes[1, 0].plot(steps, rewards, label=f'Seed {seed}', alpha=0.7)
                    
                axes[1, 0].set_title('Learning Curve')
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Mean Reward')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 与论文结果对比
            if self.env_name in self.paper_table1_results:
                paper_mean = self.paper_table1_results[self.env_name]['mean']
                our_mean = summary['mean_reward_across_seeds']
                
                categories = ['Paper', 'Ours']
                values = [paper_mean, our_mean]
                colors = ['blue', 'orange']
                
                bars = axes[1, 1].bar(categories, values, color=colors, alpha=0.7)
                axes[1, 1].set_title('Comparison with Paper Results')
                axes[1, 1].set_ylabel('Mean Reward')
                axes[1, 1].grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = os.path.join(self.save_dir, f"evaluation_plots_{self.env_name}_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"图表已保存到: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"生成图表时出错: {e}")
            
    def generate_final_report(self, summary: Dict[str, Any]) -> str:
        """
        生成最终评估报告
        
        Args:
            summary: 实验汇总结果
            
        Returns:
            str: 格式化的报告文本
        """
        report = []
        report.append("=" * 80)
        report.append(f"FINAL EVALUATION REPORT - {self.env_name}")
        report.append("=" * 80)
        report.append("")
        
        # 基本信息
        report.append("EXPERIMENT CONFIGURATION:")
        report.append(f"  Environment: {self.env_name}")
        report.append(f"  Number of seeds: {summary.get('num_seeds', 'N/A')}")
        report.append(f"  Episodes per final evaluation: {self.eval_episodes_final}")
        report.append(f"  Intermediate evaluation episodes: {self.eval_episodes_intermediate}")
        report.append(f"  Evaluation interval: {self.eval_interval} steps")
        report.append("")
        
        # 主要结果
        report.append("MAIN RESULTS:")
        report.append(f"  Mean reward across seeds: {summary.get('mean_reward_across_seeds', 0):.2f} ± {summary.get('std_reward_across_seeds', 0):.2f}")
        report.append(f"  Min reward: {summary.get('min_reward_across_seeds', 0):.2f}")
        report.append(f"  Max reward: {summary.get('max_reward_across_seeds', 0):.2f}")
        report.append(f"  Median reward: {summary.get('median_reward_across_seeds', 0):.2f}")
        
        # 置信区间
        ci_95 = summary.get('confidence_interval_95', (0, 0))
        ci_99 = summary.get('confidence_interval_99', (0, 0))
        report.append(f"  95% Confidence Interval: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
        report.append(f"  99% Confidence Interval: [{ci_99[0]:.2f}, {ci_99[1]:.2f}]")
        report.append("")
        
        # 与论文对比
        comparison = self.compare_with_paper_table1(summary)
        if comparison:
            report.append("COMPARISON WITH PAPER RESULTS:")
            report.append(f"  Paper result: {comparison['paper_result']['mean']:.2f} ± {comparison['paper_result']['std']:.2f}")
            report.append(f"  Our result: {comparison['our_result']['mean']:.2f} ± {comparison['our_result']['std']:.2f}")
            report.append(f"  Absolute difference: {comparison['absolute_difference']:.2f}")
            report.append(f"  Relative difference: {comparison['relative_difference_percent']:.1f}%")
            report.append(f"  Performance ratio: {comparison['performance_ratio']:.3f}")
            report.append(f"  Performance grade: {comparison['performance_grade']}")
            report.append("")
        
        # 统计测试
        if 'statistical_tests' in summary:
            tests = summary['statistical_tests']
            if 'one_sample_ttest' in tests:
                ttest = tests['one_sample_ttest']
                report.append("STATISTICAL SIGNIFICANCE TESTS:")
                report.append(f"  One-sample t-test vs paper mean:")
                report.append(f"    t-statistic: {ttest['t_statistic']:.3f}")
                report.append(f"    p-value: {ttest['p_value']:.6f}")
                report.append(f"    Significant at α=0.05: {ttest['significant_at_0.05']}")
                report.append(f"    Significant at α=0.01: {ttest['significant_at_0.01']}")
                
            if 'effect_size' in tests:
                effect = tests['effect_size']
                report.append(f"  Effect size (Cohen's d): {effect['cohens_d']:.3f} ({effect['interpretation']})")
                report.append("")
        
        # 各种子详细结果
        report.append("DETAILED RESULTS BY SEED:")
        for i, result in enumerate(summary.get('individual_results', [])):
            seed = result.get('seed', i)
            report.append(f"  Seed {seed}:")
            report.append(f"    Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            report.append(f"    Min/Max: {result['min_reward']:.2f} / {result['max_reward']:.2f}")
            report.append(f"    Episodes: {result['num_episodes']}")
            if 'success_rate' in result:
                report.append(f"    Success rate: {result['success_rate']:.3f}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.save_dir, f"final_report_{self.env_name}_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report_text)
            
        self.logger.info(f"最终报告已保存到: {report_file}")
        
        return report_text

# 使用示例和工具函数
def create_evaluation_callback(evaluation_manager: EvaluationManager, seed: int):
    """
    创建用于训练过程中的评估回调函数
    
    Args:
        evaluation_manager: 评估管理器
        seed: 当前种子
        
    Returns:
        function: 评估回调函数
    """
    def evaluation_callback(agent, step):
        if step % evaluation_manager.eval_interval == 0:
            evaluation_manager.intermediate_evaluation(agent, step, seed)
    
    return evaluation_callback

def run_complete_evaluation(agent_factory, 
                          training_function,
                          env_name: str,
                          num_seeds: int = 3,
                          save_dir: str = "evaluation_results"):
    """
    运行完整的评估流程
    
    Args:
        agent_factory: 智能体工厂函数
        training_function: 训练函数
        env_name: 环境名称
        num_seeds: 随机种子数量
        save_dir: 保存目录
        
    Returns:
        Dict: 完整的评估结果
    """
    # 创建评估管理器
    evaluator = EvaluationManager(
        env_name=env_name,
        num_seeds=num_seeds,
        save_dir=save_dir
    )
    
    # 运行多种子实验
    summary = evaluator.multi_seed_evaluation(
        agent_factory=agent_factory,
        training_function=training_function
    )
    
    # 生成最终报告
    report = evaluator.generate_final_report(summary)
    print(report)
    
    return summary