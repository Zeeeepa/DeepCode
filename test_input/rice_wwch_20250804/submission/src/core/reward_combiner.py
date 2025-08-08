import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

class RewardCombiner(nn.Module):
    """
    奖励组合模块，实现论文中的奖励组合策略
    
    论文公式实现：
    1. 总奖励: R'_t = R_t + λ*normalize(R_RND)
    2. 掩码训练奖励: R'_t = R_t + α*a_t^m
    
    其中：
    - R_t: 环境任务奖励
    - R_RND: RND探索奖励
    - λ: 探索奖励权重系数
    - α: 掩码奖励权重系数
    - a_t^m: 掩码动作奖励
    - normalize(): 基于运行时统计的归一化函数
    
    关键改进：
    - 实现论文中提到的RND奖励归一化机制
    - 防止RND奖励主导任务奖励
    - 基于运行时统计进行动态归一化
    """
    
    def __init__(
        self,
        exploration_weight: float = 1.0,
        mask_weight: float = 0.1,
        reward_normalization: bool = True,
        rnd_reward_scale: float = 1.0,
        task_reward_scale: float = 1.0,
        rnd_reward_clip: float = 5.0,  # 论文中提到的RND奖励裁剪
        normalization_method: str = 'running_mean_std',  # 归一化方法
        device: str = 'cpu'
    ):
        """
        初始化奖励组合器
        
        Args:
            exploration_weight (float): λ - 探索奖励权重系数
            mask_weight (float): α - 掩码奖励权重系数
            reward_normalization (bool): 是否进行奖励归一化
            rnd_reward_scale (float): RND奖励缩放因子
            task_reward_scale (float): 任务奖励缩放因子
            rnd_reward_clip (float): RND奖励裁剪阈值
            normalization_method (str): 归一化方法 ('running_mean_std', 'min_max', 'robust')
            device (str): 计算设备
        """
        super(RewardCombiner, self).__init__()
        
        self.exploration_weight = exploration_weight  # λ
        self.mask_weight = mask_weight  # α
        self.reward_normalization = reward_normalization
        self.rnd_reward_scale = rnd_reward_scale
        self.task_reward_scale = task_reward_scale
        self.rnd_reward_clip = rnd_reward_clip
        self.normalization_method = normalization_method
        self.device = device
        
        # 奖励统计信息，用于归一化 - 使用更稳定的统计方法
        self.register_buffer('task_reward_mean', torch.zeros(1))
        self.register_buffer('task_reward_std', torch.ones(1))
        self.register_buffer('task_reward_var', torch.ones(1))
        self.register_buffer('task_reward_count', torch.zeros(1))
        
        self.register_buffer('rnd_reward_mean', torch.zeros(1))
        self.register_buffer('rnd_reward_std', torch.ones(1))
        self.register_buffer('rnd_reward_var', torch.ones(1))
        self.register_buffer('rnd_reward_count', torch.zeros(1))
        
        # 用于robust归一化的分位数统计
        self.register_buffer('rnd_reward_median', torch.zeros(1))
        self.register_buffer('rnd_reward_mad', torch.ones(1))  # Median Absolute Deviation
        
        # 移动平均参数 - 论文中使用的参数
        self.momentum = 0.99
        self.epsilon = 1e-8
        self.min_std = 1e-4  # 防止标准差过小
        
        # 初始化标志
        self.initialized = False
        self.warmup_steps = 1000  # 预热步数，确保统计稳定
        self.update_count = 0
        
        # 奖励历史缓存，用于更精确的统计计算
        self.reward_history_size = 10000
        self.rnd_reward_history = []
        self.task_reward_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def _welford_update(self, existing_mean, existing_var, existing_count, new_values):
        """
        使用Welford算法进行在线均值和方差更新，数值更稳定
        
        Args:
            existing_mean: 当前均值
            existing_var: 当前方差
            existing_count: 当前样本数
            new_values: 新的数值
            
        Returns:
            updated_mean, updated_var, updated_count
        """
        new_count = len(new_values)
        total_count = existing_count + new_count
        
        if existing_count == 0:
            new_mean = new_values.mean()
            new_var = new_values.var(unbiased=False) if new_count > 1 else torch.ones_like(new_values.var())
        else:
            # Welford算法更新
            delta = new_values.mean() - existing_mean
            new_mean = existing_mean + delta * new_count / total_count
            
            # 更新方差
            delta2 = new_values.mean() - new_mean
            new_var = (existing_var * existing_count + 
                      new_values.var(unbiased=False) * new_count + 
                      delta * delta2 * existing_count * new_count / total_count) / total_count
        
        return new_mean, new_var, total_count
    
    def update_reward_statistics(
        self,
        task_rewards: torch.Tensor,
        rnd_rewards: torch.Tensor
    ) -> None:
        """
        更新奖励统计信息，使用更稳定的在线算法
        
        Args:
            task_rewards (torch.Tensor): 任务奖励 [batch_size]
            rnd_rewards (torch.Tensor): RND奖励 [batch_size]
        """
        if not self.reward_normalization:
            return
        
        self.update_count += 1
        
        # 更新奖励历史
        self.task_reward_history.extend(task_rewards.detach().cpu().numpy().tolist())
        self.rnd_reward_history.extend(rnd_rewards.detach().cpu().numpy().tolist())
        
        # 保持历史大小
        if len(self.task_reward_history) > self.reward_history_size:
            self.task_reward_history = self.task_reward_history[-self.reward_history_size:]
        if len(self.rnd_reward_history) > self.reward_history_size:
            self.rnd_reward_history = self.rnd_reward_history[-self.reward_history_size:]
        
        # 使用Welford算法更新统计信息
        task_mean, task_var, task_count = self._welford_update(
            self.task_reward_mean, self.task_reward_var, self.task_reward_count, task_rewards
        )
        
        rnd_mean, rnd_var, rnd_count = self._welford_update(
            self.rnd_reward_mean, self.rnd_reward_var, self.rnd_reward_count, rnd_rewards
        )
        
        # 更新缓冲区
        self.task_reward_mean.copy_(task_mean)
        self.task_reward_var.copy_(task_var)
        self.task_reward_std.copy_(torch.sqrt(task_var + self.epsilon).clamp(min=self.min_std))
        self.task_reward_count.copy_(task_count)
        
        self.rnd_reward_mean.copy_(rnd_mean)
        self.rnd_reward_var.copy_(rnd_var)
        self.rnd_reward_std.copy_(torch.sqrt(rnd_var + self.epsilon).clamp(min=self.min_std))
        self.rnd_reward_count.copy_(rnd_count)
        
        # 更新robust统计信息（用于robust归一化）
        if self.normalization_method == 'robust' and len(self.rnd_reward_history) > 100:
            rnd_history_tensor = torch.tensor(self.rnd_reward_history, device=self.device)
            self.rnd_reward_median.copy_(rnd_history_tensor.median())
            mad = torch.median(torch.abs(rnd_history_tensor - self.rnd_reward_median))
            self.rnd_reward_mad.copy_(mad.clamp(min=self.min_std))
        
        # 标记为已初始化
        if not self.initialized and self.update_count >= 10:
            self.initialized = True
            self.logger.info("Reward statistics initialized after warmup")
    
    def normalize_rnd_rewards(self, rnd_rewards: torch.Tensor) -> torch.Tensor:
        """
        归一化RND奖励，实现论文中的normalize(R_RND)
        
        Args:
            rnd_rewards (torch.Tensor): RND奖励 [batch_size]
            
        Returns:
            torch.Tensor: 归一化后的RND奖励
        """
        if not self.reward_normalization or not self.initialized:
            # 未初始化时使用简单裁剪
            return torch.clamp(rnd_rewards, -self.rnd_reward_clip, self.rnd_reward_clip)
        
        if self.normalization_method == 'running_mean_std':
            # 标准Z-score归一化
            normalized_rnd = (rnd_rewards - self.rnd_reward_mean) / self.rnd_reward_std
        elif self.normalization_method == 'robust':
            # 基于中位数的robust归一化，对异常值更鲁棒
            normalized_rnd = (rnd_rewards - self.rnd_reward_median) / (1.4826 * self.rnd_reward_mad)
        elif self.normalization_method == 'min_max':
            # Min-Max归一化到[-1, 1]
            if len(self.rnd_reward_history) > 100:
                rnd_min = min(self.rnd_reward_history)
                rnd_max = max(self.rnd_reward_history)
                rnd_range = rnd_max - rnd_min
                if rnd_range > self.epsilon:
                    normalized_rnd = 2 * (rnd_rewards - rnd_min) / rnd_range - 1
                else:
                    normalized_rnd = torch.zeros_like(rnd_rewards)
            else:
                normalized_rnd = rnd_rewards
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        # 应用裁剪防止极端值
        normalized_rnd = torch.clamp(normalized_rnd, -self.rnd_reward_clip, self.rnd_reward_clip)
        
        return normalized_rnd
    
    def normalize_task_rewards(self, task_rewards: torch.Tensor) -> torch.Tensor:
        """
        归一化任务奖励（可选）
        
        Args:
            task_rewards (torch.Tensor): 任务奖励 [batch_size]
            
        Returns:
            torch.Tensor: 归一化后的任务奖励
        """
        if not self.reward_normalization or not self.initialized:
            return task_rewards
        
        # 对任务奖励进行轻度归一化，保持其原始scale的相对重要性
        normalized_task = (task_rewards - self.task_reward_mean) / self.task_reward_std
        return normalized_task
    
    def combine_exploration_rewards(
        self,
        task_rewards: torch.Tensor,
        rnd_rewards: torch.Tensor,
        update_stats: bool = True
    ) -> torch.Tensor:
        """
        组合任务奖励和探索奖励
        
        实现论文公式: R'_t = R_t + λ*normalize(R_RND)
        
        Args:
            task_rewards (torch.Tensor): 环境任务奖励 R_t [batch_size]
            rnd_rewards (torch.Tensor): RND探索奖励 R_RND [batch_size]
            update_stats (bool): 是否更新统计信息
            
        Returns:
            torch.Tensor: 组合后的总奖励 R'_t [batch_size]
        """
        # 确保输入为tensor
        if not isinstance(task_rewards, torch.Tensor):
            task_rewards = torch.tensor(task_rewards, dtype=torch.float32, device=self.device)
        if not isinstance(rnd_rewards, torch.Tensor):
            rnd_rewards = torch.tensor(rnd_rewards, dtype=torch.float32, device=self.device)
        
        # 移动到正确设备
        task_rewards = task_rewards.to(self.device)
        rnd_rewards = rnd_rewards.to(self.device)
        
        # 应用初始缩放因子
        scaled_task = task_rewards * self.task_reward_scale
        scaled_rnd = rnd_rewards * self.rnd_reward_scale
        
        # 更新统计信息（在归一化之前）
        if update_stats:
            self.update_reward_statistics(scaled_task, scaled_rnd)
        
        # 归一化RND奖励 - 这是论文中的关键步骤
        normalized_rnd = self.normalize_rnd_rewards(scaled_rnd)
        
        # 可选：轻度归一化任务奖励
        if self.reward_normalization and self.initialized:
            normalized_task = self.normalize_task_rewards(scaled_task)
        else:
            normalized_task = scaled_task
        
        # 组合奖励: R'_t = R_t + λ*normalize(R_RND)
        combined_rewards = normalized_task + self.exploration_weight * normalized_rnd
        
        # 记录调试信息
        if self.update_count % 1000 == 0 and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Reward combination stats - "
                            f"Task: {normalized_task.mean().item():.4f}±{normalized_task.std().item():.4f}, "
                            f"RND: {normalized_rnd.mean().item():.4f}±{normalized_rnd.std().item():.4f}, "
                            f"Combined: {combined_rewards.mean().item():.4f}±{combined_rewards.std().item():.4f}")
        
        return combined_rewards
    
    def combine_mask_rewards(
        self,
        task_rewards: torch.Tensor,
        mask_action_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        组合任务奖励和掩码动作奖励，用于掩码网络训练
        
        实现论文公式: R'_t = R_t + α*a_t^m
        
        Args:
            task_rewards (torch.Tensor): 环境任务奖励 R_t [batch_size]
            mask_action_rewards (torch.Tensor): 掩码动作奖励 a_t^m [batch_size]
            
        Returns:
            torch.Tensor: 组合后的掩码训练奖励 R'_t [batch_size]
        """
        # 确保输入为tensor
        if not isinstance(task_rewards, torch.Tensor):
            task_rewards = torch.tensor(task_rewards, dtype=torch.float32, device=self.device)
        if not isinstance(mask_action_rewards, torch.Tensor):
            mask_action_rewards = torch.tensor(mask_action_rewards, dtype=torch.float32, device=self.device)
        
        # 移动到正确设备
        task_rewards = task_rewards.to(self.device)
        mask_action_rewards = mask_action_rewards.to(self.device)
        
        # 组合奖励: R'_t = R_t + α*a_t^m
        combined_rewards = task_rewards + self.mask_weight * mask_action_rewards
        
        return combined_rewards
    
    def compute_advantage_rewards(
        self,
        task_rewards: torch.Tensor,
        rnd_rewards: torch.Tensor,
        mask_action_rewards: Optional[torch.Tensor] = None,
        reward_type: str = 'exploration'
    ) -> torch.Tensor:
        """
        计算优势奖励，支持不同的奖励组合策略
        
        Args:
            task_rewards (torch.Tensor): 任务奖励
            rnd_rewards (torch.Tensor): RND奖励
            mask_action_rewards (Optional[torch.Tensor]): 掩码动作奖励
            reward_type (str): 奖励类型 ('exploration', 'mask', 'combined')
            
        Returns:
            torch.Tensor: 计算后的奖励
        """
        if reward_type == 'exploration':
            return self.combine_exploration_rewards(task_rewards, rnd_rewards)
        elif reward_type == 'mask' and mask_action_rewards is not None:
            return self.combine_mask_rewards(task_rewards, mask_action_rewards)
        elif reward_type == 'combined' and mask_action_rewards is not None:
            # 同时考虑探索和掩码奖励
            exploration_rewards = self.combine_exploration_rewards(
                task_rewards, rnd_rewards, update_stats=False
            )
            return self.combine_mask_rewards(exploration_rewards, mask_action_rewards)
        else:
            raise ValueError(f"Unsupported reward_type: {reward_type}")
    
    def set_exploration_weight(self, weight: float) -> None:
        """设置探索奖励权重 λ"""
        self.exploration_weight = weight
        self.logger.info(f"Updated exploration weight λ to {weight}")
    
    def set_mask_weight(self, weight: float) -> None:
        """设置掩码奖励权重 α"""
        self.mask_weight = weight
        self.logger.info(f"Updated mask weight α to {weight}")
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """获取奖励统计信息"""
        stats = {
            'task_reward_mean': self.task_reward_mean.item(),
            'task_reward_std': self.task_reward_std.item(),
            'rnd_reward_mean': self.rnd_reward_mean.item(),
            'rnd_reward_std': self.rnd_reward_std.item(),
            'exploration_weight': self.exploration_weight,
            'mask_weight': self.mask_weight,
            'update_count': self.update_count,
            'initialized': self.initialized,
            'task_reward_count': self.task_reward_count.item(),
            'rnd_reward_count': self.rnd_reward_count.item()
        }
        
        if self.normalization_method == 'robust':
            stats.update({
                'rnd_reward_median': self.rnd_reward_median.item(),
                'rnd_reward_mad': self.rnd_reward_mad.item()
            })
        
        return stats
    
    def reset_statistics(self) -> None:
        """重置奖励统计信息"""
        self.task_reward_mean.zero_()
        self.task_reward_std.fill_(1.0)
        self.task_reward_var.fill_(1.0)
        self.task_reward_count.zero_()
        
        self.rnd_reward_mean.zero_()
        self.rnd_reward_std.fill_(1.0)
        self.rnd_reward_var.fill_(1.0)
        self.rnd_reward_count.zero_()
        
        self.rnd_reward_median.zero_()
        self.rnd_reward_mad.fill_(1.0)
        
        self.initialized = False
        self.update_count = 0
        self.task_reward_history.clear()
        self.rnd_reward_history.clear()
        
        self.logger.info("Reset reward statistics")
    
    def get_normalization_info(self) -> Dict[str, Any]:
        """获取归一化相关信息，用于调试和监控"""
        return {
            'normalization_method': self.normalization_method,
            'rnd_reward_clip': self.rnd_reward_clip,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'min_std': self.min_std,
            'warmup_steps': self.warmup_steps,
            'reward_history_size': self.reward_history_size,
            'current_history_sizes': {
                'task': len(self.task_reward_history),
                'rnd': len(self.rnd_reward_history)
            }
        }
    
    def forward(
        self,
        task_rewards: torch.Tensor,
        rnd_rewards: torch.Tensor,
        mask_action_rewards: Optional[torch.Tensor] = None,
        reward_type: str = 'exploration'
    ) -> torch.Tensor:
        """
        前向传播，计算组合奖励
        
        Args:
            task_rewards (torch.Tensor): 任务奖励
            rnd_rewards (torch.Tensor): RND奖励
            mask_action_rewards (Optional[torch.Tensor]): 掩码动作奖励
            reward_type (str): 奖励组合类型
            
        Returns:
            torch.Tensor: 组合后的奖励
        """
        return self.compute_advantage_rewards(
            task_rewards, rnd_rewards, mask_action_rewards, reward_type
        )

class AdaptiveRewardCombiner(RewardCombiner):
    """
    自适应奖励组合器，动态调整权重参数
    
    扩展基础RewardCombiner，支持基于性能的权重自适应调整
    基于论文中提到的动态权重调整策略
    """
    
    def __init__(
        self,
        exploration_weight: float = 1.0,
        mask_weight: float = 0.1,
        adaptation_rate: float = 0.01,
        min_exploration_weight: float = 0.1,
        max_exploration_weight: float = 2.0,
        adaptation_window: int = 100,  # 性能评估窗口
        **kwargs
    ):
        """
        初始化自适应奖励组合器
        
        Args:
            adaptation_rate (float): 权重适应速率
            min_exploration_weight (float): 最小探索权重
            max_exploration_weight (float): 最大探索权重
            adaptation_window (int): 性能评估窗口大小
        """
        super().__init__(exploration_weight, mask_weight, **kwargs)
        
        self.adaptation_rate = adaptation_rate
        self.min_exploration_weight = min_exploration_weight
        self.max_exploration_weight = max_exploration_weight
        self.adaptation_window = adaptation_window
        
        # 性能追踪
        self.register_buffer('episode_returns', torch.zeros(adaptation_window))
        self.register_buffer('exploration_effectiveness', torch.zeros(1))
        self.register_buffer('task_progress', torch.zeros(1))
        self.episode_count = 0
        
        # 自适应参数
        self.performance_threshold = 0.1  # 性能改善阈值
        self.stagnation_threshold = 50   # 停滞检测阈值
    
    def update_exploration_weight(self, episode_return: float, exploration_bonus: float = 0.0) -> None:
        """
        基于回合回报和探索效果动态调整探索权重
        
        Args:
            episode_return (float): 当前回合回报
            exploration_bonus (float): 探索奖励贡献
        """
        # 更新回合回报历史
        idx = self.episode_count % self.adaptation_window
        self.episode_returns[idx] = episode_return
        self.episode_count += 1
        
        if self.episode_count >= self.adaptation_window:  # 有足够数据后开始调整
            # 计算性能指标
            recent_returns = self.episode_returns
            mean_return = recent_returns.mean()
            return_std = recent_returns.std()
            
            # 计算回报趋势
            half_window = self.adaptation_window // 2
            recent_half = recent_returns[half_window:]
            earlier_half = recent_returns[:half_window]
            return_trend = recent_half.mean() - earlier_half.mean()
            
            # 计算探索效果
            exploration_ratio = exploration_bonus / (abs(episode_return) + self.epsilon)
            self.exploration_effectiveness.mul_(0.9).add_(exploration_ratio, alpha=0.1)
            
            # 自适应调整策略
            if return_trend < -self.performance_threshold:
                # 性能下降，增加探索
                adjustment = self.adaptation_rate * (1 + abs(return_trend))
                self.exploration_weight = min(
                    self.max_exploration_weight,
                    self.exploration_weight + adjustment
                )
                self.logger.debug(f"Performance declining, increased exploration weight to {self.exploration_weight:.4f}")
                
            elif return_trend > self.performance_threshold:
                # 性能提升，适度减少探索
                adjustment = self.adaptation_rate * 0.5
                self.exploration_weight = max(
                    self.min_exploration_weight,
                    self.exploration_weight - adjustment
                )
                self.logger.debug(f"Performance improving, decreased exploration weight to {self.exploration_weight:.4f}")
                
            elif return_std < self.performance_threshold * 0.1:
                # 性能停滞，增加探索
                adjustment = self.adaptation_rate * 2.0
                self.exploration_weight = min(
                    self.max_exploration_weight,
                    self.exploration_weight + adjustment
                )
                self.logger.debug(f"Performance stagnant, increased exploration weight to {self.exploration_weight:.4f}")
    
    def get_adaptation_statistics(self) -> Dict[str, float]:
        """获取自适应调整的统计信息"""
        base_stats = self.get_reward_statistics()
        adaptation_stats = {
            'adaptation_rate': self.adaptation_rate,
            'min_exploration_weight': self.min_exploration_weight,
            'max_exploration_weight': self.max_exploration_weight,
            'exploration_effectiveness': self.exploration_effectiveness.item(),
            'episode_count': self.episode_count,
            'mean_episode_return': self.episode_returns.mean().item() if self.episode_count > 0 else 0.0,
            'episode_return_std': self.episode_returns.std().item() if self.episode_count > 1 else 0.0
        }
        
        return {**base_stats, **adaptation_stats}