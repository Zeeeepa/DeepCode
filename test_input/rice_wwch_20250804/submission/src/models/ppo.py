```python
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional
from collections import deque

class RolloutBuffer:
    """
    PPO算法的经验回放缓冲区实现
    
    用于存储和管理PPO训练过程中的轨迹数据，包括状态、动作、奖励、价值估计等。
    实现了GAE（Generalized Advantage Estimation）优势估计算法。
    
    论文参考: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
    GAE参考: High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)
    """
    
    def __init__(self, 
                 buffer_size: int, 
                 obs_shape: tuple, 
                 action_shape: tuple,
                 device: str = 'cpu',
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        """
        初始化回放缓冲区
        
        Args:
            buffer_size: 缓冲区大小（时间步数）
            obs_shape: 观测空间形状
            action_shape: 动作空间形状
            device: 计算设备
            gamma: 折扣因子
            gae_lambda: GAE的λ参数
        """
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 当前存储位置和数据计数
        self.ptr = 0
        self.size = 0
        self.full = False
        
        # 初始化存储缓冲区
        self._init_buffers()
        
    def _init_buffers(self):
        """初始化所有数据缓冲区"""
        # 观测数据 (buffer_size, *obs_shape)
        self.observations = torch.zeros(
            (self.buffer_size,) + self.obs_shape, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # 动作数据 (buffer_size, *action_shape)
        self.actions = torch.zeros(
            (self.buffer_size,) + self.action_shape, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # 奖励数据 (buffer_size,)
        self.rewards = torch.zeros(
            self.buffer_size, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # 价值估计 (buffer_size,)
        self.values = torch.zeros(
            self.buffer_size, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # 动作对数概率 (buffer_size,)
        self.log_probs = torch.zeros(
            self.buffer_size, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # 终止标志 (buffer_size,)
        self.dones = torch.zeros(
            self.buffer_size, 
            dtype=torch.bool, 
            device=self.device
        )
        
        # GAE计算结果
        self.advantages = torch.zeros(
            self.buffer_size, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # 目标价值（用于价值函数训练）
        self.returns = torch.zeros(
            self.buffer_size, 
            dtype=torch.float32, 
            device=self.device
        )
        
    def store(self, 
              obs: torch.Tensor, 
              action: torch.Tensor, 
              reward: float, 
              value: float, 
              log_prob: float, 
              done: bool):
        """
        存储一个时间步的数据
        
        Args:
            obs: 当前观测
            action: 执行的动作
            reward: 获得的奖励
            value: 价值函数估计
            log_prob: 动作的对数概率
            done: 是否为终止状态
        """
        assert self.ptr < self.buffer_size, "缓冲区已满，请先调用reset()清空缓冲区"
        
        # 确保数据在正确的设备上
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
            
        obs = obs.to(self.device)
        action = action.to(self.device)
        
        # 存储数据
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        # 更新指针
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
        
        if self.ptr >= self.buffer_size:
            self.full = True
            
    def finish_path(self, last_value: float = 0.0):
        """
        完成一条轨迹的数据收集，计算GAE优势和回报
        
        Args:
            last_value: 最后一个状态的价值估计（用于bootstrap）
        """
        if self.size == 0:
            return
            
        # 计算GAE优势估计
        # 论文公式: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        # 其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
        
        # 获取有效数据范围
        path_slice = slice(0, self.size)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        
        # 计算时序差分误差 δ_t
        # δ_t = r_t + γV(s_{t+1})(1-done_t) - V(s_t)
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        next_values[-1] = last_value
        
        deltas = rewards + self.gamma * next_values * (~dones) - values
        
        # 使用GAE计算优势
        advantages = torch.zeros_like(deltas)
        gae = 0
        
        # 从后往前计算GAE
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.gae_lambda * gae * (~dones[t])
            advantages[t] = gae
            
        # 计算回报 (优势 + 价值估计)
        returns = advantages + values
        
        # 存储计算结果
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        
    def get_batch(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        获取训练批次数据
        
        Args:
            batch_size: 批次大小，如果为None则返回所有数据
            
        Returns:
            包含训练数据的字典
        """
        assert self.size > 0, "缓冲区为空，无法获取批次数据"
        
        # 获取有效数据
        valid_slice = slice(0, self.size)
        
        if batch_size is None or batch_size >= self.size:
            # 返回所有数据
            indices = torch.arange(self.size, device=self.device)
        else:
            # 随机采样
            indices = torch.randperm(self.size, device=self.device)[:batch_size]
            
        batch = {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'dones': self.dones[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices]
        }
        
        return batch
        
    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        获取所有存储的数据
        
        Returns:
            包含所有数据的字典
        """
        return self.get_batch(batch_size=None)
        
    def normalize_advantages(self, eps: float = 1e-8):
        """
        标准化优势估计
        
        Args:
            eps: 数值稳定性参数
        """
        if self.size == 0:
            return
            
        valid_slice = slice(0, self.size)
        advantages = self.advantages[valid_slice]
        
        # 计算均值和标准差
        mean = advantages.mean()
        std = advantages.std()
        
        # 标准化
        self.advantages[valid_slice] = (advantages - mean) / (std + eps)
        
    def reset(self):
        """重置缓冲区"""
        self.ptr = 0
        self.size = 0
        self.full = False
        
        # 清零所有缓冲区
        self.observations.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.log_probs.zero_()
        self.dones.zero_()
        self.advantages.zero_()
        self.returns.zero_()
        
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.full
        
    def __len__(self) -> int:
        """返回当前存储的数据量"""
        return self.size
        
    def get_statistics(self) -> Dict[str, float]:
        """
        获取缓冲区数据的统计信息
        
        Returns:
            统计信息字典
        """
        if self.size == 0:
            return {}
            
        valid_slice = slice(0, self.size)
        
        stats = {
            'buffer_size': self.size,
            'mean_reward': self.rewards[valid_slice].mean().item(),
            'std_reward': self.rewards[valid_slice].std().item(),
            'mean_value': self.values[valid_slice].mean().item(),
            'std_value': self.values[valid_slice].std().item(),
            'mean_advantage': self.advantages[valid_slice].mean().item(),
            'std_advantage': self.advantages[valid_slice].std().item(),
            'mean_return': self.returns[valid_slice].mean().item(),
            'std_return': self.returns[valid_slice].std().item(),
        }
        
        return stats
        
    def save_data(self, filepath: str):
        """
        保存缓冲区数据到文件
        
        Args:
            filepath: 保存路径
        """
        data = {
            'observations': self.observations[:self.size].cpu().numpy(),
            'actions': self.actions[:self.size].cpu().numpy(),
            'rewards': self.rewards[:self.size].cpu().numpy(),
            'values': self.values[:self.size].cpu().numpy(),
            'log_probs': self.log_probs[:self.size].cpu().numpy(),
            'dones': self.dones[:self.size].cpu().numpy(),
            'advantages': self.advantages[:self.size].cpu().numpy(),
            'returns': self.returns[:self.size].cpu().numpy(),
            'buffer_config': {
                'buffer_size': self.buffer_size,
                'obs_shape': self.obs_shape,
                'action_shape': self.action_shape,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda
            }
        }
        
        np.savez_compressed(filepath, **data)
        
    def load_data(self, filepath: str):
        """
        从文件加载缓冲区数据
        
        Args:
            filepath: 文件路径
        """
        data = np.load(filepath, allow_pickle=True)
        
        # 验证配置兼容性
        config = data['buffer_config'].item()
        assert config['obs_shape'] == self.obs_shape, "观测空间形状不匹配"
        assert config['action_shape'] == self.action_shape, "动作空间形状不匹配"
        
        # 加载数据
        self.size = len(data['rewards'])
        self.ptr = self.size
        
        self.observations[:self.size] = torch.from_numpy(data['observations']).to(self.device)
        self.actions[:self.size] = torch.from_numpy(data['actions']).to(self.device)
        self.rewards[:self.size] = torch.from_numpy(data['rewards']).to(self.device)
        self.values[:self.size] = torch.from_numpy(data['values']).to(self.device)
        self.log_probs[:self.size] = torch.from_numpy(data['log_probs']).to(self.device)
        self.dones[:self.size] = torch.from_numpy(data['dones']).to(self.device)
        self.advantages[:self.size] = torch.from_numpy(data['advantages']).to(self.device)
        self.returns[:self.size] = torch.from_numpy(data['returns']).to(self.device)


class MultiEnvRolloutBuffer:
    """
    多环境并行的回放缓冲区
    
    用于处理多个环境并行收集数据的情况，每个环境维护独立的轨迹。
    """
    
    def __init__(self, 
                 num_envs: int,
                 buffer_size_per_env: int,
                 obs_shape: tuple,
                 action_shape: tuple,
                 device: str = 'cpu',
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        """
        初始化多环境缓冲区
        
        Args:
            num_envs: 环境数量
            buffer_size_per_env: 每个环境的缓冲区大小
            obs_shape: 观测空间形状
            action_shape: 动作空间形状
            device: 计算设备
            gamma: 折扣因子
            gae_lambda: GAE的λ参数
        """
        self.num_envs = num_envs
        self.buffer_size_per_env = buffer_size_per_env
        
        # 为每个环境创建独立的缓冲区
        self.buffers = [
            RolloutBuffer(
                buffer_size=buffer_size_per_env,
                obs_shape=obs_shape,
                action_shape=action_shape,
                device=device,
                gamma=gamma,
                gae_lambda=gae_lambda
            ) for _ in range(num_envs)
        ]
        
    def store(self, 
              env_id: int,
              obs: torch.Tensor,
              action: torch.Tensor,
              reward: float,
              value: float,
              log_prob: float,
              done: bool):
        """存储指定环境的数据"""
        self.buffers[env_id].store(obs, action, reward, value, log_prob, done)
        
    def finish_path(self, env_id: int, last_value: float = 0.0):
        """完成指定环境的轨迹"""
        self.buffers[env_id].finish_path(last_value)
        
    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        获取所有环境的数据并合并
        
        Returns:
            合并后的数据字典
        """
        all_data = []
        
        for buffer in self.buffers:
            if len(buffer) > 0:
                data = buffer.get_all_data()
                all_data.append(data)
                
        if not all_data:
            return {}
            
        # 合并所有数据
        merged_data = {}
        for key in all_data[0].keys():
            merged_data[key] = torch.cat([data[key] for data in all_data], dim=0)
            
        return merged_data
        
    def normalize_advantages(self):
        """标准化所有环境的优势估计"""
        # 收集所有优势数据
        all_advantages = []
        for buffer in self.buffers:
            if len(buffer) > 0:
                all_advantages.append(buffer.advantages[:buffer.size])
                
        if not all_advantages:
            return
            
        # 计算全局统计量
        all_advantages = torch.cat(all_advantages, dim=0)
        mean = all_advantages.mean()
        std = all_advantages.std()
        
        # 应用到每个缓冲区
        for buffer in self.buffers:
            if len(buffer) > 0:
                buffer.advantages[:buffer.size] = (
                    buffer.advantages[:buffer.size] - mean
                ) / (std + 1e-8)
                
    def reset(self):
        """重置所有缓冲区"""
        for buffer in self.buffers:
            buffer.reset()
            
    def __len__(self) -> int:
        """返回所有缓冲区的总数据量"""
        return sum(len(buffer) for buffer in self.buffers)
        
    def get_statistics(self) -> Dict[str, float]:
        """获取所有环境的统计信息"""
        all_stats = []
        for i, buffer in enumerate(self.buffers):
            if len(buffer) > 0:
                stats = buffer.get_statistics()
                stats['env_id'] = i
                all_stats.append(stats)
                
        if not all_stats:
            return {}
            
        # 计算平均统计量
        avg_stats = {}
        for key in ['mean_reward', 'mean_value', 'mean_advantage', 'mean_return']:
            if key in all_stats[0]:
                avg_stats[key] = np.mean([stats[key] for stats in all_stats])
                
        avg_stats['total_samples'] = sum(stats['buffer_size'] for stats in all_stats)
        avg_stats['num_active_envs'] = len(all_stats)
        
        return avg_stats
```