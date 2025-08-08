import os
import json
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class NetworkConfig:
    """网络架构配置类
    
    定义神经网络的基本架构参数，包括隐藏层维度、激活函数等
    """
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = 'relu'
    dropout_rate: float = 0.0
    batch_norm: bool = False
    layer_norm: bool = True

@dataclass
class TrainingConfig:
    """训练参数配置类
    
    包含训练过程中的所有超参数，如学习率、批次大小、训练步数等
    """
    learning_rate: float = 3e-4
    batch_size: int = 256
    max_timesteps: int = 1000000
    eval_freq: int = 5000
    save_freq: int = 10000
    warmup_steps: int = 25000
    gradient_clip: float = 1.0
    weight_decay: float = 0.0
    optimizer: str = 'adam'
    lr_scheduler: str = 'constant'

@dataclass
class EnvironmentConfig:
    """环境特定配置类
    
    根据论文Table 3设置每个环境的特定超参数
    包括p值（策略更新频率）、λ值（TD-lambda参数）、α值（学习率调节）
    """
    env_name: str
    p: float  # 策略更新频率参数
    lambda_param: float  # TD-lambda参数
    alpha: float  # 学习率调节参数
    max_episode_steps: int = 1000
    action_noise: float = 0.1
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    tau: float = 0.005  # 软更新参数
    discount: float = 0.99  # 折扣因子

@dataclass
class PathConfig:
    """路径配置类
    
    管理所有文件路径，包括模型保存、日志记录、数据存储等
    """
    root_dir: str = "./results"
    model_dir: str = "./models"
    log_dir: str = "./logs"
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    tensorboard_dir: str = "./tensorboard"
    
    def __post_init__(self):
        """初始化后创建必要的目录"""
        for path in [self.root_dir, self.model_dir, self.log_dir, 
                    self.data_dir, self.checkpoint_dir, self.tensorboard_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)

@dataclass
class LogConfig:
    """日志配置类
    
    配置日志记录的详细参数，包括日志级别、格式、输出方式等
    """
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_logging: bool = True
    console_logging: bool = True
    log_file: str = 'training.log'
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

class ConfigManager:
    """配置管理器
    
    集中管理所有配置参数，提供配置的加载、保存、验证等功能
    根据论文Table 3为不同环境预设正确的超参数组合
    """
    
    # 论文Table 3中的环境特定超参数配置
    ENVIRONMENT_CONFIGS = {
        'Hopper-v2': {
            'p': 0.25,
            'lambda_param': 0.95,
            'alpha': 0.2,
            'max_episode_steps': 1000,
            'action_noise': 0.1,
            'exploration_noise': 0.1
        },
        'Walker2d-v2': {
            'p': 0.25,
            'lambda_param': 0.95,
            'alpha': 0.2,
            'max_episode_steps': 1000,
            'action_noise': 0.1,
            'exploration_noise': 0.1
        },
        'HalfCheetah-v2': {
            'p': 0.25,
            'lambda_param': 0.95,
            'alpha': 0.2,
            'max_episode_steps': 1000,
            'action_noise': 0.1,
            'exploration_noise': 0.1
        },
        'Ant-v2': {
            'p': 0.25,
            'lambda_param': 0.95,
            'alpha': 0.2,
            'max_episode_steps': 1000,
            'action_noise': 0.1,
            'exploration_noise': 0.1
        },
        'Reacher-v2': {
            'p': 0.5,
            'lambda_param': 0.9,
            'alpha': 0.3,
            'max_episode_steps': 50,
            'action_noise': 0.1,
            'exploration_noise': 0.1
        },
        'Pusher-v2': {
            'p': 0.5,
            'lambda_param': 0.9,
            'alpha': 0.3,
            'max_episode_steps': 100,
            'action_noise': 0.1,
            'exploration_noise': 0.1
        },
        'Swimmer-v2': {
            'p': 0.25,
            'lambda_param': 0.95,
            'alpha': 0.2,
            'max_episode_steps': 1000,
            'action_noise': 0.1,
            'exploration_noise': 0.1
        },
        'InvertedPendulum-v2': {
            'p': 0.5,
            'lambda_param': 0.9,
            'alpha': 0.3,
            'max_episode_steps': 1000,
            'action_noise': 0.1,
            'exploration_noise': 0.1
        },
        'InvertedDoublePendulum-v2': {
            'p': 0.5,
            'lambda_param': 0.9,
            'alpha': 0.3,
            'max_episode_steps': 1000,
            'action_noise': 0.1,
            'exploration_noise': 0.1
        },
        'Humanoid-v2': {
            'p': 0.25,
            'lambda_param': 0.95,
            'alpha': 0.2,
            'max_episode_steps': 1000,
            'action_noise': 0.1,
            'exploration_noise': 0.1
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_file: 可选的配置文件路径
        """
        self.config_file = config_file
        self.network_config = NetworkConfig()
        self.training_config = TrainingConfig()
        self.path_config = PathConfig()
        self.log_config = LogConfig()
        self.environment_configs = {}
        
        # 初始化所有环境配置
        self._initialize_environment_configs()
        
        # 如果提供了配置文件，则加载配置
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def _initialize_environment_configs(self):
        """初始化所有环境的配置"""
        for env_name, params in self.ENVIRONMENT_CONFIGS.items():
            self.environment_configs[env_name] = EnvironmentConfig(
                env_name=env_name,
                **params
            )
    
    def get_environment_config(self, env_name: str) -> EnvironmentConfig:
        """获取指定环境的配置
        
        Args:
            env_name: 环境名称
            
        Returns:
            环境配置对象
            
        Raises:
            ValueError: 如果环境名称不存在
        """
        if env_name not in self.environment_configs:
            raise ValueError(f"Environment {env_name} not found in predefined configurations. "
                           f"Available environments: {list(self.environment_configs.keys())}")
        
        return self.environment_configs[env_name]
    
    def update_environment_config(self, env_name: str, **kwargs):
        """更新指定环境的配置参数
        
        Args:
            env_name: 环境名称
            **kwargs: 要更新的参数
        """
        if env_name not in self.environment_configs:
            # 如果环境不存在，创建新的配置
            base_params = {
                'p': 0.25,
                'lambda_param': 0.95,
                'alpha': 0.2,
                'max_episode_steps': 1000,
                'action_noise': 0.1,
                'exploration_noise': 0.1
            }
            base_params.update(kwargs)
            self.environment_configs[env_name] = EnvironmentConfig(
                env_name=env_name,
                **base_params
            )
        else:
            # 更新现有配置
            config = self.environment_configs[env_name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logging.warning(f"Unknown parameter {key} for environment {env_name}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置的字典表示
        
        Returns:
            包含所有配置的字典
        """
        return {
            'network': asdict(self.network_config),
            'training': asdict(self.training_config),
            'path': asdict(self.path_config),
            'log': asdict(self.log_config),
            'environments': {name: asdict(config) 
                           for name, config in self.environment_configs.items()}
        }
    
    def save_config(self, filepath: str):
        """保存配置到文件
        
        Args:
            filepath: 配置文件保存路径
        """
        config_dict = self.get_all_configs()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath: str):
        """从文件加载配置
        
        Args:
            filepath: 配置文件路径
            
        Raises:
            FileNotFoundError: 如果配置文件不存在
            json.JSONDecodeError: 如果配置文件格式错误
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file {filepath} not found")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 更新各个配置
        if 'network' in config_dict:
            self.network_config = NetworkConfig(**config_dict['network'])
        
        if 'training' in config_dict:
            self.training_config = TrainingConfig(**config_dict['training'])
        
        if 'path' in config_dict:
            self.path_config = PathConfig(**config_dict['path'])
        
        if 'log' in config_dict:
            self.log_config = LogConfig(**config_dict['log'])
        
        if 'environments' in config_dict:
            self.environment_configs = {}
            for env_name, env_params in config_dict['environments'].items():
                self.environment_configs[env_name] = EnvironmentConfig(**env_params)
        
        logging.info(f"Configuration loaded from {filepath}")
    
    def validate_config(self, env_name: str) -> bool:
        """验证指定环境的配置是否有效
        
        Args:
            env_name: 环境名称
            
        Returns:
            配置是否有效
        """
        try:
            env_config = self.get_environment_config(env_name)
            
            # 验证关键参数范围
            if not (0 < env_config.p <= 1):
                logging.error(f"Invalid p value for {env_name}: {env_config.p}")
                return False
            
            if not (0 < env_config.lambda_param <= 1):
                logging.error(f"Invalid lambda value for {env_name}: {env_config.lambda_param}")
                return False
            
            if not (0 < env_config.alpha <= 1):
                logging.error(f"Invalid alpha value for {env_name}: {env_config.alpha}")
                return False
            
            if env_config.max_episode_steps <= 0:
                logging.error(f"Invalid max_episode_steps for {env_name}: {env_config.max_episode_steps}")
                return False
            
            # 验证训练配置
            if self.training_config.learning_rate <= 0:
                logging.error(f"Invalid learning rate: {self.training_config.learning_rate}")
                return False
            
            if self.training_config.batch_size <= 0:
                logging.error(f"Invalid batch size: {self.training_config.batch_size}")
                return False
            
            # 验证网络配置
            if self.network_config.hidden_dim <= 0:
                logging.error(f"Invalid hidden dimension: {self.network_config.hidden_dim}")
                return False
            
            if self.network_config.num_layers <= 0:
                logging.error(f"Invalid number of layers: {self.network_config.num_layers}")
                return False
            
            logging.info(f"Configuration validation passed for {env_name}")
            return True
            
        except Exception as e:
            logging.error(f"Configuration validation failed for {env_name}: {str(e)}")
            return False
    
    def setup_logging(self):
        """设置日志配置"""
        # 创建日志目录
        log_dir = Path(self.path_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志格式
        formatter = logging.Formatter(self.log_config.format)
        
        # 获取根日志记录器
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.log_config.level.upper()))
        
        # 清除现有的处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 控制台日志处理器
        if self.log_config.console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 文件日志处理器
        if self.log_config.file_logging:
            from logging.handlers import RotatingFileHandler
            log_file_path = log_dir / self.log_config.log_file
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=self.log_config.max_log_size,
                backupCount=self.log_config.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        logging.info("Logging configuration completed")
    
    def print_config_summary(self, env_name: str):
        """打印配置摘要
        
        Args:
            env_name: 环境名称
        """
        print(f"\n{'='*60}")
        print(f"Configuration Summary for {env_name}")
        print(f"{'='*60}")
        
        env_config = self.get_environment_config(env_name)
        
        print(f"\nEnvironment Parameters:")
        print(f"  p (policy update frequency): {env_config.p}")
        print(f"  λ (TD-lambda parameter): {env_config.lambda_param}")
        print(f"  α (learning rate adjustment): {env_config.alpha}")
        print(f"  Max episode steps: {env_config.max_episode_steps}")
        print(f"  Action noise: {env_config.action_noise}")
        print(f"  Exploration noise: {env_config.exploration_noise}")
        
        print(f"\nNetwork Architecture:")
        print(f"  Hidden dimension: {self.network_config.hidden_dim}")
        print(f"  Number of layers: {self.network_config.num_layers}")
        print(f"  Activation: {self.network_config.activation}")
        print(f"  Layer normalization: {self.network_config.layer_norm}")
        
        print(f"\nTraining Parameters:")
        print(f"  Learning rate: {self.training_config.learning_rate}")
        print(f"  Batch size: {self.training_config.batch_size}")
        print(f"  Max timesteps: {self.training_config.max_timesteps}")
        print(f"  Evaluation frequency: {self.training_config.eval_freq}")
        print(f"  Warmup steps: {self.training_config.warmup_steps}")
        
        print(f"\nPaths:")
        print(f"  Root directory: {self.path_config.root_dir}")
        print(f"  Model directory: {self.path_config.model_dir}")
        print(f"  Log directory: {self.path_config.log_dir}")
        
        print(f"{'='*60}\n")

def create_default_config(env_name: str, config_file: str = None) -> ConfigManager:
    """创建默认配置管理器
    
    Args:
        env_name: 环境名称
        config_file: 可选的配置文件路径
        
    Returns:
        配置管理器实例
    """
    config_manager = ConfigManager(config_file)
    
    # 验证环境配置
    if not config_manager.validate_config(env_name):
        raise ValueError(f"Invalid configuration for environment {env_name}")
    
    # 设置日志
    config_manager.setup_logging()
    
    # 打印配置摘要
    config_manager.print_config_summary(env_name)
    
    return config_manager

def get_environment_list() -> list:
    """获取所有支持的环境列表
    
    Returns:
        支持的环境名称列表
    """
    return list(ConfigManager.ENVIRONMENT_CONFIGS.keys())

def check_environment_support(env_name: str) -> bool:
    """检查环境是否被支持
    
    Args:
        env_name: 环境名称
        
    Returns:
        是否支持该环境
    """
    return env_name in ConfigManager.ENVIRONMENT_CONFIGS

# 示例使用函数
def example_usage():
    """配置管理器使用示例"""
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 获取Hopper环境的配置
    hopper_config = config_manager.get_environment_config('Hopper-v2')
    print(f"Hopper p value: {hopper_config.p}")
    
    # 更新环境配置
    config_manager.update_environment_config('Hopper-v2', p=0.3, alpha=0.25)
    
    # 保存配置
    config_manager.save_config('./config/hopper_config.json')
    
    # 验证配置
    is_valid = config_manager.validate_config('Hopper-v2')
    print(f"Configuration valid: {is_valid}")
    
    # 打印配置摘要
    config_manager.print_config_summary('Hopper-v2')

if __name__ == "__main__":
    example_usage()