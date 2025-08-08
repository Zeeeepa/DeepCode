"""
Agent配置管理器

简化的配置管理器，只支持环境变量/.env文件配置。

主要功能:
- get_config(): 从环境变量获取配置值
"""

import os
from typing import Any


class AgentConfig:
    """
    Agent配置管理器
    
    只支持环境变量配置方式：
    1. 环境变量 (直接设置)
    2. .env文件 (项目根目录)
    
    所有配置通过AGENT_前缀的环境变量设置
    """
    
    def __init__(self):
        """
        初始化配置管理器
        """
        self._load_dotenv()
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        从环境变量获取配置值
        
        参数:
            key (str): 配置键名（不含AGENT_前缀）
            default: 默认值
        
        返回:
            Any: 配置值
        """
        env_key = f"AGENT_{key.upper()}"
        value = os.getenv(env_key, default)
        
        # 转换数据类型
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            elif value.isdigit():
                return int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                return float(value)
        
        return value

    def _load_dotenv(self) -> None:
        """加载.env文件中的环境变量"""
        try:
            env_file = '.env'
            if os.path.exists(env_file):
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
        except Exception:
            # 静默失败，使用现有环境变量
            pass