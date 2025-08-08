"""
Agent配置管理模块

提供统一的配置管理功能，支持多种配置源。

主要功能:
- AgentConfig: 配置管理器
- 环境变量配置
- 配置文件加载
- 配置验证
"""

from .agent_config import AgentConfig

__all__ = ['AgentConfig'] 