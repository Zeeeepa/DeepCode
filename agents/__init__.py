"""
智能Agent框架

本包提供了一个灵活的Agent框架，支持多种类型的智能Agent。

主要组件:
- BaseAgent: 所有Agent的基础类
- JudgerAgent: 判断Agent（判断程序输出是否正确）
- AnalyzerAgent: 分析Agent（分析错误并生成修复方案）
- CoderAgent: 修复Agent（执行代码修复）
- DebugSystem: 三个Agent协作的调试系统

使用方法:
    from agents import JudgerAgent, AnalyzerAgent, CoderAgent, DebugSystem
    
    # 单独使用
    agent = JudgerAgent()
    result = agent.judge_output(stdout)
    
    # 协作调试系统
    debug_system = DebugSystem()
    result = debug_system.debug_program(repo_path, main_file)
"""

__version__ = "1.0.0"
__author__ = "Debug Agent"

# 导入主要组件
from .base_agent import BaseAgent
from .judger_agent import JudgerAgent
from .analyzer_agent import AnalyzerAgent
from .coder_agent import CoderAgent
from .debug_system import DebugSystem

__all__ = ['BaseAgent', 'JudgerAgent', 'AnalyzerAgent', 'CoderAgent', 'DebugSystem'] 