"""
核心功能模块包

本包提供多种代码分析工具来分析Python项目的结构和依赖关系。

主要功能:
- 使用pydeps分析模块依赖关系
- 使用code2flow生成函数调用流程图
- 使用AST直接提取项目结构和代码定义
- 使用tree-sitter解析多种编程语言的语法树
- 提取仓库结构信息
- 生成可视化依赖图和流程图
- 输出JSON格式的详细分析结果
"""

__version__ = "1.0.0"
__author__ = "Debug Agent"

# 导入核心模块
from .pydeps_analyzer import PyDepsAnalyzer
from .code2flow_analyzer import Code2FlowAnalyzer
from .project_structure_analyzer import ProjectStructureAnalyzer
from .simple_structure_analyzer import SimpleStructureAnalyzer

# 尝试导入tree-sitter分析器（可选）
try:
    from .treesitter_analyzer import TreeSitterAnalyzer
    TREESITTER_AVAILABLE = True
    __all__ = ['PyDepsAnalyzer', 'Code2FlowAnalyzer', 'ProjectStructureAnalyzer', 
               'SimpleStructureAnalyzer', 'TreeSitterAnalyzer']
except ImportError:
    TREESITTER_AVAILABLE = False
    __all__ = ['PyDepsAnalyzer', 'Code2FlowAnalyzer', 'ProjectStructureAnalyzer', 
               'SimpleStructureAnalyzer'] 