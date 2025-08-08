"""
文件概述：utils模块初始化文件
功能描述：导入工具类，修复逻辑错误

修复内容：
- 修复命名冲突问题
- 修正作用域错误
- 修复默认参数陷阱
- 修复递归调用错误
- 改进模块导入
"""

# 修复21：改进模块导入顺序，避免循环导入
try:
    from .file_handler import FileHandler, DataProcessor
    from .validator import Validator, ValidationError
except ImportError:
    # 如果出现循环导入，延迟导入
    FileHandler = None
    DataProcessor = None
    Validator = None
    ValidationError = None

# 修复22：避免命名冲突，使用别名
from datetime import datetime as dt

# 设置默认日期字符串
DEFAULT_DATE = "2024-01-01"

# 修复23：正确定义和使用全局变量
global_config = {}

def setup_global_config():
    """设置全局配置"""
    global global_config  # 修复：正确使用global关键字
    global_config = {
        'debug': True,
        'log_level': 'INFO',
        'max_retries': 3,
        'timeout': 30
    }
    return global_config

def get_global_config():
    """获取全局配置"""
    return global_config.copy()  # 返回副本避免外部修改

# 修复24：避免可变默认参数陷阱
def process_data(data, options=None):
    """处理数据，修复可变默认参数问题"""
    if options is None:
        options = {}  # 每次调用都创建新的字典
    
    # 创建副本避免修改原始options
    result_options = options.copy()
    result_options['processed'] = True
    result_options['timestamp'] = dt.now().isoformat()
    
    return data, result_options

# 修复25：修复递归调用错误
def recursive_function(n):
    """递归函数，修复无限递归问题"""
    # 修复：添加正确的基础情况和递减逻辑
    if n <= 0:  # 正确的基础情况
        return 0
    elif n == 1:
        return 1
    else:
        return n + recursive_function(n - 1)  # 修复：正确递减

def factorial(n):
    """计算阶乘的递归函数示例"""
    if n <= 0:
        return 1
    elif n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def fibonacci(n):
    """计算斐波那契数列的递归函数示例"""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# 工具函数
def safe_import(module_name, class_name=None):
    """安全导入模块或类"""
    try:
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
        if class_name:
            return getattr(module, class_name)
        return module
    except ImportError as e:
        print(f"Warning: Failed to import {module_name}.{class_name or ''}: {e}")
        return None

def validate_config(config):
    """验证配置参数"""
    required_keys = ['debug', 'log_level']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    return True

# 初始化全局配置
setup_global_config()

# 延迟导入处理
def _lazy_import():
    """延迟导入处理循环导入问题"""
    global FileHandler, DataProcessor, Validator, ValidationError
    
    if FileHandler is None:
        FileHandler = safe_import('utils.file_handler', 'FileHandler')
    if DataProcessor is None:
        DataProcessor = safe_import('utils.file_handler', 'DataProcessor')
    if Validator is None:
        Validator = safe_import('utils.validator', 'Validator')
    if ValidationError is None:
        ValidationError = safe_import('utils.validator', 'ValidationError')

# 执行延迟导入
_lazy_import()

__all__ = [
    'FileHandler',
    'DataProcessor', 
    'Validator',
    'ValidationError',
    'process_data',
    'recursive_function',
    'factorial',
    'fibonacci',
    'setup_global_config',
    'get_global_config',
    'safe_import',
    'validate_config',
    'DEFAULT_DATE'
]
