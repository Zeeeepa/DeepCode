"""
文件概述：utils模块初始化文件
功能描述：导入工具类，包含更多逻辑错误

错误类型：逻辑错误 (21-25)
21. 模块导入顺序错误
22. 命名冲突
23. 作用域错误
24. 默认参数陷阱
25. 递归调用错误
"""

# 错误21：模块导入顺序错误 - 循环导入
from .file_handler import FileHandler, DataProcessor
from .validator import Validator, ValidationError

# 错误22：命名冲突
from datetime import datetime
datetime = "2024-01-01"  # 错误：覆盖了datetime模块

# 错误23：作用域错误 - 全局变量定义
global_config = {}

def setup_global_config():
    # 错误23：在函数内修改全局变量但没有使用global关键字
    global_config = {  # 应该是 global global_config
        'debug': True,
        'log_level': 'INFO'
    }

# 错误24：默认参数陷阱
def process_data(data, options={}):  # 错误：可变默认参数
    options['processed'] = True
    return data, options

# 错误25：递归调用错误
def recursive_function(n):
    if n > 0:  # 错误：没有正确的基础情况
        return recursive_function(n)  # 错误：没有递减，会无限递归
    return 0

__all__ = [
    'FileHandler',
    'DataProcessor', 
    'Validator',
    'ValidationError',
    'process_data',
    'recursive_function'
]
