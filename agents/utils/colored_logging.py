"""
彩色日志系统 - 修复版

提供彩色的日志输出功能，准确定位调用位置。

主要功能:
- get_colored_logger(): 获取彩色日志记录器  
- 准确的调用栈定位
- 彩色输出支持
"""

import logging
import sys
import os
import inspect
from datetime import datetime
from typing import Optional, Dict, Any


class ColorCodes:
    """ANSI颜色代码"""
    # 基础颜色
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # 亮色
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # 样式
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    
    # 重置
    RESET = '\033[0m'
    ENDC = '\033[0m'


class AccurateColoredFormatter(logging.Formatter):
    """精确定位的彩色日志格式化器"""
    
    def __init__(self, use_colors: bool = True):
        """
        初始化彩色格式化器
        
        参数:
            use_colors (bool): 是否使用颜色
        """
        self.use_colors = use_colors and self._supports_color()
        
        # 定义各级别的颜色和图标
        self.level_styles = {
            'DEBUG': {'color': ColorCodes.BRIGHT_BLACK, 'icon': '🐛'},
            'INFO': {'color': ColorCodes.BRIGHT_GREEN, 'icon': '✅'},
            'WARNING': {'color': ColorCodes.BRIGHT_YELLOW, 'icon': '⚠️'},
            'ERROR': {'color': ColorCodes.BRIGHT_RED, 'icon': '❌'},
            'CRITICAL': {'color': ColorCodes.BRIGHT_MAGENTA, 'icon': '💥'},
        }
        
        # 定义组件颜色
        self.component_colors = {
            'timestamp': ColorCodes.GREEN,
            'filename': ColorCodes.CYAN,
            'function': ColorCodes.BLUE,
            'line': ColorCodes.MAGENTA,
            'message': ColorCodes.WHITE,
            'category': ColorCodes.BRIGHT_CYAN,
        }
        
        super().__init__()
    
    def _supports_color(self) -> bool:
        """检查终端是否支持颜色"""
        return (
            hasattr(sys.stderr, "isatty") and sys.stderr.isatty() and
            os.environ.get('TERM') != 'dumb'
        )
    
    def _colorize(self, text: str, color: str) -> str:
        """给文本添加颜色"""
        if not self.use_colors:
            return text
        return f"{color}{text}{ColorCodes.RESET}"
    
    def _get_real_caller_info(self) -> tuple:
        """
        获取真正的调用者信息
        通过检查调用栈，跳过所有logger相关的栈帧
        """
        current_frame = inspect.currentframe()
        
        try:
            # 遍历调用栈，找到第一个不在logging系统内的调用者
            frame = current_frame
            while frame:
                frame = frame.f_back
                if not frame:
                    break
                    
                filename = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                
                # 跳过logging相关的栈帧
                if ('colored_logging.py' in filename or 
                    'logging' in filename or
                    function_name in ['_log', 'log', 'debug', 'info', 'warning', 'error', 'critical',
                                     'log_detailed', 'log_llm_call', 'log_operation_start', 
                                     'log_operation_success', 'log_operation_error', 'wrapper']):
                    continue
                
                # 找到真正的调用者
                return (
                    os.path.basename(filename),
                    function_name,
                    frame.f_lineno
                )
                
            # 如果没找到，使用默认值
            return ("unknown.py", "unknown_function", 0)
            
        finally:
            del current_frame
    
    def format(self, record):
        """格式化日志记录"""
        # 获取时间戳  
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        
        # 获取真正的调用者信息
        real_filename, real_function, real_lineno = self._get_real_caller_info()
        
        # 获取日志级别信息
        level_name = record.levelname
        level_style = self.level_styles.get(level_name, {'color': ColorCodes.WHITE, 'icon': '📝'})
        
        # 获取消息和分类信息
        message = record.getMessage()
        category = getattr(record, 'category', None)
        
        # 应用颜色
        if self.use_colors:
            colored_timestamp = self._colorize(timestamp, self.component_colors['timestamp'])
            colored_filename = self._colorize(real_filename, self.component_colors['filename'])
            colored_function = self._colorize(real_function, self.component_colors['function'])
            colored_line = self._colorize(str(real_lineno), self.component_colors['line'])
            colored_level = self._colorize(level_name, level_style['color'])
            colored_message = self._colorize(message, self.component_colors['message'])
            level_icon = level_style['icon']
        else:
            colored_timestamp = timestamp
            colored_filename = real_filename
            colored_function = real_function
            colored_line = str(real_lineno)
            colored_level = level_name
            colored_message = message
            level_icon = ''
        
        # 构建基础格式
        formatted_message = (
            f"{colored_timestamp} "
            f"[{colored_filename}:{colored_function}:{colored_line}] "
            f"{level_icon} {colored_level}"
        )
        
        # 添加分类信息
        if category:
            colored_category = self._colorize(f"[{category}]", self.component_colors['category'])
            formatted_message += f" {colored_category}"
        
        # 添加消息
        formatted_message += f": {colored_message}"
        
        return formatted_message


class ColoredLogger:
    """精确定位的彩色日志记录器"""
    
    def __init__(self, name: str, level: int = logging.INFO, use_colors: bool = True):
        """
        初始化彩色日志记录器
        
        参数:
            name (str): 日志记录器名称
            level (int): 日志级别
            use_colors (bool): 是否使用颜色
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            # 创建控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            # 设置精确定位的彩色格式化器
            formatter = AccurateColoredFormatter(use_colors=use_colors)
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            self.logger.addHandler(console_handler)
        
        # 防止日志传播到父记录器
        self.logger.propagate = False
    
    def _create_record_with_category(self, level: str, message: str, category: str = None):
        """创建带分类信息的日志记录"""
        # 直接调用logger的_log方法，不经过额外的包装
        if category:
            extra = {'category': category}
        else:
            extra = {}
            
        # 使用getattr获取对应的日志方法
        log_method = getattr(self.logger, level.lower())
        log_method(message, extra=extra)
    
    def debug(self, message: str, category: str = None):
        """调试日志"""
        self._create_record_with_category('DEBUG', message, category)
    
    def info(self, message: str, category: str = None):
        """信息日志"""
        self._create_record_with_category('INFO', message, category)
    
    def warning(self, message: str, category: str = None):
        """警告日志"""
        self._create_record_with_category('WARNING', message, category)
    
    def error(self, message: str, category: str = None):
        """错误日志"""
        self._create_record_with_category('ERROR', message, category)
    
    def critical(self, message: str, category: str = None):
        """严重错误日志"""
        self._create_record_with_category('CRITICAL', message, category)
    
    def step(self, step_name: str, message: str):
        """步骤日志（特殊格式）"""
        self.info(f"[STEP] {step_name} -> {message}", category="STEP")
    
    def checkpoint(self, checkpoint_name: str):
        """检查点日志"""
        self.info(f"CHECKPOINT: {checkpoint_name}", category="CHECKPOINT")
    
    def progress(self, current: int, total: int, message: str = ""):
        """进度日志"""
        percentage = (current / total) * 100 if total > 0 else 0
        progress_msg = f"Progress: {current}/{total} ({percentage:.1f}%)"
        if message:
            progress_msg += f" - {message}"
        self.info(progress_msg, category="PROGRESS")


def get_colored_logger(name: str, level: int = logging.INFO, use_colors: bool = True) -> ColoredLogger:
    """
    获取彩色日志记录器
    
    参数:
        name (str): 日志记录器名称
        level (int): 日志级别
        use_colors (bool): 是否使用颜色
    
    返回:
        ColoredLogger: 彩色日志记录器实例
    """
    return ColoredLogger(name, level, use_colors)


# =============== 工具函数 ===============
# 这些函数会正确地传递调用者信息

def log_detailed(logger: ColoredLogger, title: str, details: Dict[str, Any] = None, category: str = None):
    """
    记录详细日志
    
    参数:
        logger (ColoredLogger): 日志记录器
        title (str): 标题
        details (Dict[str, Any], optional): 详细信息字典
        category (str, optional): 日志分类
    """
    # 使用特殊的图标和格式来标识详细信息
    logger.info(f"📋 {title}", category=category or "DETAILS")
    if details:
        for key, value in details.items():
            logger.info(f"    • {key}: {value}", category=category or "DETAILS")


def log_llm_call(logger: ColoredLogger, model: str, tokens: int, message_length: int, category: str = None):
    """
    记录LLM调用日志
    
    参数:
        logger (ColoredLogger): 日志记录器
        model (str): 模型名称
        tokens (int): token数量
        message_length (int): 消息长度
        category (str, optional): 日志分类
    """
    logger.info(f"🤖 调用LLM: {model}, tokens: {tokens}", category=category or "LLM")
    logger.info(f"📏 消息长度: {message_length} 字符", category=category or "LLM")


def log_operation_start(logger: ColoredLogger, operation: str, category: str = None):
    """记录操作开始"""
    logger.info(f"🚀 开始操作: {operation}", category=category or "OPERATION")


def log_operation_success(logger: ColoredLogger, operation: str, category: str = None):
    """记录操作成功"""
    logger.info(f"✨ 操作成功: {operation}", category=category or "OPERATION")


def log_operation_error(logger: ColoredLogger, operation: str, error: str, category: str = None):
    """记录操作错误"""
    logger.error(f"💥 操作失败: {operation} - {error}", category=category or "OPERATION")


# =============== 装饰器 ===============

def log_function_call(logger: ColoredLogger, category: str = "FUNCTION"):
    """
    函数调用日志装饰器
    
    参数:
        logger (ColoredLogger): 日志记录器
        category (str): 日志分类
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.info(f"🔧 调用函数: {func_name}", category=category)
            try:
                result = func(*args, **kwargs)
                logger.info(f"✅ 函数完成: {func_name}", category=category)
                return result
            except Exception as e:
                logger.error(f"❌ 函数异常: {func_name} - {str(e)}", category=category)
                raise
        return wrapper
    return decorator 