"""
Logger implementation for MCP Agent framework.
"""

import logging
from typing import Any, Dict, Optional, Union


class MCPLogger:
    """
    Logger for MCP Agent framework.
    """
    
    def __init__(self, name: str, level: str = "info"):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            level: Log level
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Set level
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        self.logger.setLevel(level_map.get(level.lower(), logging.INFO))
        
        # Add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def debug(self, message: str, data: Any = None):
        """Log a debug message."""
        self.logger.debug(message)
    
    def info(self, message: str, data: Any = None):
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str, data: Any = None):
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str, data: Any = None):
        """Log an error message."""
        self.logger.error(message)
    
    def critical(self, message: str, data: Any = None):
        """Log a critical message."""
        self.logger.critical(message)


def get_logger(name: str) -> MCPLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return MCPLogger(name)

