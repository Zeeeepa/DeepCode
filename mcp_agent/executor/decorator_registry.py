"""
Decorator registry for MCP Agent framework.
"""

from typing import Any, Dict, List, Optional, Callable


class DecoratorRegistry:
    """
    Registry for decorators.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self.decorators: Dict[str, Callable] = {}
    
    def register(self, name: str, decorator: Callable) -> None:
        """
        Register a decorator.
        
        Args:
            name: Decorator name
            decorator: Decorator function
        """
        self.decorators[name] = decorator
    
    def get(self, name: str) -> Optional[Callable]:
        """
        Get a decorator by name.
        
        Args:
            name: Decorator name
            
        Returns:
            Decorator function, or None if not found
        """
        return self.decorators.get(name)


def register_asyncio_decorators(registry: DecoratorRegistry) -> None:
    """
    Register asyncio decorators.
    
    Args:
        registry: Decorator registry
    """
    # Stub implementation
    pass


def register_temporal_decorators(registry: DecoratorRegistry) -> None:
    """
    Register temporal decorators.
    
    Args:
        registry: Decorator registry
    """
    # Stub implementation
    pass

