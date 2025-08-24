"""
Signal registry for MCP Agent framework.
"""

from typing import Any, Dict, List, Optional, Callable


class SignalRegistry:
    """
    Registry for signals.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self.signals: Dict[str, Any] = {}
    
    def register(self, name: str, signal: Any) -> None:
        """
        Register a signal.
        
        Args:
            name: Signal name
            signal: Signal object
        """
        self.signals[name] = signal
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get a signal by name.
        
        Args:
            name: Signal name
            
        Returns:
            Signal object, or None if not found
        """
        return self.signals.get(name)

