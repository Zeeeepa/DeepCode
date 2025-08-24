"""
Task registry for MCP Agent framework.
"""

from typing import Any, Dict, List, Optional, Callable


class ActivityRegistry:
    """
    Registry for activities.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self.activities: Dict[str, Any] = {}
    
    def register(self, name: str, activity: Any) -> None:
        """
        Register an activity.
        
        Args:
            name: Activity name
            activity: Activity object
        """
        self.activities[name] = activity
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get an activity by name.
        
        Args:
            name: Activity name
            
        Returns:
            Activity object, or None if not found
        """
        return self.activities.get(name)

