"""
Workflow task for MCP Agent framework.
"""

from typing import Any, Dict, List, Optional, Callable


class GlobalWorkflowTaskRegistry:
    """
    Global registry for workflow tasks.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self.tasks: Dict[str, Any] = {}
    
    def register(self, name: str, task: Any) -> None:
        """
        Register a task.
        
        Args:
            name: Task name
            task: Task object
        """
        self.tasks[name] = task
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get a task by name.
        
        Args:
            name: Task name
            
        Returns:
            Task object, or None if not found
        """
        return self.tasks.get(name)

