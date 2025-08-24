"""
Workflow signal for MCP Agent framework.
"""

from typing import Any, Dict, List, Optional, Callable, Awaitable, Union


# Type for signal wait callback
SignalWaitCallback = Union[
    Callable[[str, Dict[str, Any]], None],
    Callable[[str, Dict[str, Any]], Awaitable[None]],
]

