"""
Type definitions for elicitation.
"""

from typing import Any, Callable, Awaitable, Union, Dict, Optional

# Type for elicitation callback
ElicitationCallback = Union[
    Callable[[str, Dict[str, Any]], Any],
    Callable[[str, Dict[str, Any]], Awaitable[Any]],
]

