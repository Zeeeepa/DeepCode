"""
Type definitions for human input.
"""

from typing import Any, Callable, Awaitable, Union, Dict, Optional

# Type for human input callback
HumanInputCallback = Union[
    Callable[[str, Dict[str, Any]], str],
    Callable[[str, Dict[str, Any]], Awaitable[str]],
]

