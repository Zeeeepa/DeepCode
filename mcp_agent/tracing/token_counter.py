"""
Token counting and cost tracking system for MCP Agent framework.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set, Union, Tuple, Awaitable
from datetime import datetime
import uuid


@dataclass
class TokenUsageBase:
    """Base class for token usage information"""
    
    input_tokens: int = 0
    """Number of tokens in the input/prompt"""
    
    output_tokens: int = 0
    """Number of tokens in the output/completion"""
    
    total_tokens: int = 0
    """Total number of tokens (input + output)"""
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class TokenUsage(TokenUsageBase):
    """Token usage for a single LLM call with metadata"""
    
    model_name: Optional[str] = None
    """Name of the model used (e.g., 'gpt-4o', 'claude-3-opus')"""
    
    model_info: Optional[Any] = None
    """Full model metadata including provider, costs, capabilities"""
    
    timestamp: datetime = field(default_factory=datetime.now)
    """When this usage was recorded"""


@dataclass
class TokenNode:
    """Node in the token usage tree"""
    
    name: str
    """Name of this node (e.g., agent name, workflow name)"""
    
    node_type: str = "generic"
    """Type of this node (e.g., 'agent', 'workflow', 'app')"""
    
    parent: Optional["TokenNode"] = None
    """Parent node in the hierarchy"""
    
    children: List["TokenNode"] = field(default_factory=list)
    """Child nodes in the hierarchy"""
    
    usage: TokenUsage = field(default_factory=TokenUsage)
    """Token usage for this node"""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this node"""


@dataclass
class TokenSummary:
    """Summary of token usage"""
    
    usage: TokenUsage
    """Total token usage"""
    
    cost: float = 0.0
    """Estimated cost in USD"""
    
    node_count: int = 0
    """Number of nodes in the tree"""


class TokenCounter:
    """
    Token counter for tracking token usage across agents and subagents.
    """
    
    def __init__(self):
        """Initialize the token counter."""
        self._root = TokenNode(name="root", node_type="root")
        self._current_path: List[TokenNode] = [self._root]
        self._watches: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async def push(self, name: str, node_type: str = "generic") -> TokenNode:
        """
        Push a new node onto the stack.
        
        Args:
            name: Node name
            node_type: Node type
            
        Returns:
            The new node
        """
        async with self._lock:
            parent = self._current_path[-1]
            
            # Create new node
            node = TokenNode(name=name, node_type=node_type, parent=parent)
            parent.children.append(node)
            self._current_path.append(node)
            
            return node
    
    async def pop(self) -> Optional[TokenNode]:
        """
        Pop the current node from the stack.
        
        Returns:
            The popped node, or None if at root
        """
        async with self._lock:
            if len(self._current_path) <= 1:
                return None
            
            node = self._current_path.pop()
            return node
    
    async def add_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model_name: Optional[str] = None,
        model_info: Optional[Any] = None,
    ) -> TokenUsage:
        """
        Add token usage to the current node.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Model name
            model_info: Model info
            
        Returns:
            Updated token usage
        """
        async with self._lock:
            node = self._current_path[-1]
            
            # Update usage
            node.usage.input_tokens += input_tokens
            node.usage.output_tokens += output_tokens
            node.usage.total_tokens += input_tokens + output_tokens
            
            if model_name and not node.usage.model_name:
                node.usage.model_name = model_name
            
            if model_info and not node.usage.model_info:
                node.usage.model_info = model_info
            
            # Propagate to ancestors
            current = node.parent
            while current:
                current.usage.input_tokens += input_tokens
                current.usage.output_tokens += output_tokens
                current.usage.total_tokens += input_tokens + output_tokens
                current = current.parent
            
            # Trigger watches
            await self._trigger_watches(node)
            
            return node.usage
    
    async def get_summary(self) -> TokenSummary:
        """
        Get a summary of token usage.
        
        Returns:
            Token usage summary
        """
        async with self._lock:
            return TokenSummary(
                usage=self._root.usage,
                cost=0.0,  # Simplified implementation
                node_count=self._count_nodes(self._root),
            )
    
    def _count_nodes(self, node: TokenNode) -> int:
        """
        Count nodes in the tree.
        
        Args:
            node: Root node
            
        Returns:
            Number of nodes
        """
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    async def watch(
        self,
        callback: Union[
            Callable[[TokenNode, TokenUsage], None],
            Callable[[TokenNode, TokenUsage], Awaitable[None]],
        ],
        node: Optional[TokenNode] = None,
        node_type: Optional[str] = None,
        threshold: Optional[int] = None,
        throttle_ms: Optional[int] = None,
    ) -> str:
        """
        Watch for token usage changes.
        
        Args:
            callback: Callback function
            node: Specific node to watch
            node_type: Node type to watch
            threshold: Threshold for triggering callback
            throttle_ms: Throttle in milliseconds
            
        Returns:
            Watch ID
        """
        watch_id = str(uuid.uuid4())
        
        self._watches[watch_id] = {
            "callback": callback,
            "node": node,
            "node_type": node_type,
            "threshold": threshold,
            "throttle_ms": throttle_ms,
            "last_triggered": {},
            "is_async": asyncio.iscoroutinefunction(callback),
        }
        
        return watch_id
    
    async def unwatch(self, watch_id: str) -> bool:
        """
        Remove a watch.
        
        Args:
            watch_id: Watch ID
            
        Returns:
            True if watch was removed
        """
        if watch_id in self._watches:
            del self._watches[watch_id]
            return True
        return False
    
    async def _trigger_watches(self, node: TokenNode) -> None:
        """
        Trigger watches for a node.
        
        Args:
            node: Node that changed
        """
        for watch_id, watch in self._watches.items():
            # Check if this node matches the watch
            if watch["node"] and watch["node"].id != node.id:
                continue
            
            if watch["node_type"] and watch["node_type"] != node.node_type:
                continue
            
            # Check threshold
            if watch["threshold"] and node.usage.total_tokens < watch["threshold"]:
                continue
            
            # Check throttle
            now = time.time()
            last_time = watch["last_triggered"].get(node.id, 0)
            if watch["throttle_ms"] and (now - last_time) * 1000 < watch["throttle_ms"]:
                continue
            
            # Update last triggered time
            watch["last_triggered"][node.id] = now
            
            # Call callback
            if watch["is_async"]:
                asyncio.create_task(watch["callback"](node, node.usage))
            else:
                watch["callback"](node, node.usage)
    
    async def get_app_node(self) -> Optional[TokenNode]:
        """
        Get the app node.
        
        Returns:
            App node, or None if not found
        """
        # Find first node with type "app"
        for node in self._current_path:
            if node.node_type == "app":
                return node
        
        # Check children of root
        for child in self._root.children:
            if child.node_type == "app":
                return child
        
        return None

