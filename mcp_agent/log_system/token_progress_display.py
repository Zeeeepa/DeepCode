"""
Token usage progress display.
"""

import asyncio
import sys
from typing import Optional, Dict, Any
from contextlib import contextmanager


class TokenProgressDisplay:
    """
    Display for token usage.
    
    This is a simplified implementation that displays token usage
    at the beginning of the input line.
    """
    
    def __init__(self, token_counter: Any, console: Optional[Any] = None):
        """
        Initialize the token progress display.
        
        Args:
            token_counter: Token counter instance
            console: Optional console instance
        """
        self.token_counter = token_counter
        self._watch_ids = []
        self._paused = False
        
    def start(self):
        """Start the progress display."""
        # Register watch on token counter
        asyncio.create_task(self._register_watch())
    
    async def _register_watch(self):
        """Register watch on token counter."""
        try:
            if hasattr(self.token_counter, "watch"):
                watch_id = await self.token_counter.watch(
                    callback=self._on_token_update,
                    node_type="app",
                    threshold=1,
                    throttle_ms=100,
                )
                self._watch_ids.append(watch_id)
        except Exception:
            # Silently ignore errors
            pass
    
    async def _unregister_watches(self):
        """Unregister all watches."""
        for watch_id in self._watch_ids:
            if hasattr(self.token_counter, "unwatch"):
                await self.token_counter.unwatch(watch_id)
        self._watch_ids.clear()
    
    def stop(self):
        """Stop the progress display."""
        if self._watch_ids:
            asyncio.create_task(self._unregister_watches())
    
    def pause(self):
        """Pause the progress display."""
        self._paused = True
    
    def resume(self):
        """Resume the progress display."""
        self._paused = False
    
    @contextmanager
    def paused(self):
        """Context manager for temporarily pausing the display."""
        self.pause()
        try:
            yield
        finally:
            self.resume()
    
    async def _on_token_update(self, node: Any, usage: Any):
        """
        Handle token usage updates.
        
        Args:
            node: Token node
            usage: Token usage
        """
        if self._paused:
            return
        
        # Get summary
        summary = None
        if hasattr(self.token_counter, "get_summary"):
            summary = await self.token_counter.get_summary()
        
        # Display token usage
        if summary:
            tokens = getattr(summary.usage, "total_tokens", 0)
            cost = getattr(summary, "cost", 0.0)
            
            # Print token usage at beginning of line
            sys.stdout.write(f"\r{tokens} tokens | ${cost:.4f}")
            sys.stdout.flush()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

