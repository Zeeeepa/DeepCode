"""
Main application class for MCP Agent framework.
"""

import asyncio
import os
import sys
from typing import Any, Dict, Optional, Type, TypeVar, Callable, List
from contextlib import asynccontextmanager

from mcp_agent.core.context import Context

R = TypeVar("R")


class MCPApp:
    """
    Main application class that manages global state and can host workflows.

    Example usage:
        app = MCPApp(name="cli_agent_orchestration")

        async with app.run() as mcp_agent_app:
            # App is initialized here
            pass
    """

    def __init__(
        self,
        name: str = "mcp_application",
        description: str | None = None,
    ):
        """
        Initialize the MCP application.

        Args:
            name: Name of the application
            description: Optional description
        """
        self.name = name
        self.description = description
        self.logger = None
        self.context = Context()

    async def initialize(self):
        """
        Initialize the application.
        """
        # Initialize minimal context
        self.context.config = self._load_config()
        
        # Set up logger
        from mcp_agent.log_system.logger import get_logger
        self.logger = get_logger(self.name)
        self.context.logger = self.logger
        
        # Set up token counter
        from mcp_agent.tracing.token_counter import TokenCounter
        self.context.token_counter = TokenCounter()
        
        # Log initialization
        self.logger.info(f"Initialized MCP application: {self.name}")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from mcp_agent.config.yaml.
        """
        # Simple config loader
        config = {
            "execution_engine": "asyncio",
            "logger": {
                "transports": ["console"],
                "level": "info",
                "progress_display": True
            },
            "mcp": {
                "servers": {}
            }
        }
        
        # Try to load from file if exists
        try:
            import yaml
            if os.path.exists("mcp_agent.config.yaml"):
                with open("mcp_agent.config.yaml", "r") as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            
        return config

    @asynccontextmanager
    async def run(self):
        """
        Run the application. Use as context manager.

        Example:
            async with app.run() as running_app:
                # App is initialized here
                pass
        """
        await self.initialize()

        # Push token tracking context for the app
        if hasattr(self.context, "token_counter") and self.context.token_counter:
            await self.context.token_counter.push(name=self.name, node_type="app")

        try:
            yield self
        finally:
            # Cleanup
            if hasattr(self.context, "token_counter") and self.context.token_counter:
                await self.context.token_counter.pop()
            
            self.logger.info(f"Shutting down MCP application: {self.name}")

