"""
UI package for Snake Game.

This package contains all user interface components including:
- Renderer: Game rendering and graphics
- Menu: Start menu and game over screen
- HUD: Score display and UI elements
"""

from .renderer import GameRenderer
from .menu import MenuManager
from .hud import HUDManager

__all__ = ['GameRenderer', 'MenuManager', 'HUDManager']