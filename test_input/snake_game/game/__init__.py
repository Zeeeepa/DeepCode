"""
Game package for Snake Game (贪吃蛇)

This package contains all the core game logic components:
- Snake: Snake entity and movement logic
- Food: Food generation and management
- GameState: Game state management
- Collision: Collision detection logic
"""

from .snake import Snake
from .food import Food
from .game_state import GameStateManager
from .collision import CollisionDetector

__all__ = [
    'Snake',
    'Food', 
    'GameStateManager',
    'CollisionDetector'
]

__version__ = '1.0.0'
__author__ = 'Snake Game Developer'