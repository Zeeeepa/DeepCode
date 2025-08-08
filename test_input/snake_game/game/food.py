"""
Food management module for the Snake game.
Handles food generation, positioning, and visual effects.
"""

import pygame
import random
import math
from typing import Tuple, List, Optional
from ..config.settings import (
    GRID_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, FOOD_SPAWN_MARGIN,
    FOOD_SIZE, FOOD_SCORE_VALUE, FOOD_ANIMATION_SPEED, FOOD_PULSE_AMPLITUDE
)
from ..config.colors import FOOD_COLOR, FOOD_GLOW_COLOR, FOOD_SPECIAL_COLOR


class Food:
    """
    Manages food items in the Snake game.
    Handles spawning, positioning, animations, and special food types.
    """
    
    def __init__(self):
        """Initialize the food manager."""
        self.position: Optional[Tuple[int, int]] = None
        self.is_special: bool = False
        self.special_timer: int = 0
        self.special_duration: int = 300  # frames for special food
        self.animation_timer: float = 0.0
        self.pulse_scale: float = 1.0
        self.glow_alpha: int = 128
        self.spawn_count: int = 0
        
        # Calculate grid boundaries
        self.min_x = FOOD_SPAWN_MARGIN
        self.max_x = (SCREEN_WIDTH - FOOD_SPAWN_MARGIN) // GRID_SIZE
        self.min_y = FOOD_SPAWN_MARGIN
        self.max_y = (SCREEN_HEIGHT - FOOD_SPAWN_MARGIN) // GRID_SIZE
        
        # Special food properties
        self.special_chance = 0.1  # 10% chance for special food
        self.special_score_multiplier = 3
        
    def spawn(self, occupied_positions: List[Tuple[int, int]]) -> bool:
        """
        Spawn a new food item at a random valid position.
        
        Args:
            occupied_positions: List of positions occupied by the snake
            
        Returns:
            bool: True if food was successfully spawned, False otherwise
        """
        # Get all valid positions
        valid_positions = self._get_valid_positions(occupied_positions)
        
        if not valid_positions:
            return False
            
        # Choose random position
        self.position = random.choice(valid_positions)
        
        # Determine if this should be special food
        self.is_special = random.random() < self.special_chance
        if self.is_special:
            self.special_timer = self.special_duration
        else:
            self.special_timer = 0
            
        self.spawn_count += 1
        self.animation_timer = 0.0
        
        return True
        
    def _get_valid_positions(self, occupied_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Get all valid positions where food can be spawned.
        
        Args:
            occupied_positions: List of positions occupied by the snake
            
        Returns:
            List of valid grid positions
        """
        valid_positions = []
        
        for x in range(self.min_x, self.max_x):
            for y in range(self.min_y, self.max_y):
                position = (x, y)
                if position not in occupied_positions:
                    valid_positions.append(position)
                    
        return valid_positions
        
    def update(self, dt: float) -> None:
        """
        Update food animations and special food timer.
        
        Args:
            dt: Delta time in seconds
        """
        # Update animation timer
        self.animation_timer += dt * FOOD_ANIMATION_SPEED
        
        # Calculate pulsing effect
        self.pulse_scale = 1.0 + math.sin(self.animation_timer) * FOOD_PULSE_AMPLITUDE
        
        # Update glow effect
        self.glow_alpha = int(128 + math.sin(self.animation_timer * 2) * 64)
        
        # Update special food timer
        if self.is_special and self.special_timer > 0:
            self.special_timer -= 1
            if self.special_timer <= 0:
                self.is_special = False
                
    def get_position(self) -> Optional[Tuple[int, int]]:
        """
        Get the current food position.
        
        Returns:
            Current food position or None if no food exists
        """
        return self.position
        
    def get_pixel_position(self) -> Optional[Tuple[int, int]]:
        """
        Get the food position in pixel coordinates.
        
        Returns:
            Food position in pixels or None if no food exists
        """
        if self.position is None:
            return None
            
        x, y = self.position
        pixel_x = x * GRID_SIZE + GRID_SIZE // 2
        pixel_y = y * GRID_SIZE + GRID_SIZE // 2
        return (pixel_x, pixel_y)
        
    def consume(self) -> int:
        """
        Consume the food and return its score value.
        
        Returns:
            Score value of the consumed food
        """
        if self.position is None:
            return 0
            
        score = FOOD_SCORE_VALUE
        if self.is_special:
            score *= self.special_score_multiplier
            
        # Reset food state
        self.position = None
        self.is_special = False
        self.special_timer = 0
        
        return score
        
    def exists(self) -> bool:
        """
        Check if food currently exists.
        
        Returns:
            True if food exists, False otherwise
        """
        return self.position is not None
        
    def is_at_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if food is at the specified position.
        
        Args:
            position: Grid position to check
            
        Returns:
            True if food is at the position, False otherwise
        """
        return self.position == position
        
    def get_score_value(self) -> int:
        """
        Get the score value of the current food.
        
        Returns:
            Score value (higher for special food)
        """
        if not self.exists():
            return 0
            
        score = FOOD_SCORE_VALUE
        if self.is_special:
            score *= self.special_score_multiplier
            
        return score
        
    def get_render_info(self) -> dict:
        """
        Get rendering information for the food.
        
        Returns:
            Dictionary containing rendering data
        """
        if not self.exists():
            return {}
            
        pixel_pos = self.get_pixel_position()
        if pixel_pos is None:
            return {}
            
        # Calculate size with pulsing effect
        base_size = FOOD_SIZE
        current_size = int(base_size * self.pulse_scale)
        
        # Choose color based on special status
        if self.is_special:
            # Flash between special and normal color for special food
            flash_speed = 10
            if (self.special_timer // flash_speed) % 2:
                color = FOOD_SPECIAL_COLOR
            else:
                color = FOOD_COLOR
        else:
            color = FOOD_COLOR
            
        return {
            'position': pixel_pos,
            'size': current_size,
            'color': color,
            'glow_color': FOOD_GLOW_COLOR,
            'glow_alpha': self.glow_alpha,
            'is_special': self.is_special,
            'special_timer': self.special_timer,
            'pulse_scale': self.pulse_scale
        }
        
    def reset(self) -> None:
        """Reset the food manager to initial state."""
        self.position = None
        self.is_special = False
        self.special_timer = 0
        self.animation_timer = 0.0
        self.pulse_scale = 1.0
        self.glow_alpha = 128
        self.spawn_count = 0
        
    def get_stats(self) -> dict:
        """
        Get food statistics.
        
        Returns:
            Dictionary containing food stats
        """
        return {
            'spawn_count': self.spawn_count,
            'is_special': self.is_special,
            'special_timer': self.special_timer,
            'position': self.position,
            'exists': self.exists()
        }
        
    def force_special(self) -> None:
        """Force the current food to become special (for testing/cheats)."""
        if self.exists():
            self.is_special = True
            self.special_timer = self.special_duration
            
    def get_distance_to(self, position: Tuple[int, int]) -> float:
        """
        Calculate distance from food to given position.
        
        Args:
            position: Grid position to calculate distance to
            
        Returns:
            Distance in grid units, or infinity if no food exists
        """
        if not self.exists():
            return float('inf')
            
        food_x, food_y = self.position
        pos_x, pos_y = position
        
        return math.sqrt((food_x - pos_x) ** 2 + (food_y - pos_y) ** 2)
        
    def get_direction_to(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get the general direction from given position to food.
        
        Args:
            position: Starting position
            
        Returns:
            Direction tuple (dx, dy) where each component is -1, 0, or 1
        """
        if not self.exists():
            return (0, 0)
            
        food_x, food_y = self.position
        pos_x, pos_y = position
        
        dx = 0 if food_x == pos_x else (1 if food_x > pos_x else -1)
        dy = 0 if food_y == pos_y else (1 if food_y > pos_y else -1)
        
        return (dx, dy)


class FoodManager:
    """
    Advanced food management system that can handle multiple food items.
    Currently manages a single food item but designed for future expansion.
    """
    
    def __init__(self):
        """Initialize the food manager."""
        self.food = Food()
        self.total_food_consumed = 0
        self.special_food_consumed = 0
        
    def spawn_food(self, occupied_positions: List[Tuple[int, int]]) -> bool:
        """
        Spawn new food if none exists.
        
        Args:
            occupied_positions: List of positions occupied by the snake
            
        Returns:
            True if food was spawned, False otherwise
        """
        if not self.food.exists():
            return self.food.spawn(occupied_positions)
        return False
        
    def update(self, dt: float) -> None:
        """
        Update all food items.
        
        Args:
            dt: Delta time in seconds
        """
        self.food.update(dt)
        
    def check_collision(self, position: Tuple[int, int]) -> int:
        """
        Check if position collides with food and consume it.
        
        Args:
            position: Position to check for collision
            
        Returns:
            Score value if food was consumed, 0 otherwise
        """
        if self.food.is_at_position(position):
            score = self.food.consume()
            self.total_food_consumed += 1
            if score > FOOD_SCORE_VALUE:  # Was special food
                self.special_food_consumed += 1
            return score
        return 0
        
    def get_food_position(self) -> Optional[Tuple[int, int]]:
        """Get the current food position."""
        return self.food.get_position()
        
    def get_render_info(self) -> List[dict]:
        """
        Get rendering information for all food items.
        
        Returns:
            List of rendering dictionaries
        """
        info = self.food.get_render_info()
        return [info] if info else []
        
    def reset(self) -> None:
        """Reset the food manager."""
        self.food.reset()
        self.total_food_consumed = 0
        self.special_food_consumed = 0
        
    def get_stats(self) -> dict:
        """
        Get comprehensive food statistics.
        
        Returns:
            Dictionary containing all food stats
        """
        return {
            'total_consumed': self.total_food_consumed,
            'special_consumed': self.special_food_consumed,
            'current_food': self.food.get_stats()
        }