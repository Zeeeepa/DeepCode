"""
Collision Detection System for Snake Game

This module provides comprehensive collision detection functionality for the Snake game,
including wall collisions, self-collisions, food collisions, and advanced collision
utilities for game mechanics.
"""

import pygame
from typing import List, Tuple, Optional, Set, Dict, Any
from enum import Enum
import math

from ..config.settings import GAME_SETTINGS


class CollisionType(Enum):
    """Types of collisions that can occur in the game."""
    NONE = "none"
    WALL = "wall"
    SELF = "self"
    FOOD = "food"
    BOUNDARY = "boundary"
    OBSTACLE = "obstacle"


class CollisionResult:
    """Result of a collision detection check."""
    
    def __init__(self, collision_type: CollisionType = CollisionType.NONE, 
                 position: Optional[Tuple[int, int]] = None,
                 data: Optional[Dict[str, Any]] = None):
        self.type = collision_type
        self.position = position
        self.data = data or {}
        self.timestamp = pygame.time.get_ticks()
    
    def __bool__(self) -> bool:
        """Return True if there was a collision."""
        return self.type != CollisionType.NONE
    
    def is_fatal(self) -> bool:
        """Return True if this collision should end the game."""
        return self.type in [CollisionType.WALL, CollisionType.SELF, CollisionType.BOUNDARY]
    
    def get_score_value(self) -> int:
        """Get the score value associated with this collision."""
        if self.type == CollisionType.FOOD:
            return self.data.get('score_value', 10)
        return 0


class CollisionDetector:
    """
    Advanced collision detection system for the Snake game.
    
    Provides methods for detecting various types of collisions including
    wall collisions, self-collisions, food collisions, and boundary checks.
    """
    
    def __init__(self):
        """Initialize the collision detector with game settings."""
        self.grid_width = GAME_SETTINGS['GRID_WIDTH']
        self.grid_height = GAME_SETTINGS['GRID_HEIGHT']
        self.cell_size = GAME_SETTINGS['CELL_SIZE']
        
        # Collision history for debugging and analytics
        self.collision_history: List[CollisionResult] = []
        self.max_history_size = 100
        
        # Performance optimization: cache frequently used calculations
        self._boundary_cache = self._calculate_boundaries()
    
    def _calculate_boundaries(self) -> Dict[str, int]:
        """Calculate and cache boundary values for performance."""
        return {
            'min_x': 0,
            'max_x': self.grid_width - 1,
            'min_y': 0,
            'max_y': self.grid_height - 1
        }
    
    def check_wall_collision(self, position: Tuple[int, int]) -> CollisionResult:
        """
        Check if a position collides with the game boundaries.
        
        Args:
            position: The (x, y) grid position to check
            
        Returns:
            CollisionResult indicating if there was a wall collision
        """
        x, y = position
        boundaries = self._boundary_cache
        
        if (x < boundaries['min_x'] or x > boundaries['max_x'] or 
            y < boundaries['min_y'] or y > boundaries['max_y']):
            
            result = CollisionResult(
                CollisionType.WALL,
                position,
                {
                    'boundary_exceeded': {
                        'x': x < boundaries['min_x'] or x > boundaries['max_x'],
                        'y': y < boundaries['min_y'] or y > boundaries['max_y']
                    },
                    'distance_from_boundary': min(
                        x, boundaries['max_x'] - x, y, boundaries['max_y'] - y
                    )
                }
            )
            self._add_to_history(result)
            return result
        
        return CollisionResult()
    
    def check_self_collision(self, head_position: Tuple[int, int], 
                           body_positions: List[Tuple[int, int]]) -> CollisionResult:
        """
        Check if the snake's head collides with its own body.
        
        Args:
            head_position: The position of the snake's head
            body_positions: List of all body segment positions
            
        Returns:
            CollisionResult indicating if there was a self-collision
        """
        # Skip the first body segment (neck) to avoid false positives
        body_to_check = body_positions[1:] if len(body_positions) > 1 else []
        
        if head_position in body_to_check:
            # Find which segment was hit
            collision_index = body_to_check.index(head_position) + 1
            
            result = CollisionResult(
                CollisionType.SELF,
                head_position,
                {
                    'collision_segment_index': collision_index,
                    'total_body_length': len(body_positions),
                    'collision_distance_from_tail': len(body_positions) - collision_index
                }
            )
            self._add_to_history(result)
            return result
        
        return CollisionResult()
    
    def check_food_collision(self, head_position: Tuple[int, int], 
                           food_position: Optional[Tuple[int, int]],
                           food_data: Optional[Dict[str, Any]] = None) -> CollisionResult:
        """
        Check if the snake's head collides with food.
        
        Args:
            head_position: The position of the snake's head
            food_position: The position of the food item
            food_data: Additional data about the food (score value, type, etc.)
            
        Returns:
            CollisionResult indicating if there was a food collision
        """
        if food_position is None:
            return CollisionResult()
        
        if head_position == food_position:
            data = food_data.copy() if food_data else {}
            data.update({
                'food_position': food_position,
                'score_value': data.get('score_value', 10)
            })
            
            result = CollisionResult(
                CollisionType.FOOD,
                head_position,
                data
            )
            self._add_to_history(result)
            return result
        
        return CollisionResult()
    
    def check_multiple_collisions(self, head_position: Tuple[int, int],
                                body_positions: List[Tuple[int, int]],
                                food_position: Optional[Tuple[int, int]] = None,
                                food_data: Optional[Dict[str, Any]] = None) -> List[CollisionResult]:
        """
        Check for multiple types of collisions simultaneously.
        
        Args:
            head_position: The position of the snake's head
            body_positions: List of all body segment positions
            food_position: The position of the food item
            food_data: Additional data about the food
            
        Returns:
            List of CollisionResult objects for all detected collisions
        """
        collisions = []
        
        # Check wall collision
        wall_collision = self.check_wall_collision(head_position)
        if wall_collision:
            collisions.append(wall_collision)
        
        # Check self collision
        self_collision = self.check_self_collision(head_position, body_positions)
        if self_collision:
            collisions.append(self_collision)
        
        # Check food collision
        food_collision = self.check_food_collision(head_position, food_position, food_data)
        if food_collision:
            collisions.append(food_collision)
        
        return collisions
    
    def is_position_valid(self, position: Tuple[int, int],
                         occupied_positions: Optional[List[Tuple[int, int]]] = None) -> bool:
        """
        Check if a position is valid (within bounds and not occupied).
        
        Args:
            position: The position to check
            occupied_positions: List of positions that are already occupied
            
        Returns:
            True if the position is valid, False otherwise
        """
        # Check boundaries
        wall_collision = self.check_wall_collision(position)
        if wall_collision:
            return False
        
        # Check if position is occupied
        if occupied_positions and position in occupied_positions:
            return False
        
        return True
    
    def get_valid_positions(self, occupied_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Get all valid positions on the grid that are not occupied.
        
        Args:
            occupied_positions: List of positions that are already occupied
            
        Returns:
            List of all valid, unoccupied positions
        """
        valid_positions = []
        occupied_set = set(occupied_positions)  # Use set for O(1) lookup
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                position = (x, y)
                if position not in occupied_set:
                    valid_positions.append(position)
        
        return valid_positions
    
    def get_safe_spawn_positions(self, snake_positions: List[Tuple[int, int]],
                               min_distance: int = 3) -> List[Tuple[int, int]]:
        """
        Get positions that are safe for spawning food (away from snake).
        
        Args:
            snake_positions: List of all snake segment positions
            min_distance: Minimum distance from snake required
            
        Returns:
            List of safe spawn positions
        """
        safe_positions = []
        snake_set = set(snake_positions)
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                position = (x, y)
                
                # Skip if position is occupied by snake
                if position in snake_set:
                    continue
                
                # Check minimum distance from all snake segments
                min_dist_to_snake = min(
                    abs(x - sx) + abs(y - sy)  # Manhattan distance
                    for sx, sy in snake_positions
                )
                
                if min_dist_to_snake >= min_distance:
                    safe_positions.append(position)
        
        return safe_positions
    
    def predict_collision(self, current_position: Tuple[int, int],
                         direction: Tuple[int, int],
                         steps: int,
                         body_positions: List[Tuple[int, int]]) -> Optional[CollisionResult]:
        """
        Predict if a collision will occur within a certain number of steps.
        
        Args:
            current_position: Current head position
            direction: Movement direction (dx, dy)
            steps: Number of steps to look ahead
            body_positions: Current body positions
            
        Returns:
            CollisionResult if a collision is predicted, None otherwise
        """
        x, y = current_position
        dx, dy = direction
        
        for step in range(1, steps + 1):
            future_position = (x + dx * step, y + dy * step)
            
            # Check wall collision
            wall_collision = self.check_wall_collision(future_position)
            if wall_collision:
                wall_collision.data['predicted_steps'] = step
                return wall_collision
            
            # Check self collision (body will move too, so this is approximate)
            # For simplicity, we check against current body positions
            if step == 1:  # Only check immediate next step for self collision
                self_collision = self.check_self_collision(future_position, body_positions)
                if self_collision:
                    self_collision.data['predicted_steps'] = step
                    return self_collision
        
        return None
    
    def get_collision_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collisions that have occurred.
        
        Returns:
            Dictionary containing collision statistics
        """
        if not self.collision_history:
            return {
                'total_collisions': 0,
                'collision_types': {},
                'fatal_collisions': 0,
                'average_time_between_collisions': 0
            }
        
        collision_counts = {}
        fatal_count = 0
        
        for collision in self.collision_history:
            collision_type = collision.type.value
            collision_counts[collision_type] = collision_counts.get(collision_type, 0) + 1
            
            if collision.is_fatal():
                fatal_count += 1
        
        # Calculate average time between collisions
        if len(self.collision_history) > 1:
            time_diffs = []
            for i in range(1, len(self.collision_history)):
                time_diff = (self.collision_history[i].timestamp - 
                           self.collision_history[i-1].timestamp)
                time_diffs.append(time_diff)
            avg_time = sum(time_diffs) / len(time_diffs)
        else:
            avg_time = 0
        
        return {
            'total_collisions': len(self.collision_history),
            'collision_types': collision_counts,
            'fatal_collisions': fatal_count,
            'average_time_between_collisions': avg_time,
            'most_recent_collision': self.collision_history[-1].type.value if self.collision_history else None
        }
    
    def _add_to_history(self, collision: CollisionResult) -> None:
        """Add a collision to the history, maintaining size limit."""
        self.collision_history.append(collision)
        
        # Maintain history size limit
        if len(self.collision_history) > self.max_history_size:
            self.collision_history.pop(0)
    
    def reset(self) -> None:
        """Reset the collision detector state."""
        self.collision_history.clear()
    
    def get_distance_to_wall(self, position: Tuple[int, int], 
                           direction: Tuple[int, int]) -> int:
        """
        Get the distance to the nearest wall in a given direction.
        
        Args:
            position: Starting position
            direction: Direction to check (dx, dy)
            
        Returns:
            Number of steps to reach the wall
        """
        x, y = position
        dx, dy = direction
        
        if dx == 0 and dy == 0:
            return 0
        
        steps = 0
        while True:
            steps += 1
            next_pos = (x + dx * steps, y + dy * steps)
            
            if self.check_wall_collision(next_pos):
                return steps - 1  # Return steps before hitting wall
        
        return 0  # Should never reach here
    
    def get_nearest_wall_distance(self, position: Tuple[int, int]) -> int:
        """
        Get the distance to the nearest wall from a position.
        
        Args:
            position: The position to check from
            
        Returns:
            Distance to the nearest wall
        """
        x, y = position
        boundaries = self._boundary_cache
        
        distances = [
            x - boundaries['min_x'],  # Distance to left wall
            boundaries['max_x'] - x,  # Distance to right wall
            y - boundaries['min_y'],  # Distance to top wall
            boundaries['max_y'] - y   # Distance to bottom wall
        ]
        
        return min(distances)


# Utility functions for common collision scenarios
def check_basic_collision(head_position: Tuple[int, int],
                         body_positions: List[Tuple[int, int]],
                         grid_width: int, grid_height: int) -> bool:
    """
    Simple collision check for basic game logic.
    
    Args:
        head_position: Snake head position
        body_positions: Snake body positions
        grid_width: Width of the game grid
        grid_height: Height of the game grid
        
    Returns:
        True if there's a collision, False otherwise
    """
    x, y = head_position
    
    # Check wall collision
    if x < 0 or x >= grid_width or y < 0 or y >= grid_height:
        return True
    
    # Check self collision
    if head_position in body_positions[1:]:  # Skip neck
        return True
    
    return False


def get_collision_direction(position1: Tuple[int, int], 
                          position2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Get the direction vector from position1 to position2.
    
    Args:
        position1: Starting position
        position2: Target position
        
    Returns:
        Direction tuple (dx, dy) normalized to unit values
    """
    x1, y1 = position1
    x2, y2 = position2
    
    dx = x2 - x1
    dy = y2 - y1
    
    # Normalize to unit vector
    if dx != 0:
        dx = 1 if dx > 0 else -1
    if dy != 0:
        dy = 1 if dy > 0 else -1
    
    return (dx, dy)


def calculate_collision_impact(collision_result: CollisionResult,
                             snake_length: int) -> Dict[str, Any]:
    """
    Calculate the impact of a collision on the game state.
    
    Args:
        collision_result: The collision that occurred
        snake_length: Current length of the snake
        
    Returns:
        Dictionary containing impact analysis
    """
    impact = {
        'game_over': collision_result.is_fatal(),
        'score_change': collision_result.get_score_value(),
        'length_change': 0,
        'special_effects': []
    }
    
    if collision_result.type == CollisionType.FOOD:
        impact['length_change'] = 1
        impact['special_effects'].append('growth')
        
        # Bonus effects based on snake length
        if snake_length > 10:
            impact['score_change'] *= 2
            impact['special_effects'].append('length_bonus')
    
    elif collision_result.type == CollisionType.WALL:
        impact['special_effects'].append('wall_hit')
    
    elif collision_result.type == CollisionType.SELF:
        impact['special_effects'].append('self_bite')
    
    return impact