"""
Snake class implementation for the Snake game.
Handles snake movement, growth, and collision detection.
"""

import pygame
from typing import List, Tuple, Optional
from config.settings import (
    GRID_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT,
    SNAKE_INITIAL_POSITION, SNAKE_INITIAL_DIRECTION,
    INITIAL_SNAKE_LENGTH
)
from config.colors import SNAKE_HEAD, SNAKE_BODY, SNAKE_TAIL


class Direction:
    """Direction constants for snake movement."""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class Snake:
    """
    Snake class that handles all snake-related functionality.
    
    Manages snake position, movement, growth, and rendering.
    """
    
    def __init__(self):
        """Initialize the snake with default position and direction."""
        self.reset()
    
    def reset(self):
        """Reset the snake to its initial state."""
        # Initialize snake body segments
        self.body = []
        start_x, start_y = SNAKE_INITIAL_POSITION
        
        # Create initial snake body
        for i in range(INITIAL_SNAKE_LENGTH):
            segment_x = start_x - (i * GRID_SIZE)
            self.body.append((segment_x, start_y))
        
        # Set initial direction
        self.direction = SNAKE_INITIAL_DIRECTION
        self.next_direction = SNAKE_INITIAL_DIRECTION
        
        # Growth tracking
        self.grow_pending = 0
        self.last_tail_position = None
    
    def get_head_position(self) -> Tuple[int, int]:
        """Get the position of the snake's head."""
        return self.body[0] if self.body else (0, 0)
    
    def get_body_segments(self) -> List[Tuple[int, int]]:
        """Get all body segments except the head."""
        return self.body[1:] if len(self.body) > 1 else []
    
    def get_tail_position(self) -> Tuple[int, int]:
        """Get the position of the snake's tail."""
        return self.body[-1] if self.body else (0, 0)
    
    def set_direction(self, new_direction: Tuple[int, int]):
        """
        Set the next direction for the snake.
        
        Args:
            new_direction: Tuple representing the direction (dx, dy)
        """
        # Prevent the snake from moving directly backwards
        current_dx, current_dy = self.direction
        new_dx, new_dy = new_direction
        
        # Check if the new direction is opposite to current direction
        if (current_dx + new_dx == 0 and current_dy + new_dy == 0):
            return  # Ignore the direction change
        
        self.next_direction = new_direction
    
    def move(self):
        """Move the snake one step in the current direction."""
        if not self.body:
            return
        
        # Update direction
        self.direction = self.next_direction
        
        # Calculate new head position
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx * GRID_SIZE, head_y + dy * GRID_SIZE)
        
        # Store the current tail position before moving
        self.last_tail_position = self.body[-1]
        
        # Add new head
        self.body.insert(0, new_head)
        
        # Handle growth
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            # Remove tail if not growing
            self.body.pop()
    
    def grow(self, segments: int = 1):
        """
        Make the snake grow by the specified number of segments.
        
        Args:
            segments: Number of segments to grow
        """
        self.grow_pending += segments
    
    def check_wall_collision(self) -> bool:
        """
        Check if the snake has collided with the walls.
        
        Returns:
            True if collision detected, False otherwise
        """
        if not self.body:
            return False
        
        head_x, head_y = self.body[0]
        
        # Check boundaries
        if (head_x < 0 or head_x >= SCREEN_WIDTH or
            head_y < 0 or head_y >= SCREEN_HEIGHT):
            return True
        
        return False
    
    def check_self_collision(self) -> bool:
        """
        Check if the snake has collided with itself.
        
        Returns:
            True if collision detected, False otherwise
        """
        if len(self.body) < 4:  # Can't collide with itself if too short
            return False
        
        head = self.body[0]
        
        # Check if head collides with any body segment (excluding head itself)
        for segment in self.body[1:]:
            if head == segment:
                return True
        
        return False
    
    def get_length(self) -> int:
        """Get the current length of the snake."""
        return len(self.body)
    
    def get_score_value(self) -> int:
        """Get the score value based on snake length."""
        return max(0, self.get_length() - INITIAL_SNAKE_LENGTH)
    
    def render(self, surface: pygame.Surface):
        """
        Render the snake on the given surface.
        
        Args:
            surface: Pygame surface to render on
        """
        if not self.body:
            return
        
        # Render body segments
        for i, (x, y) in enumerate(self.body):
            # Create rectangle for the segment
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            
            if i == 0:
                # Head
                pygame.draw.rect(surface, SNAKE_HEAD, rect)
                pygame.draw.rect(surface, (255, 255, 255), rect, 2)  # White border
                
                # Draw eyes
                eye_size = GRID_SIZE // 6
                eye_offset = GRID_SIZE // 4
                
                # Determine eye positions based on direction
                dx, dy = self.direction
                if dx > 0:  # Moving right
                    eye1_pos = (x + GRID_SIZE - eye_offset, y + eye_offset)
                    eye2_pos = (x + GRID_SIZE - eye_offset, y + GRID_SIZE - eye_offset)
                elif dx < 0:  # Moving left
                    eye1_pos = (x + eye_offset, y + eye_offset)
                    eye2_pos = (x + eye_offset, y + GRID_SIZE - eye_offset)
                elif dy > 0:  # Moving down
                    eye1_pos = (x + eye_offset, y + GRID_SIZE - eye_offset)
                    eye2_pos = (x + GRID_SIZE - eye_offset, y + GRID_SIZE - eye_offset)
                else:  # Moving up
                    eye1_pos = (x + eye_offset, y + eye_offset)
                    eye2_pos = (x + GRID_SIZE - eye_offset, y + eye_offset)
                
                # Draw eyes
                pygame.draw.circle(surface, (255, 255, 255), eye1_pos, eye_size)
                pygame.draw.circle(surface, (255, 255, 255), eye2_pos, eye_size)
                pygame.draw.circle(surface, (0, 0, 0), eye1_pos, eye_size // 2)
                pygame.draw.circle(surface, (0, 0, 0), eye2_pos, eye_size // 2)
                
            elif i == len(self.body) - 1:
                # Tail
                pygame.draw.rect(surface, SNAKE_TAIL, rect)
                pygame.draw.rect(surface, (200, 200, 200), rect, 1)  # Light border
            else:
                # Body
                pygame.draw.rect(surface, SNAKE_BODY, rect)
                pygame.draw.rect(surface, (150, 150, 150), rect, 1)  # Gray border
    
    def get_head_rect(self) -> pygame.Rect:
        """Get the rectangle representing the snake's head."""
        if not self.body:
            return pygame.Rect(0, 0, 0, 0)
        
        head_x, head_y = self.body[0]
        return pygame.Rect(head_x, head_y, GRID_SIZE, GRID_SIZE)
    
    def contains_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if the given position is occupied by the snake.
        
        Args:
            position: Position to check (x, y)
            
        Returns:
            True if position is occupied by snake, False otherwise
        """
        return position in self.body