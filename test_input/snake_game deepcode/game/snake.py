"""
Snake class implementation for the Snake Game
"""
import pygame
from collections import deque
from . import settings

class Snake:
    def __init__(self):
        """Initialize the snake with starting position and length"""
        # Start at the center of the grid
        start_x = settings.GRID_DIMENSION // 2
        start_y = settings.GRID_DIMENSION // 2
        
        # Initialize snake body as a deque for efficient append/pop operations
        self.body = deque([(start_x - i, start_y) for i in range(settings.INITIAL_SNAKE_LENGTH)])
        
        # Movement direction - start moving right
        self.direction = settings.RIGHT
        self.next_direction = settings.RIGHT
        
        # Game properties
        self.speed = settings.INITIAL_SNAKE_SPEED
        self.growth_pending = 0
        self.alive = True

    @property
    def head(self):
        """Get the position of snake's head"""
        return self.body[0]

    def update(self):
        """Update snake position and check for collisions"""
        if not self.alive:
            return

        # Update direction
        self.direction = self.next_direction

        # Calculate new head position
        new_x = (self.head[0] + self.direction[0]) % settings.GRID_DIMENSION
        new_y = (self.head[1] + self.direction[1]) % settings.GRID_DIMENSION
        new_head = (new_x, new_y)

        # Check for self collision
        if new_head in list(self.body)[1:]:
            self.alive = False
            return

        # Add new head
        self.body.appendleft(new_head)

        # Remove tail if no growth is pending
        if self.growth_pending > 0:
            self.growth_pending -= 1
        else:
            self.body.pop()

    def move(self):
        """Move the snake - alias for update() method"""
        self.update()

    def check_collision(self):
        """Check if snake has collided with itself or boundaries"""
        # Check if snake is dead (already handled collision)
        if not self.alive:
            return True
        
        # Get current head position
        head_x, head_y = self.head
        
        # Check boundary collision (if not using wrap-around)
        # Note: Current implementation uses wrap-around, so no boundary collision
        # But we'll keep this method for potential future use
        if (head_x < 0 or head_x >= settings.GRID_DIMENSION or 
            head_y < 0 or head_y >= settings.GRID_DIMENSION):
            self.alive = False
            return True
        
        # Check self collision (head hitting body)
        if self.head in list(self.body)[1:]:
            self.alive = False
            return True
        
        return False

    def change_direction(self, new_direction):
        """Change snake's direction ensuring it can't reverse into itself"""
        # Prevent 180-degree turns
        if (new_direction[0] != -self.direction[0] or 
            new_direction[1] != -self.direction[1]):
            self.next_direction = new_direction

    def grow(self):
        """Increase snake length"""
        self.growth_pending += 1

    def increase_speed(self):
        """Increase snake speed within limits"""
        self.speed = min(
            self.speed * settings.SPEED_INCREASE_FACTOR,
            settings.MAX_SPEED
        )

    def draw(self, screen):
        """Draw the snake on the screen"""
        cell_size = settings.WINDOW_SIZE // settings.GRID_DIMENSION
        
        # Draw body segments
        for segment in self.body:
            x = segment[0] * cell_size
            y = segment[1] * cell_size
            
            # Draw segment with a slightly smaller size for visual separation
            pygame.draw.rect(
                screen,
                settings.GREEN,
                (x + 1, y + 1, cell_size - 2, cell_size - 2)
            )

        # Draw head with a different shade or marking
        head_x = self.head[0] * cell_size
        head_y = self.head[1] * cell_size
        pygame.draw.rect(
            screen,
            settings.BLUE,  # Different color for head
            (head_x + 1, head_y + 1, cell_size - 2, cell_size - 2)
        )

    def reset(self):
        """Reset snake to initial state"""
        start_x = settings.GRID_DIMENSION // 2
        start_y = settings.GRID_DIMENSION // 2
        self.body = deque([(start_x - i, start_y) for i in range(settings.INITIAL_SNAKE_LENGTH)])
        self.direction = settings.RIGHT
        self.next_direction = settings.RIGHT
        self.speed = settings.INITIAL_SNAKE_SPEED
        self.growth_pending = 0
        self.alive = True
