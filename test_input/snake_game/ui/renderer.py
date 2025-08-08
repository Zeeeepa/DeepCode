"""
Game rendering and graphics module for Snake Game.
Handles all visual rendering including snake, food, backgrounds, and effects.
"""

import pygame
import math
from typing import List, Tuple, Optional, Dict, Any
from ..config.settings import *
from ..config.colors import *


class GameRenderer:
    """Handles all game rendering and visual effects."""
    
    def __init__(self, screen: pygame.Surface):
        """
        Initialize the game renderer.
        
        Args:
            screen: The main pygame screen surface
        """
        self.screen = screen
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        
        # Animation variables
        self.animation_time = 0.0
        self.pulse_offset = 0.0
        
        # Background gradient cache
        self._background_surface = None
        self._create_background()
        
        # Effect surfaces for performance
        self._glow_surfaces = {}
        
    def _create_background(self):
        """Create a gradient background surface."""
        self._background_surface = pygame.Surface((self.screen_width, self.screen_height))
        
        # Create vertical gradient
        for y in range(self.screen_height):
            factor = y / self.screen_height
            color = get_gradient_color(BACKGROUND_PRIMARY, BACKGROUND_SECONDARY, factor)
            pygame.draw.line(self._background_surface, color, (0, y), (self.screen_width, y))
    
    def update_animation(self, dt: float):
        """
        Update animation timers.
        
        Args:
            dt: Delta time in seconds
        """
        self.animation_time += dt
        self.pulse_offset = math.sin(self.animation_time * PULSE_SPEED) * 0.5 + 0.5
    
    def clear_screen(self):
        """Clear the screen with the background."""
        self.screen.blit(self._background_surface, (0, 0))
    
    def draw_grid(self, alpha: int = 30):
        """
        Draw a subtle grid overlay.
        
        Args:
            alpha: Transparency of the grid lines
        """
        grid_color = add_alpha(BORDER_COLOR, alpha)
        grid_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        
        # Vertical lines
        for x in range(0, self.screen_width, GRID_SIZE):
            pygame.draw.line(grid_surface, grid_color, (x, 0), (x, self.screen_height))
        
        # Horizontal lines
        for y in range(0, self.screen_height, GRID_SIZE):
            pygame.draw.line(grid_surface, grid_color, (0, y), (self.screen_width, y))
        
        self.screen.blit(grid_surface, (0, 0))
    
    def draw_snake(self, snake_segments: List[Tuple[int, int]], direction: Tuple[int, int]):
        """
        Draw the snake with gradient and glow effects.
        
        Args:
            snake_segments: List of (x, y) positions for snake segments
            direction: Current movement direction as (dx, dy)
        """
        if not snake_segments:
            return
        
        # Draw snake segments with gradient from head to tail
        for i, (x, y) in enumerate(snake_segments):
            # Calculate position on screen
            screen_x = x * GRID_SIZE
            screen_y = y * GRID_SIZE
            
            # Calculate color gradient (head is brightest)
            gradient_factor = 1.0 - (i / max(len(snake_segments), 1))
            
            if i == 0:  # Head
                # Pulsing head with direction indicator
                pulse_factor = 0.8 + 0.2 * self.pulse_offset
                head_color = tuple(int(c * pulse_factor) for c in SNAKE_HEAD_COLOR)
                
                # Draw head with glow effect
                self._draw_segment_with_glow(screen_x, screen_y, head_color, SNAKE_GLOW_COLOR)
                
                # Draw direction indicator (eyes)
                self._draw_snake_eyes(screen_x, screen_y, direction)
                
            else:  # Body segments
                body_color = get_gradient_color(SNAKE_BODY_COLOR, SNAKE_TAIL_COLOR, 1.0 - gradient_factor)
                self._draw_segment(screen_x, screen_y, body_color)
    
    def _draw_segment_with_glow(self, x: int, y: int, color: Tuple[int, int, int], glow_color: Tuple[int, int, int]):
        """Draw a snake segment with glow effect."""
        # Draw glow (larger, semi-transparent)
        glow_size = GRID_SIZE + 4
        glow_rect = pygame.Rect(x - 2, y - 2, glow_size, glow_size)
        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surface, add_alpha(glow_color, 100), glow_surface.get_rect())
        self.screen.blit(glow_surface, glow_rect.topleft)
        
        # Draw main segment
        self._draw_segment(x, y, color)
    
    def _draw_segment(self, x: int, y: int, color: Tuple[int, int, int]):
        """Draw a single snake segment."""
        rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
        
        # Draw main body
        pygame.draw.ellipse(self.screen, color, rect)
        
        # Draw highlight for 3D effect
        highlight_color = lighten_color(color, 0.3)
        highlight_rect = pygame.Rect(x + 2, y + 2, GRID_SIZE - 8, GRID_SIZE - 8)
        pygame.draw.ellipse(self.screen, highlight_color, highlight_rect)
        
        # Draw border
        pygame.draw.ellipse(self.screen, darken_color(color, 0.3), rect, 2)
    
    def _draw_snake_eyes(self, x: int, y: int, direction: Tuple[int, int]):
        """Draw eyes on the snake head based on direction."""
        eye_color = SNAKE_EYE_COLOR
        eye_size = 3
        
        # Calculate eye positions based on direction
        if direction == (0, -1):  # Up
            eye1_pos = (x + GRID_SIZE // 3, y + GRID_SIZE // 3)
            eye2_pos = (x + 2 * GRID_SIZE // 3, y + GRID_SIZE // 3)
        elif direction == (0, 1):  # Down
            eye1_pos = (x + GRID_SIZE // 3, y + 2 * GRID_SIZE // 3)
            eye2_pos = (x + 2 * GRID_SIZE // 3, y + 2 * GRID_SIZE // 3)
        elif direction == (-1, 0):  # Left
            eye1_pos = (x + GRID_SIZE // 3, y + GRID_SIZE // 3)
            eye2_pos = (x + GRID_SIZE // 3, y + 2 * GRID_SIZE // 3)
        else:  # Right
            eye1_pos = (x + 2 * GRID_SIZE // 3, y + GRID_SIZE // 3)
            eye2_pos = (x + 2 * GRID_SIZE // 3, y + 2 * GRID_SIZE // 3)
        
        pygame.draw.circle(self.screen, eye_color, eye1_pos, eye_size)
        pygame.draw.circle(self.screen, eye_color, eye2_pos, eye_size)
    
    def draw_food(self, food_position: Tuple[int, int], food_type: str = "normal"):
        """
        Draw food with pulsing animation.
        
        Args:
            food_position: (x, y) grid position of the food
            food_type: Type of food ("normal", "bonus", "special")
        """
        x, y = food_position
        screen_x = x * GRID_SIZE
        screen_y = y * GRID_SIZE
        
        # Get food color based on type
        if food_type == "bonus":
            base_color = FOOD_BONUS_COLOR
        elif food_type == "special":
            base_color = FOOD_SPECIAL_COLOR
        else:
            base_color = FOOD_COLOR
        
        # Pulsing effect
        pulse_factor = 0.8 + 0.2 * self.pulse_offset
        food_color = tuple(int(c * pulse_factor) for c in base_color)
        
        # Draw glow effect
        glow_size = GRID_SIZE + 6
        glow_rect = pygame.Rect(screen_x - 3, screen_y - 3, glow_size, glow_size)
        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surface, add_alpha(FOOD_GLOW_COLOR, 80), glow_surface.get_rect())
        self.screen.blit(glow_surface, glow_rect.topleft)
        
        # Draw main food
        food_rect = pygame.Rect(screen_x, screen_y, GRID_SIZE, GRID_SIZE)
        pygame.draw.ellipse(self.screen, food_color, food_rect)
        
        # Draw highlight
        highlight_color = lighten_color(food_color, 0.4)
        highlight_rect = pygame.Rect(screen_x + 3, screen_y + 3, GRID_SIZE - 6, GRID_SIZE - 6)
        pygame.draw.ellipse(self.screen, highlight_color, highlight_rect)
        
        # Draw sparkle effect for special food
        if food_type in ["bonus", "special"]:
            self._draw_sparkles(screen_x + GRID_SIZE // 2, screen_y + GRID_SIZE // 2)
    
    def _draw_sparkles(self, center_x: int, center_y: int):
        """Draw sparkle effects around special food."""
        sparkle_color = EFFECT_SPARKLE_COLOR
        time_offset = self.animation_time * 3
        
        for i in range(4):
            angle = (i * math.pi / 2) + time_offset
            distance = 15 + 5 * math.sin(time_offset * 2)
            
            sparkle_x = center_x + int(distance * math.cos(angle))
            sparkle_y = center_y + int(distance * math.sin(angle))
            
            # Draw sparkle as small diamond
            points = [
                (sparkle_x, sparkle_y - 3),
                (sparkle_x + 3, sparkle_y),
                (sparkle_x, sparkle_y + 3),
                (sparkle_x - 3, sparkle_y)
            ]
            pygame.draw.polygon(self.screen, sparkle_color, points)
    
    def draw_border(self):
        """Draw game area border."""
        border_rect = pygame.Rect(0, 0, self.screen_width, self.screen_height)
        pygame.draw.rect(self.screen, BORDER_COLOR, border_rect, BORDER_WIDTH)
        
        # Draw inner shadow
        shadow_color = darken_color(BORDER_COLOR, 0.5)
        inner_rect = pygame.Rect(BORDER_WIDTH, BORDER_WIDTH, 
                                self.screen_width - 2 * BORDER_WIDTH, 
                                self.screen_height - 2 * BORDER_WIDTH)
        pygame.draw.rect(self.screen, shadow_color, inner_rect, 2)
    
    def draw_game_over_overlay(self, alpha: int = 128):
        """
        Draw game over overlay effect.
        
        Args:
            alpha: Transparency of the overlay
        """
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill(add_alpha(GAME_OVER_OVERLAY_COLOR, alpha))
        self.screen.blit(overlay, (0, 0))
    
    def draw_pause_overlay(self, alpha: int = 100):
        """
        Draw pause overlay effect.
        
        Args:
            alpha: Transparency of the overlay
        """
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill(add_alpha(PAUSE_OVERLAY_COLOR, alpha))
        self.screen.blit(overlay, (0, 0))
    
    def draw_fade_transition(self, fade_factor: float, color: Tuple[int, int, int] = (0, 0, 0)):
        """
        Draw fade transition effect.
        
        Args:
            fade_factor: 0.0 (transparent) to 1.0 (opaque)
            color: Color to fade to
        """
        alpha = int(255 * fade_factor)
        fade_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        fade_surface.fill(add_alpha(color, alpha))
        self.screen.blit(fade_surface, (0, 0))
    
    def draw_particle_effect(self, particles: List[Dict[str, Any]]):
        """
        Draw particle effects.
        
        Args:
            particles: List of particle dictionaries with position, color, size, etc.
        """
        for particle in particles:
            pos = particle.get('position', (0, 0))
            color = particle.get('color', (255, 255, 255))
            size = particle.get('size', 2)
            alpha = particle.get('alpha', 255)
            
            if alpha > 0:
                particle_color = add_alpha(color, alpha)
                particle_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surface, particle_color, (size, size), size)
                self.screen.blit(particle_surface, (pos[0] - size, pos[1] - size))
    
    def get_screen_position(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        """
        Convert grid coordinates to screen coordinates.
        
        Args:
            grid_x: X position in grid
            grid_y: Y position in grid
            
        Returns:
            Screen coordinates as (x, y)
        """
        return (grid_x * GRID_SIZE, grid_y * GRID_SIZE)
    
    def get_grid_position(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """
        Convert screen coordinates to grid coordinates.
        
        Args:
            screen_x: X position on screen
            screen_y: Y position on screen
            
        Returns:
            Grid coordinates as (x, y)
        """
        return (screen_x // GRID_SIZE, screen_y // GRID_SIZE)