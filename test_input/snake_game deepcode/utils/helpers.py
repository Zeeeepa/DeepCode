import math
import pygame
import random

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

def get_random_position(grid_size, cell_size, margin=0):
    """Get a random position on the game grid."""
    x = random.randint(margin, grid_size[0] - margin - 1) * cell_size
    y = random.randint(margin, grid_size[1] - margin - 1) * cell_size
    return (x, y)

def grid_to_pixel(grid_pos, cell_size):
    """Convert grid coordinates to pixel coordinates."""
    return (grid_pos[0] * cell_size, grid_pos[1] * cell_size)

def pixel_to_grid(pixel_pos, cell_size):
    """Convert pixel coordinates to grid coordinates."""
    return (pixel_pos[0] // cell_size, pixel_pos[1] // cell_size)

def is_point_in_rect(point, rect):
    """Check if a point is inside a rectangle."""
    return rect.collidepoint(point)

def create_gradient_surface(size, start_color, end_color, vertical=True):
    """Create a surface with a color gradient."""
    surface = pygame.Surface(size)
    
    for i in range(size[1] if vertical else size[0]):
        factor = i / (size[1] - 1 if vertical else size[0] - 1)
        color = [
            start_color[j] + (end_color[j] - start_color[j]) * factor
            for j in range(3)
        ]
        if vertical:
            pygame.draw.line(surface, color, (0, i), (size[0], i))
        else:
            pygame.draw.line(surface, color, (i, 0), (i, size[1]))
    
    return surface

def lerp_color(color1, color2, factor):
    """Linear interpolation between two colors."""
    return [
        color1[i] + (color2[i] - color1[i]) * factor
        for i in range(3)
    ]

def create_shadow_text(text, font, color, shadow_color, offset=(2, 2)):
    """Create text surface with shadow effect."""
    # Create shadow surface
    shadow_surface = font.render(text, True, shadow_color)
    shadow_rect = shadow_surface.get_rect(topleft=offset)
    
    # Create main text surface
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(topleft=(0, 0))
    
    # Combine surfaces
    final_surface = pygame.Surface(
        (max(shadow_rect.right, text_rect.right),
         max(shadow_rect.bottom, text_rect.bottom)),
        pygame.SRCALPHA
    )
    final_surface.blit(shadow_surface, shadow_rect)
    final_surface.blit(text_surface, text_rect)
    
    return final_surface

def ease_out_quad(x):
    """Quadratic easing function for smooth animations."""
    return 1 - (1 - x) * (1 - x)

def clamp(value, min_value, max_value):
    """Clamp a value between min and max values."""
    return max(min_value, min(value, max_value))