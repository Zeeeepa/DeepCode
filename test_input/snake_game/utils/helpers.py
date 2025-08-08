# Helper utility functions for Snake Game
# Contains common utility functions used throughout the game

import pygame
import json
import os
from typing import Tuple, List, Dict, Any, Optional
import math


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between min and max values.
    
    Args:
        value: The value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Clamped value between min_value and max_value
    """
    return max(min_value, min(value, max_value))


def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance between the two points
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def lerp(start: float, end: float, t: float) -> float:
    """
    Linear interpolation between two values.
    
    Args:
        start: Starting value
        end: Ending value
        t: Interpolation factor (0.0 to 1.0)
        
    Returns:
        Interpolated value
    """
    return start + (end - start) * clamp(t, 0.0, 1.0)


def lerp_color(color1: Tuple[int, int, int], color2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """
    Linear interpolation between two RGB colors.
    
    Args:
        color1: Starting color (r, g, b)
        color2: Ending color (r, g, b)
        t: Interpolation factor (0.0 to 1.0)
        
    Returns:
        Interpolated color (r, g, b)
    """
    r = int(lerp(color1[0], color2[0], t))
    g = int(lerp(color1[1], color2[1], t))
    b = int(lerp(color1[2], color2[2], t))
    return (r, g, b)


def grid_to_pixel(grid_pos: Tuple[int, int], cell_size: int, offset: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
    """
    Convert grid coordinates to pixel coordinates.
    
    Args:
        grid_pos: Grid position (x, y)
        cell_size: Size of each grid cell in pixels
        offset: Pixel offset (x, y)
        
    Returns:
        Pixel coordinates (x, y)
    """
    return (grid_pos[0] * cell_size + offset[0], grid_pos[1] * cell_size + offset[1])


def pixel_to_grid(pixel_pos: Tuple[int, int], cell_size: int, offset: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
    """
    Convert pixel coordinates to grid coordinates.
    
    Args:
        pixel_pos: Pixel position (x, y)
        cell_size: Size of each grid cell in pixels
        offset: Pixel offset (x, y)
        
    Returns:
        Grid coordinates (x, y)
    """
    return ((pixel_pos[0] - offset[0]) // cell_size, (pixel_pos[1] - offset[1]) // cell_size)


def load_json_file(file_path: str, default_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load data from a JSON file with error handling.
    
    Args:
        file_path: Path to the JSON file
        default_data: Default data to return if file doesn't exist or is invalid
        
    Returns:
        Loaded data or default data
    """
    if default_data is None:
        default_data = {}
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return default_data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return default_data


def save_json_file(file_path: str, data: Dict[str, Any]) -> bool:
    """
    Save data to a JSON file with error handling.
    
    Args:
        file_path: Path to save the JSON file
        data: Data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except (IOError, TypeError) as e:
        print(f"Error saving JSON file {file_path}: {e}")
        return False


def create_gradient_surface(size: Tuple[int, int], color1: Tuple[int, int, int], 
                          color2: Tuple[int, int, int], direction: str = 'vertical') -> pygame.Surface:
    """
    Create a gradient surface.
    
    Args:
        size: Surface size (width, height)
        color1: Starting color (r, g, b)
        color2: Ending color (r, g, b)
        direction: Gradient direction ('vertical', 'horizontal', 'diagonal')
        
    Returns:
        Pygame surface with gradient
    """
    surface = pygame.Surface(size)
    width, height = size
    
    if direction == 'vertical':
        for y in range(height):
            t = y / height
            color = lerp_color(color1, color2, t)
            pygame.draw.line(surface, color, (0, y), (width, y))
    elif direction == 'horizontal':
        for x in range(width):
            t = x / width
            color = lerp_color(color1, color2, t)
            pygame.draw.line(surface, color, (x, 0), (x, height))
    elif direction == 'diagonal':
        for y in range(height):
            for x in range(width):
                t = (x + y) / (width + height)
                color = lerp_color(color1, color2, t)
                surface.set_at((x, y), color)
    
    return surface


def draw_rounded_rect(surface: pygame.Surface, color: Tuple[int, int, int], 
                     rect: pygame.Rect, radius: int) -> None:
    """
    Draw a rounded rectangle on a surface.
    
    Args:
        surface: Surface to draw on
        color: Rectangle color (r, g, b)
        rect: Rectangle dimensions
        radius: Corner radius
    """
    if radius <= 0:
        pygame.draw.rect(surface, color, rect)
        return
    
    # Clamp radius to prevent overlapping
    radius = min(radius, rect.width // 2, rect.height // 2)
    
    # Draw main rectangles
    pygame.draw.rect(surface, color, (rect.x + radius, rect.y, rect.width - 2 * radius, rect.height))
    pygame.draw.rect(surface, color, (rect.x, rect.y + radius, rect.width, rect.height - 2 * radius))
    
    # Draw corner circles
    pygame.draw.circle(surface, color, (rect.x + radius, rect.y + radius), radius)
    pygame.draw.circle(surface, color, (rect.x + rect.width - radius, rect.y + radius), radius)
    pygame.draw.circle(surface, color, (rect.x + radius, rect.y + rect.height - radius), radius)
    pygame.draw.circle(surface, color, (rect.x + rect.width - radius, rect.y + rect.height - radius), radius)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def ease_in_out(t: float) -> float:
    """
    Ease-in-out animation curve.
    
    Args:
        t: Input value (0.0 to 1.0)
        
    Returns:
        Eased value
    """
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> List[str]:
    """
    Wrap text to fit within a maximum width.
    
    Args:
        text: Text to wrap
        font: Font to use for measuring
        max_width: Maximum width in pixels
        
    Returns:
        List of wrapped text lines
    """
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines


def get_resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource file.
    
    Args:
        relative_path: Relative path to resource
        
    Returns:
        Absolute path to resource
    """
    # Get the directory where the script is located
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def safe_load_image(image_path: str, default_size: Tuple[int, int] = (32, 32)) -> pygame.Surface:
    """
    Safely load an image with fallback to colored rectangle.
    
    Args:
        image_path: Path to image file
        default_size: Size of fallback rectangle
        
    Returns:
        Loaded image surface or colored rectangle
    """
    try:
        return pygame.image.load(image_path)
    except (pygame.error, FileNotFoundError):
        # Create a fallback colored rectangle
        surface = pygame.Surface(default_size)
        surface.fill((255, 0, 255))  # Magenta to indicate missing image
        return surface


def safe_load_sound(sound_path: str) -> Optional[pygame.mixer.Sound]:
    """
    Safely load a sound file with error handling.
    
    Args:
        sound_path: Path to sound file
        
    Returns:
        Loaded sound object or None if failed
    """
    try:
        return pygame.mixer.Sound(sound_path)
    except (pygame.error, FileNotFoundError):
        print(f"Warning: Could not load sound file: {sound_path}")
        return None