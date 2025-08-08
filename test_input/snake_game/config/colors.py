"""
Color scheme definitions for the Snake game.
Provides a beautiful and consistent color palette throughout the game.
"""

# Background colors
BACKGROUND_COLOR = (20, 25, 40)  # Dark blue-gray
BACKGROUND_GRADIENT_START = (20, 25, 40)
BACKGROUND_GRADIENT_END = (40, 45, 60)

# Snake colors
SNAKE_HEAD_COLOR = (50, 205, 50)  # Lime green
SNAKE_BODY_COLOR = (34, 139, 34)  # Forest green
SNAKE_BODY_GRADIENT = (46, 125, 50)  # Medium green
SNAKE_OUTLINE_COLOR = (25, 100, 25)  # Dark green outline

# Snake color aliases for compatibility
SNAKE_COLOR = SNAKE_HEAD_COLOR  # Main snake color alias
SNAKE_HEAD = SNAKE_HEAD_COLOR
SNAKE_BODY = SNAKE_BODY_COLOR
SNAKE_TAIL = SNAKE_BODY_COLOR  # Tail uses same color as body

# Food colors
FOOD_COLOR = (255, 69, 0)  # Red orange
FOOD_OUTLINE_COLOR = (200, 50, 0)  # Dark red
FOOD_GLOW_COLOR = (255, 100, 50)  # Light orange glow

# UI colors
TEXT_COLOR = (255, 255, 255)  # White
TEXT_SHADOW_COLOR = (0, 0, 0)  # Black
MENU_TEXT_COLOR = (220, 220, 220)  # Light gray
MENU_HIGHLIGHT_COLOR = (100, 200, 100)  # Light green

# UI status colors
UI_SUCCESS_COLOR = (50, 205, 50)  # Lime green
UI_WARNING_COLOR = (255, 165, 0)  # Orange
UI_ERROR_COLOR = (255, 69, 0)  # Red orange
UI_INFO_COLOR = (135, 206, 235)  # Sky blue

# Score and HUD colors
SCORE_COLOR = (255, 255, 255)  # White
HIGH_SCORE_COLOR = (255, 215, 0)  # Gold
SCORE_BACKGROUND = (0, 0, 0, 128)  # Semi-transparent black

# Game state colors
PAUSE_OVERLAY_COLOR = (0, 0, 0, 180)  # Semi-transparent black
GAME_OVER_OVERLAY_COLOR = (139, 0, 0, 200)  # Semi-transparent dark red
MENU_BACKGROUND_COLOR = (30, 35, 50)  # Dark blue

# Border and grid colors
BORDER_COLOR = (100, 100, 100)  # Gray
GRID_COLOR = (40, 45, 60)  # Subtle grid lines
GRID_ALPHA = 50  # Grid transparency

# Button colors
BUTTON_COLOR = (70, 130, 180)  # Steel blue
BUTTON_HOVER_COLOR = (100, 149, 237)  # Cornflower blue
BUTTON_PRESSED_COLOR = (65, 105, 225)  # Royal blue
BUTTON_TEXT_COLOR = (255, 255, 255)  # White
BUTTON_BORDER_COLOR = (47, 79, 79)  # Dark slate gray

# Animation colors
FADE_COLOR = (0, 0, 0)  # Black for fade effects
PULSE_COLOR_MIN = (255, 255, 255, 100)  # Semi-transparent white
PULSE_COLOR_MAX = (255, 255, 255, 255)  # Opaque white

# Special effect colors
PARTICLE_COLORS = [
    (255, 69, 0),   # Red orange
    (255, 140, 0),  # Dark orange
    (255, 215, 0),  # Gold
    (255, 255, 0),  # Yellow
]

# Color utility functions
def get_gradient_color(start_color, end_color, factor):
    """
    Calculate a color between start_color and end_color based on factor (0.0 to 1.0).
    
    Args:
        start_color: RGB tuple for starting color
        end_color: RGB tuple for ending color
        factor: Float between 0.0 and 1.0
    
    Returns:
        RGB tuple of interpolated color
    """
    if factor < 0:
        factor = 0
    elif factor > 1:
        factor = 1
    
    r = int(start_color[0] + (end_color[0] - start_color[0]) * factor)
    g = int(start_color[1] + (end_color[1] - start_color[1]) * factor)
    b = int(start_color[2] + (end_color[2] - start_color[2]) * factor)
    
    return (r, g, b)

def add_alpha(color, alpha):
    """
    Add alpha channel to RGB color.
    
    Args:
        color: RGB tuple
        alpha: Alpha value (0-255)
    
    Returns:
        RGBA tuple
    """
    return (*color, alpha)

def darken_color(color, factor=0.7):
    """
    Darken a color by multiplying each component by factor.
    
    Args:
        color: RGB tuple
        factor: Darkening factor (0.0 to 1.0)
    
    Returns:
        RGB tuple of darkened color
    """
    return tuple(int(c * factor) for c in color)

def lighten_color(color, factor=1.3):
    """
    Lighten a color by multiplying each component by factor.
    
    Args:
        color: RGB tuple
        factor: Lightening factor (>= 1.0)
    
    Returns:
        RGB tuple of lightened color
    """
    return tuple(min(255, int(c * factor)) for c in color)