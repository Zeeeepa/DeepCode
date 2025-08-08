"""
Game settings and constants for the Snake Game
"""
import os
import pygame

# Window Settings
WINDOW_SIZE = 800
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
GRID_SIZE = 20
GRID_DIMENSION = WINDOW_SIZE // GRID_SIZE
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GREEN = (0, 128, 0)
ORANGE = (255, 165, 0)

# Game Settings
INITIAL_SNAKE_LENGTH = 3
INITIAL_SNAKE_SPEED = 10
SPEED_INCREASE_FACTOR = 1.1
MAX_SPEED = 20

# Score Settings
REGULAR_FOOD_SCORE = 10
SPECIAL_FOOD_SCORE = 25
HIGH_SCORE_FILE = "highscore.txt"

# Asset Paths
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
FONT_DIR = os.path.join(ASSET_DIR, "fonts")
IMAGE_DIR = os.path.join(ASSET_DIR, "images")
SOUND_DIR = os.path.join(ASSET_DIR, "sounds")

# Font Settings
FONT_SIZE_LARGE = 74
FONT_SIZE_MEDIUM = 48
FONT_SIZE_SMALL = 24
TITLE_FONT_SIZE = 74
DEFAULT_FONT_PATH = os.path.join(FONT_DIR, "pixel.ttf")
FONT_PATH = DEFAULT_FONT_PATH

# Sound Files
SOUND_EAT = os.path.join(SOUND_DIR, "eat.wav")
SOUND_GAMEOVER = os.path.join(SOUND_DIR, "gameover.wav")

# Direction Vectors
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Game States
STATE_MENU = "menu"
STATE_PLAYING = "playing"
STATE_GAMEOVER = "gameover"
STATE_PAUSED = "paused"

# Additional Game States for compatibility
GAME_STATE_START = "start"
GAME_STATE_PLAYING = "playing"
GAME_STATE_PAUSED = "paused"
GAME_STATE_OVER = "over"

# Control Keys - Using pygame key constants
CONTROLS = {
    "UP": [pygame.K_UP, pygame.K_w],
    "DOWN": [pygame.K_DOWN, pygame.K_s],
    "LEFT": [pygame.K_LEFT, pygame.K_a],
    "RIGHT": [pygame.K_RIGHT, pygame.K_d],
    "PAUSE": [pygame.K_SPACE],
    "QUIT": [pygame.K_ESCAPE],
    "RESTART": [pygame.K_RETURN]
}

# Food Settings
FOOD_TYPES = {
    "regular": {
        "color": RED,
        "score": REGULAR_FOOD_SCORE,
        "probability": 0.8
    },
    "special": {
        "color": YELLOW,
        "score": SPECIAL_FOOD_SCORE,
        "probability": 0.2
    }
}

# Grid and Cell Settings
CELL_SIZE = GRID_SIZE
GRID_WIDTH = GRID_DIMENSION
GRID_HEIGHT = GRID_DIMENSION

# Snake Settings
SNAKE_COLOR = GREEN
SNAKE_HEAD_COLOR = DARK_GREEN

# Food Colors (additional definitions for compatibility)
FOOD_COLOR = RED
SPECIAL_FOOD_COLOR = YELLOW

# UI Settings
BACKGROUND_COLOR = BLACK
TEXT_COLOR = WHITE
UI_TEXT_COLOR = WHITE
BUTTON_COLOR = GRAY
BUTTON_HOVER_COLOR = WHITE
BUTTON_TEXT_COLOR = BLACK

# Game Speed Settings
MIN_SPEED = 5
SPEED_INCREMENT = 1

# Border Settings
BORDER_COLOR = WHITE
BORDER_WIDTH = 2

# Animation Settings
BLINK_SPEED = 500  # milliseconds

# Menu Settings
MENU_TITLE_COLOR = GREEN
MENU_OPTION_COLOR = WHITE
MENU_SELECTED_COLOR = YELLOW

# UI Class for managing user interface elements
class UI:
    def __init__(self, screen):
        self.screen = screen
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.load_fonts()
    
    def load_fonts(self):
        """Load fonts with fallback to default system font"""
        try:
            if os.path.exists(FONT_PATH):
                self.font_large = pygame.font.Font(FONT_PATH, FONT_SIZE_LARGE)
                self.font_medium = pygame.font.Font(FONT_PATH, FONT_SIZE_MEDIUM)
                self.font_small = pygame.font.Font(FONT_PATH, FONT_SIZE_SMALL)
            else:
                raise FileNotFoundError("Custom font not found")
        except:
            # Fallback to default system font
            self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
            self.font_medium = pygame.font.Font(None, FONT_SIZE_MEDIUM)
            self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)
    
    def draw_text(self, text, font_size, color, x, y, center=False):
        """Draw text on screen"""
        if font_size == FONT_SIZE_LARGE:
            font = self.font_large
        elif font_size == FONT_SIZE_MEDIUM:
            font = self.font_medium
        else:
            font = self.font_small
        
        text_surface = font.render(text, True, color)
        if center:
            text_rect = text_surface.get_rect(center=(x, y))
            self.screen.blit(text_surface, text_rect)
        else:
            self.screen.blit(text_surface, (x, y))
    
    def draw_button(self, text, x, y, width, height, color, text_color, hover=False):
        """Draw a button with text"""
        button_color = BUTTON_HOVER_COLOR if hover else color
        pygame.draw.rect(self.screen, button_color, (x, y, width, height))
        pygame.draw.rect(self.screen, BORDER_COLOR, (x, y, width, height), 2)
        
        text_surface = self.font_medium.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(text_surface, text_rect)
