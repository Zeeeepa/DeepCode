# Game Configuration Settings
"""
Snake Game Configuration
Contains all game settings and constants
"""

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# Game settings
FPS = 10  # Frames per second (game speed)
INITIAL_SNAKE_LENGTH = 3
SCORE_PER_FOOD = 10
SPEED_INCREASE_INTERVAL = 5  # Increase speed every 5 foods eaten
MAX_FPS = 20  # Maximum game speed

# Snake settings
SNAKE_INITIAL_POSITION = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
SNAKE_INITIAL_DIRECTION = (1, 0)  # Moving right initially

# Food settings
FOOD_SPAWN_MARGIN = 1  # Minimum distance from borders

# Sound settings
SOUND_ENABLED = True
SOUND_VOLUME = 0.7

# File paths
HIGH_SCORE_FILE = "high_score.json"
FONT_PATH = "assets/fonts/game_font.ttf"
SOUND_EAT_PATH = "assets/sounds/eat.wav"
SOUND_GAME_OVER_PATH = "assets/sounds/game_over.wav"

# UI settings
MENU_FONT_SIZE = 48
SCORE_FONT_SIZE = 24
BUTTON_FONT_SIZE = 32
UI_MARGIN = 20

# Animation settings
FADE_SPEED = 5
PULSE_SPEED = 0.1

# Game states
class GameState:
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    HIGH_SCORE = "high_score"

# Controls
CONTROLS = {
    'up': ['w', 'up'],
    'down': ['s', 'down'],
    'left': ['a', 'left'],
    'right': ['d', 'right'],
    'pause': ['space', 'p'],
    'restart': ['r'],
    'quit': ['escape', 'q']
}