# Beautiful Snake Game (贪吃蛇)

A modern implementation of the classic Snake game with polished UI, smooth animations, and complete game mechanics.

## Requirements

- Python 3.8+
- Pygame 2.5.2
- Numpy 1.24.0

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On Unix or MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Asset Requirements

Before running the game, you need to place the following assets in their respective directories:

### Fonts (`assets/fonts/`)
- `pixel.ttf` - A pixel-style font for the game UI
  - Recommended: "Press Start 2P" or similar pixel font
  - Size: Regular/Normal weight

### Images (`assets/images/`)
- `snake.png` - Snake body segment texture
  - Size: 32x32 pixels
  - Format: PNG with transparency
- `food.png` - Food item texture
  - Size: 32x32 pixels
  - Format: PNG with transparency
- `background.png` - Game background
  - Size: 800x600 pixels
  - Format: PNG

### Sounds (`assets/sounds/`)
- `eat.wav` - Sound effect when snake eats food
  - Format: WAV
  - Duration: ~0.5 seconds
- `gameover.wav` - Sound effect for game over
  - Format: WAV
  - Duration: ~2 seconds

## How to Play

1. Run the game:
```bash
python main.py
```

2. Controls:
- Arrow keys or WASD: Control snake direction
- Space: Start game / Pause
- Enter: Restart after game over
- Escape: Quit game

## Features

- Smooth snake movement with grid-based mechanics
- Beautiful UI with modern design elements
- Responsive controls
- Score tracking and high score system
- Multiple food types with different effects
- Game over screen with restart option
- Background music and sound effects
- Progressive difficulty (increasing speed)

## Game Mechanics

- Grid-based movement system (20x20 grid)
- 60 FPS game loop
- Collision detection for walls and self
- Score increases based on food type
- Snake grows longer when eating food

## Development

The project structure is organized as follows:

```
snake_game/
├── main.py               # Game entry point and main loop
├── assets/
│   ├── fonts/           # Custom fonts for UI
│   ├── images/          # Game graphics
│   └── sounds/          # Game audio
├── game/
│   ├── __init__.py
│   ├── snake.py         # Snake class and movement logic
│   ├── food.py          # Food generation and collision
│   ├── ui.py            # UI rendering and menus
│   └── settings.py      # Game constants and configuration
├── utils/
│   ├── __init__.py
│   └── helpers.py       # Utility functions
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## License

This project is open source and available under the MIT License.