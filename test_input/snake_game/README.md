# Snake Game (贪吃蛇) 🐍

A beautiful and feature-rich implementation of the classic Snake game with modern graphics, smooth animations, and immersive sound effects.

![Snake Game](https://img.shields.io/badge/Game-Snake-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Pygame](https://img.shields.io/badge/Pygame-2.5.2-red)

## 🎮 Features

### Core Gameplay
- **Smooth Snake Movement**: Responsive controls with WASD or Arrow keys
- **Dynamic Food Spawning**: Random food generation with collision detection
- **Progressive Difficulty**: Game speed increases as snake grows longer
- **Collision Detection**: Border and self-collision detection with game over

### Visual Experience
- **Beautiful Graphics**: Gradient backgrounds and smooth animations
- **Modern UI Design**: Polished start menu, pause screen, and game over screen
- **Responsive HUD**: Real-time score display and game statistics
- **Visual Effects**: Fade transitions and pulse animations

### Audio Experience
- **Sound Effects**: Eating sounds and game over audio feedback
- **Generated Audio**: Procedurally generated sound effects using pygame
- **Volume Control**: Configurable sound settings

### Game Features
- **High Score System**: Persistent high score tracking
- **Game States**: Start menu, playing, paused, and game over states
- **Save System**: Automatic saving of game progress and settings
- **Configurable Settings**: Customizable game parameters

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**:
   ```bash
   # If you have the project files, navigate to the snake_game directory
   cd snake_game
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv snake_env
   
   # Activate the virtual environment
   # On Windows:
   snake_env\Scripts\activate
   # On macOS/Linux:
   source snake_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate sound effects** (optional):
   ```bash
   python assets/sounds/generate_sounds.py
   ```

5. **Run the game**:
   ```bash
   python main.py
   ```

## 🎯 How to Play

### Controls
- **Movement**: Use WASD keys or Arrow keys to control the snake
  - `W` or `↑`: Move Up
  - `S` or `↓`: Move Down
  - `A` or `←`: Move Left
  - `D` or `→`: Move Right

### Game Controls
- **Space**: Start game from menu or pause/unpause during gameplay
- **Enter**: Confirm menu selections
- **Escape**: Return to menu or quit game

### Objective
1. Control the snake to eat food (red squares)
2. Each food eaten increases your score and snake length
3. Avoid hitting the walls or the snake's own body
4. Try to achieve the highest score possible!

### Scoring
- **Food eaten**: +10 points per food item
- **High Score**: Your best score is automatically saved
- **Progressive Speed**: Game becomes faster as snake grows

## 🏗️ Project Structure

```
snake_game/
├── main.py                 # Entry point and main game loop
├── game/                   # Core game logic
│   ├── __init__.py
│   ├── snake.py           # Snake class and movement logic
│   ├── food.py            # Food generation and management
│   ├── game_state.py      # Game state management
│   └── collision.py       # Collision detection logic
├── ui/                     # User interface components
│   ├── __init__.py
│   ├── renderer.py        # Game rendering and graphics
│   ├── menu.py            # Start menu and game over screen
│   └── hud.py             # Score display and UI elements
├── assets/                 # Game assets
│   └── sounds/            # Sound effects
│       └── generate_sounds.py  # Sound generation script
├── config/                 # Configuration files
│   ├── __init__.py
│   ├── settings.py        # Game configuration
│   └── colors.py          # Color scheme definitions
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── helpers.py         # Helper functions
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## ⚙️ Configuration

The game can be customized by modifying the settings in `config/settings.py`:

### Screen Settings
- `SCREEN_WIDTH`: Game window width (default: 800)
- `SCREEN_HEIGHT`: Game window height (default: 600)
- `FPS`: Frames per second (default: 60)

### Game Settings
- `INITIAL_SNAKE_LENGTH`: Starting snake length (default: 3)
- `SNAKE_SPEED`: Base movement speed (default: 5)
- `FOOD_SPAWN_MARGIN`: Border margin for food spawning (default: 20)

### Visual Settings
- Color schemes in `config/colors.py`
- Animation speeds and effects
- UI font sizes and styling

### Audio Settings
- `SOUND_ENABLED`: Enable/disable sound effects
- `SOUND_VOLUME`: Master volume control (0.0 to 1.0)

## 🔧 Technical Details

### Dependencies
- **pygame==2.5.2**: Game development framework
- **numpy==1.24.3**: Numerical computations for game logic

### Architecture
- **Modular Design**: Separated concerns for maintainability
- **Event-Driven**: Pygame event system for user input
- **State Management**: Clean game state transitions
- **Object-Oriented**: Well-structured classes for game entities

### Performance
- **Optimized Rendering**: Efficient drawing and update cycles
- **Memory Management**: Proper resource cleanup
- **Smooth Animation**: Consistent frame rate and timing

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Make sure you're in the correct directory and virtual environment is activated
   pip install -r requirements.txt
   ```

2. **Sound Issues**:
   ```bash
   # Generate sound files if missing
   python assets/sounds/generate_sounds.py
   ```

3. **Performance Issues**:
   - Lower the FPS in `config/settings.py`
   - Disable sound effects if needed
   - Close other applications to free up resources

4. **Display Issues**:
   - Check your display resolution
   - Modify screen dimensions in settings if needed

### Error Messages
- **"pygame not found"**: Install pygame with `pip install pygame==2.5.2`
- **"No module named 'numpy'"**: Install numpy with `pip install numpy==1.24.3`
- **"Sound file not found"**: Run the sound generation script

## 🎨 Customization

### Adding New Features
1. **New Game Modes**: Extend `game_state.py` with additional states
2. **Power-ups**: Modify `food.py` to include special food types
3. **Themes**: Create new color schemes in `config/colors.py`
4. **Sound Effects**: Add new sounds in `assets/sounds/`

### Modifying Gameplay
- **Snake Speed**: Adjust `SNAKE_SPEED` in settings
- **Grid Size**: Modify `GRID_SIZE` for different gameplay feel
- **Scoring System**: Update scoring logic in game components

## 📝 Development

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Document functions and classes
- Keep functions focused and modular

### Testing
```bash
# Run the game to test functionality
python main.py

# Test individual components
python -c "from game.snake import Snake; print('Snake module works!')"
python -c "from ui.renderer import Renderer; print('Renderer module works!')"
```

## 🤝 Contributing

1. Fork the project
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with Python and Pygame
- Inspired by the classic Snake game
- Sound effects generated using pygame's audio synthesis
- Modern UI design principles applied to retro gaming

---

**Enjoy playing Snake! 🐍🎮**

*For support or questions, please check the troubleshooting section or review the code documentation.*