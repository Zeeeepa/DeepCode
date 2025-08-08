"""
Game State Management for Snake Game
Handles different game states, transitions, and game flow control.
"""

import pygame
import time
from typing import Optional, Dict, Any, Callable
from enum import Enum

from ..config.settings import GameState, FPS, HIGH_SCORE_FILE
from .snake import Snake
from .food import FoodManager
from .collision import CollisionDetector


class GameStateManager:
    """
    Manages the overall game state and transitions between different states.
    Handles game flow, state persistence, and game loop control.
    """
    
    def __init__(self):
        """Initialize the game state manager."""
        self.current_state = GameState.MENU
        self.previous_state = None
        self.state_start_time = time.time()
        self.state_data = {}
        
        # Game components
        self.snake = None
        self.food_manager = None
        self.collision_detector = None
        
        # Game statistics
        self.score = 0
        self.high_score = self._load_high_score()
        self.level = 1
        self.game_time = 0.0
        self.foods_eaten = 0
        
        # State transition callbacks
        self.state_callbacks = {
            GameState.MENU: self._handle_menu_state,
            GameState.PLAYING: self._handle_playing_state,
            GameState.PAUSED: self._handle_paused_state,
            GameState.GAME_OVER: self._handle_game_over_state,
            GameState.HIGH_SCORE: self._handle_high_score_state
        }
        
        # Game timing
        self.last_move_time = 0
        self.move_interval = 1.0 / 8  # Initial snake speed (8 moves per second)
        self.min_move_interval = 1.0 / 20  # Maximum speed
        
        # Pause functionality
        self.pause_start_time = 0
        self.total_pause_time = 0
        
    def initialize_game_components(self):
        """Initialize game components for a new game."""
        self.snake = Snake()
        self.food_manager = FoodManager()
        self.collision_detector = CollisionDetector()
        
        # Spawn initial food
        self.food_manager.spawn_food(self.snake.get_body_positions())
        
    def change_state(self, new_state: GameState, state_data: Optional[Dict[str, Any]] = None):
        """
        Change to a new game state.
        
        Args:
            new_state: The new state to transition to
            state_data: Optional data to pass to the new state
        """
        if new_state == self.current_state:
            return
            
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()
        
        if state_data:
            self.state_data.update(state_data)
        else:
            self.state_data.clear()
            
        # Handle state entry logic
        self._on_state_enter(new_state)
        
    def _on_state_enter(self, state: GameState):
        """Handle logic when entering a new state."""
        if state == GameState.PLAYING:
            if self.previous_state == GameState.MENU:
                self._start_new_game()
            elif self.previous_state == GameState.PAUSED:
                self._resume_game()
                
        elif state == GameState.PAUSED:
            self.pause_start_time = time.time()
            
        elif state == GameState.GAME_OVER:
            self._end_game()
            
    def _start_new_game(self):
        """Start a new game session."""
        self.score = 0
        self.level = 1
        self.game_time = 0.0
        self.foods_eaten = 0
        self.last_move_time = time.time()
        self.total_pause_time = 0
        
        # Reset move interval to initial speed
        self.move_interval = 1.0 / 8
        
        # Initialize game components
        self.initialize_game_components()
        
    def _resume_game(self):
        """Resume game from pause."""
        if self.pause_start_time > 0:
            pause_duration = time.time() - self.pause_start_time
            self.total_pause_time += pause_duration
            self.pause_start_time = 0
            
    def _end_game(self):
        """Handle game over logic."""
        # Update high score if necessary
        if self.score > self.high_score:
            self.high_score = self.score
            self._save_high_score()
            self.state_data['new_high_score'] = True
        else:
            self.state_data['new_high_score'] = False
            
        # Store final game statistics
        self.state_data.update({
            'final_score': self.score,
            'final_level': self.level,
            'game_time': self.game_time,
            'foods_eaten': self.foods_eaten,
            'snake_length': self.snake.get_length() if self.snake else 0
        })
        
    def update(self, dt: float, events: list):
        """
        Update the current game state.
        
        Args:
            dt: Delta time since last update
            events: List of pygame events
        """
        # Update game time if playing
        if self.current_state == GameState.PLAYING:
            self.game_time += dt
            
        # Handle state-specific updates
        if self.current_state in self.state_callbacks:
            self.state_callbacks[self.current_state](dt, events)
            
    def _handle_menu_state(self, dt: float, events: list):
        """Handle menu state updates."""
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                    self.change_state(GameState.PLAYING)
                elif event.key == pygame.K_h:
                    self.change_state(GameState.HIGH_SCORE)
                elif event.key == pygame.K_ESCAPE:
                    return False  # Signal to quit game
        return True
        
    def _handle_playing_state(self, dt: float, events: list):
        """Handle playing state updates."""
        current_time = time.time()
        
        # Handle input events
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_p:
                    self.change_state(GameState.PAUSED)
                    return True
                    
                # Handle snake direction changes
                direction_map = {
                    pygame.K_UP: (0, -1),
                    pygame.K_DOWN: (0, 1),
                    pygame.K_LEFT: (-1, 0),
                    pygame.K_RIGHT: (1, 0),
                    pygame.K_w: (0, -1),
                    pygame.K_s: (0, 1),
                    pygame.K_a: (-1, 0),
                    pygame.K_d: (1, 0)
                }
                
                if event.key in direction_map:
                    self.snake.set_direction(direction_map[event.key])
                    
        # Update snake movement
        if current_time - self.last_move_time >= self.move_interval:
            if not self.snake.move():
                # Collision detected
                self.change_state(GameState.GAME_OVER)
                return True
                
            self.last_move_time = current_time
            
            # Check food collision
            snake_head = self.snake.get_head_position()
            if self.food_manager.check_collision(snake_head):
                # Food eaten
                food_score = self.food_manager.consume_food()
                self.score += food_score
                self.foods_eaten += 1
                
                # Grow snake
                self.snake.grow()
                
                # Increase difficulty
                self._update_difficulty()
                
                # Spawn new food
                self.food_manager.spawn_food(self.snake.get_body_positions())
                
        # Update food animations
        self.food_manager.update(dt)
        
        return True
        
    def _handle_paused_state(self, dt: float, events: list):
        """Handle paused state updates."""
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_p:
                    self.change_state(GameState.PLAYING)
                elif event.key == pygame.K_q:
                    self.change_state(GameState.MENU)
        return True
        
    def _handle_game_over_state(self, dt: float, events: list):
        """Handle game over state updates."""
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                    self.change_state(GameState.PLAYING)
                elif event.key == pygame.K_ESCAPE:
                    self.change_state(GameState.MENU)
        return True
        
    def _handle_high_score_state(self, dt: float, events: list):
        """Handle high score state updates."""
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                    self.change_state(GameState.MENU)
        return True
        
    def _update_difficulty(self):
        """Update game difficulty based on score."""
        # Increase level every 100 points
        new_level = (self.score // 100) + 1
        if new_level > self.level:
            self.level = new_level
            
            # Increase snake speed (decrease move interval)
            speed_increase = 0.02 * (self.level - 1)
            self.move_interval = max(
                self.min_move_interval,
                1.0 / 8 - speed_increase
            )
            
    def _load_high_score(self) -> int:
        """Load high score from file."""
        try:
            with open(HIGH_SCORE_FILE, 'r') as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            return 0
            
    def _save_high_score(self):
        """Save high score to file."""
        try:
            with open(HIGH_SCORE_FILE, 'w') as f:
                f.write(str(self.high_score))
        except IOError:
            pass  # Silently fail if can't save
            
    def get_current_state(self) -> GameState:
        """Get the current game state."""
        return self.current_state
        
    def get_state_duration(self) -> float:
        """Get how long we've been in the current state."""
        return time.time() - self.state_start_time
        
    def get_game_stats(self) -> Dict[str, Any]:
        """Get current game statistics."""
        return {
            'score': self.score,
            'high_score': self.high_score,
            'level': self.level,
            'game_time': self.game_time,
            'foods_eaten': self.foods_eaten,
            'snake_length': self.snake.get_length() if self.snake else 0,
            'current_speed': 1.0 / self.move_interval if self.move_interval > 0 else 0
        }
        
    def get_state_data(self) -> Dict[str, Any]:
        """Get data associated with the current state."""
        return self.state_data.copy()
        
    def is_game_active(self) -> bool:
        """Check if the game is currently active (playing or paused)."""
        return self.current_state in [GameState.PLAYING, GameState.PAUSED]
        
    def can_pause(self) -> bool:
        """Check if the game can be paused."""
        return self.current_state == GameState.PLAYING
        
    def can_resume(self) -> bool:
        """Check if the game can be resumed."""
        return self.current_state == GameState.PAUSED
        
    def reset_game(self):
        """Reset the game to initial state."""
        self.change_state(GameState.MENU)
        self.score = 0
        self.level = 1
        self.game_time = 0.0
        self.foods_eaten = 0
        self.total_pause_time = 0
        
        # Clean up game components
        self.snake = None
        self.food_manager = None
        self.collision_detector = None


class GameTimer:
    """
    Utility class for managing game timing and frame rate.
    """
    
    def __init__(self, target_fps: int = FPS):
        """
        Initialize the game timer.
        
        Args:
            target_fps: Target frames per second
        """
        self.target_fps = target_fps
        self.clock = pygame.time.Clock()
        self.last_time = time.time()
        self.delta_time = 0.0
        self.frame_count = 0
        self.fps_counter = 0.0
        self.fps_update_time = 0.0
        
    def tick(self) -> float:
        """
        Update the timer and return delta time.
        
        Returns:
            Delta time since last tick in seconds
        """
        current_time = time.time()
        self.delta_time = current_time - self.last_time
        self.last_time = current_time
        
        # Update FPS counter
        self.frame_count += 1
        self.fps_update_time += self.delta_time
        
        if self.fps_update_time >= 1.0:  # Update FPS every second
            self.fps_counter = self.frame_count / self.fps_update_time
            self.frame_count = 0
            self.fps_update_time = 0.0
            
        # Limit frame rate
        self.clock.tick(self.target_fps)
        
        return self.delta_time
        
    def get_fps(self) -> float:
        """Get the current FPS."""
        return self.fps_counter
        
    def get_delta_time(self) -> float:
        """Get the last delta time."""
        return self.delta_time