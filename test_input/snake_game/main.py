#!/usr/bin/env python3
"""
Snake Game (贪吃蛇) - Main Entry Point
A beautiful and complete Snake game implementation with modern graphics and sound effects.

Author: Snake Game Development Team
Version: 1.0.0
"""

import pygame
import sys
import os
import traceback
from typing import Optional

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import game modules
from config.settings import *
from config.colors import *
from game.game_state import GameStateManager
from ui.renderer import GameRenderer
from ui.menu import MenuManager
from ui.hud import HUD
from utils.helpers import load_high_score, save_high_score


class SnakeGame:
    """Main game class that manages the entire Snake game."""
    
    def __init__(self):
        """Initialize the Snake game."""
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.running = False
        self.game_state_manager: Optional[GameStateManager] = None
        self.renderer: Optional[GameRenderer] = None
        self.menu_manager: Optional[MenuManager] = None
        self.hud: Optional[HUD] = None
        self.high_score = 0
        self.sound_enabled = SOUND_ENABLED
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = 0.0
        self.current_fps = 0.0
        
        # Initialize pygame and game components
        self._initialize_pygame()
        self._initialize_game_components()
        self._load_game_data()
    
    def _initialize_pygame(self):
        """Initialize pygame and create the main window."""
        try:
            pygame.init()
            
            # Initialize sound system
            if self.sound_enabled:
                try:
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                    pygame.mixer.set_num_channels(8)
                except pygame.error as e:
                    print(f"Warning: Could not initialize sound system: {e}")
                    self.sound_enabled = False
            
            # Create the main window
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake Game (贪吃蛇)")
            
            # Set window icon (if available)
            try:
                icon_path = os.path.join("assets", "icon.png")
                if os.path.exists(icon_path):
                    icon = pygame.image.load(icon_path)
                    pygame.display.set_icon(icon)
            except pygame.error:
                pass  # Icon not available, continue without it
            
            # Create clock for frame rate control
            self.clock = pygame.time.Clock()
            
            print("Pygame initialized successfully")
            
        except Exception as e:
            print(f"Error initializing pygame: {e}")
            sys.exit(1)
    
    def _initialize_game_components(self):
        """Initialize all game components."""
        try:
            # Initialize game state manager
            self.game_state_manager = GameStateManager()
            
            # Initialize renderer
            self.renderer = GameRenderer(self.screen)
            
            # Initialize menu manager
            self.menu_manager = MenuManager(self.screen)
            
            # Initialize HUD
            self.hud = HUD(self.screen)
            
            print("Game components initialized successfully")
            
        except Exception as e:
            print(f"Error initializing game components: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def _load_game_data(self):
        """Load saved game data like high scores."""
        try:
            self.high_score = load_high_score()
            self.hud.set_high_score(self.high_score)
            print(f"High score loaded: {self.high_score}")
        except Exception as e:
            print(f"Warning: Could not load high score: {e}")
            self.high_score = 0
    
    def _save_game_data(self):
        """Save game data like high scores."""
        try:
            save_high_score(self.high_score)
            print(f"High score saved: {self.high_score}")
        except Exception as e:
            print(f"Warning: Could not save high score: {e}")
    
    def _handle_events(self):
        """Handle pygame events."""
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                return
            
            elif event.type == pygame.KEYDOWN:
                # Global key handlers
                if event.key == pygame.K_F11:
                    # Toggle fullscreen (if supported)
                    try:
                        pygame.display.toggle_fullscreen()
                    except:
                        pass
                
                elif event.key == pygame.K_F1:
                    # Toggle FPS display
                    self.hud.toggle_fps_display()
                
                elif event.key == pygame.K_m:
                    # Toggle sound
                    self.sound_enabled = not self.sound_enabled
                    if hasattr(self.game_state_manager, 'sound_enabled'):
                        self.game_state_manager.sound_enabled = self.sound_enabled
                
                # Handle state-specific events
                current_state = self.game_state_manager.get_current_state()
                
                if current_state == GameState.MENU:
                    self._handle_menu_events(event)
                elif current_state == GameState.PLAYING:
                    self._handle_game_events(event)
                elif current_state == GameState.PAUSED:
                    self._handle_pause_events(event)
                elif current_state == GameState.GAME_OVER:
                    self._handle_game_over_events(event)
        
        # Update game state with events
        self.game_state_manager.update(self.clock.get_time() / 1000.0, events)
    
    def _handle_menu_events(self, event):
        """Handle events in menu state."""
        if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
            # Start new game
            self.game_state_manager.change_state(GameState.PLAYING)
            self.hud.start_timer()
            self.hud.show_status("Game Started!", UI_SUCCESS_COLOR)
        
        elif event.key == pygame.K_ESCAPE:
            # Quit game
            self.running = False
    
    def _handle_game_events(self, event):
        """Handle events during gameplay."""
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_p:
            # Pause game
            self.game_state_manager.change_state(GameState.PAUSED)
            self.hud.stop_timer()
            self.hud.show_status("Game Paused", UI_WARNING_COLOR)
        
        # Movement controls are handled by the game state manager
    
    def _handle_pause_events(self, event):
        """Handle events in pause state."""
        if event.key == pygame.K_SPACE or event.key == pygame.K_p:
            # Resume game
            self.game_state_manager.change_state(GameState.PLAYING)
            self.hud.start_timer()
            self.hud.hide_status()
        
        elif event.key == pygame.K_ESCAPE:
            # Return to menu
            self.game_state_manager.change_state(GameState.MENU)
            self.hud.reset_timer()
            self.hud.hide_status()
    
    def _handle_game_over_events(self, event):
        """Handle events in game over state."""
        if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
            # Start new game
            self.game_state_manager.reset_game()
            self.game_state_manager.change_state(GameState.PLAYING)
            self.hud.start_timer()
            self.hud.hide_status()
        
        elif event.key == pygame.K_ESCAPE:
            # Return to menu
            self.game_state_manager.change_state(GameState.MENU)
            self.hud.reset_timer()
            self.hud.hide_status()
    
    def _update_game(self, dt: float):
        """Update game logic."""
        # Update renderer animations
        self.renderer.update_animation(dt)
        
        # Update FPS counter
        self.fps_timer += dt
        self.fps_counter += 1
        
        if self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / self.fps_timer
            self.fps_counter = 0
            self.fps_timer = 0.0
        
        # Update game statistics
        game_stats = self.game_state_manager.get_game_stats()
        
        # Update HUD
        self.hud.set_score(game_stats.get('score', 0))
        self.hud.set_level(game_stats.get('level', 1))
        
        # Check for new high score
        current_score = game_stats.get('score', 0)
        if current_score > self.high_score:
            self.high_score = current_score
            self.hud.set_high_score(self.high_score, is_new_record=True)
        
        # Handle game state changes
        current_state = self.game_state_manager.get_current_state()
        
        if current_state == GameState.GAME_OVER:
            self.hud.stop_timer()
            if current_score > 0:
                self.hud.show_status(f"Game Over! Final Score: {current_score}", UI_ERROR_COLOR)
            else:
                self.hud.show_status("Game Over!", UI_ERROR_COLOR)
    
    def _render_game(self):
        """Render the game."""
        current_state = self.game_state_manager.get_current_state()
        
        # Clear screen
        self.renderer.clear_screen()
        
        if current_state == GameState.MENU:
            self._render_menu()
        
        elif current_state in [GameState.PLAYING, GameState.PAUSED, GameState.GAME_OVER]:
            self._render_gameplay()
            
            # Draw overlays for non-playing states
            if current_state == GameState.PAUSED:
                self.renderer.draw_pause_overlay()
            elif current_state == GameState.GAME_OVER:
                self.renderer.draw_game_over_overlay()
        
        # Draw HUD (always visible except in menu)
        if current_state != GameState.MENU:
            self.hud.draw()
            
            # Draw additional game info
            game_stats = self.game_state_manager.get_game_stats()
            snake_length = game_stats.get('snake_length', 1)
            food_eaten = game_stats.get('food_eaten', 0)
            self.hud.draw_game_info(snake_length, food_eaten)
            
            # Draw FPS counter if enabled
            self.hud.draw_fps(self.current_fps)
        
        # Update display
        pygame.display.flip()
    
    def _render_menu(self):
        """Render the main menu."""
        self.menu_manager.draw_main_menu()
    
    def _render_gameplay(self):
        """Render the gameplay screen."""
        # Draw grid
        self.renderer.draw_grid()
        
        # Draw border
        self.renderer.draw_border()
        
        # Get game objects from state manager
        game_stats = self.game_state_manager.get_game_stats()
        
        # Draw snake
        if 'snake_segments' in game_stats and 'snake_direction' in game_stats:
            self.renderer.draw_snake(
                game_stats['snake_segments'],
                game_stats['snake_direction']
            )
        
        # Draw food
        if 'food_position' in game_stats:
            food_type = game_stats.get('food_type', 'normal')
            self.renderer.draw_food(game_stats['food_position'], food_type)
        
        # Draw particle effects if any
        if 'particles' in game_stats:
            self.renderer.draw_particle_effect(game_stats['particles'])
    
    def run(self):
        """Main game loop."""
        self.running = True
        
        print("Starting Snake Game...")
        print("Controls:")
        print("  WASD or Arrow Keys - Move snake")
        print("  SPACE/ENTER - Start game / Restart")
        print("  P/ESC - Pause / Resume")
        print("  F1 - Toggle FPS display")
        print("  M - Toggle sound")
        print("  F11 - Toggle fullscreen")
        print("  ESC - Quit (from menu)")
        
        try:
            while self.running:
                # Calculate delta time
                dt = self.clock.tick(FPS) / 1000.0
                
                # Handle events
                self._handle_events()
                
                if not self.running:
                    break
                
                # Update game
                self._update_game(dt)
                
                # Render game
                self._render_game()
            
        except KeyboardInterrupt:
            print("\nGame interrupted by user")
        
        except Exception as e:
            print(f"Unexpected error during game loop: {e}")
            traceback.print_exc()
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources before exiting."""
        print("Cleaning up...")
        
        # Save game data
        self._save_game_data()
        
        # Quit pygame
        try:
            pygame.mixer.quit()
        except:
            pass
        
        try:
            pygame.quit()
        except:
            pass
        
        print("Game cleanup completed")


def main():
    """Main entry point for the Snake game."""
    print("=" * 50)
    print("🐍 Snake Game (贪吃蛇) 🐍")
    print("=" * 50)
    
    try:
        # Create and run the game
        game = SnakeGame()
        game.run()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        return 1
    
    print("Thanks for playing!")
    return 0


if __name__ == "__main__":
    sys.exit(main())