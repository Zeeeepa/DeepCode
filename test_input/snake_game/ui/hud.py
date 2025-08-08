"""
HUD (Heads-Up Display) module for the Snake game.
Handles score display, game statistics, and UI elements during gameplay.
"""

import pygame
import math
from typing import Optional, Tuple, Dict, Any
from ..config.settings import *
from ..config.colors import *


class ScoreDisplay:
    """Handles score display with animations and effects."""
    
    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.current_score = 0
        self.displayed_score = 0
        self.target_score = 0
        self.score_animation_speed = 5.0
        self.score_pulse = 0.0
        self.score_pulse_speed = 8.0
        self.new_score_effect = False
        self.effect_timer = 0.0
        
    def set_score(self, score: int):
        """Set the target score with animation."""
        if score > self.current_score:
            self.new_score_effect = True
            self.effect_timer = 0.5  # Effect duration
        self.current_score = score
        self.target_score = score
        
    def update(self, dt: float):
        """Update score animations."""
        # Animate score counting
        if self.displayed_score < self.target_score:
            diff = self.target_score - self.displayed_score
            self.displayed_score += max(1, int(diff * self.score_animation_speed * dt))
            if self.displayed_score > self.target_score:
                self.displayed_score = self.target_score
        
        # Update pulse animation
        self.score_pulse += self.score_pulse_speed * dt
        
        # Update score effect
        if self.new_score_effect:
            self.effect_timer -= dt
            if self.effect_timer <= 0:
                self.new_score_effect = False
                
    def draw(self, screen: pygame.Surface, x: int, y: int):
        """Draw the score with effects."""
        # Calculate pulse scale
        pulse_scale = 1.0 + 0.1 * math.sin(self.score_pulse)
        
        # Add extra scale for new score effect
        if self.new_score_effect:
            effect_scale = 1.0 + 0.3 * (self.effect_timer / 0.5)
            pulse_scale *= effect_scale
        
        # Create score text
        score_text = f"Score: {self.displayed_score:,}"
        
        # Calculate scaled font size
        scaled_font_size = int(self.font.get_height() * pulse_scale)
        scaled_font = pygame.font.Font(None, scaled_font_size)
        
        # Render text with glow effect
        text_surface = scaled_font.render(score_text, True, SCORE_COLOR)
        glow_surface = scaled_font.render(score_text, True, SCORE_GLOW_COLOR)
        
        # Calculate centered position
        text_rect = text_surface.get_rect()
        text_rect.center = (x, y)
        glow_rect = glow_surface.get_rect()
        glow_rect.center = (x, y)
        
        # Draw glow effect
        for offset in [(2, 2), (-2, 2), (2, -2), (-2, -2), (0, 2), (0, -2), (2, 0), (-2, 0)]:
            glow_pos = (glow_rect.x + offset[0], glow_rect.y + offset[1])
            screen.blit(glow_surface, glow_pos)
        
        # Draw main text
        screen.blit(text_surface, text_rect)


class HighScoreDisplay:
    """Handles high score display."""
    
    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.high_score = 0
        self.is_new_record = False
        self.record_pulse = 0.0
        
    def set_high_score(self, high_score: int, is_new_record: bool = False):
        """Set the high score."""
        self.high_score = high_score
        self.is_new_record = is_new_record
        
    def update(self, dt: float):
        """Update high score animations."""
        if self.is_new_record:
            self.record_pulse += 6.0 * dt
            
    def draw(self, screen: pygame.Surface, x: int, y: int):
        """Draw the high score."""
        high_score_text = f"High Score: {self.high_score:,}"
        
        # Choose color based on record status
        if self.is_new_record:
            color_factor = 0.5 + 0.5 * math.sin(self.record_pulse)
            color = get_gradient_color(HIGH_SCORE_COLOR, SCORE_NEW_RECORD_COLOR, color_factor)
        else:
            color = HIGH_SCORE_COLOR
            
        text_surface = self.font.render(high_score_text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.center = (x, y)
        
        screen.blit(text_surface, text_rect)


class GameTimer:
    """Handles game timer display."""
    
    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.elapsed_time = 0.0
        self.is_running = False
        
    def start(self):
        """Start the timer."""
        self.is_running = True
        
    def stop(self):
        """Stop the timer."""
        self.is_running = False
        
    def reset(self):
        """Reset the timer."""
        self.elapsed_time = 0.0
        self.is_running = False
        
    def update(self, dt: float):
        """Update the timer."""
        if self.is_running:
            self.elapsed_time += dt
            
    def get_time_string(self) -> str:
        """Get formatted time string."""
        minutes = int(self.elapsed_time // 60)
        seconds = int(self.elapsed_time % 60)
        return f"{minutes:02d}:{seconds:02d}"
        
    def draw(self, screen: pygame.Surface, x: int, y: int):
        """Draw the timer."""
        time_text = f"Time: {self.get_time_string()}"
        text_surface = self.font.render(time_text, True, UI_TEXT_COLOR)
        text_rect = text_surface.get_rect()
        text_rect.center = (x, y)
        
        screen.blit(text_surface, text_rect)


class LevelDisplay:
    """Handles level/difficulty display."""
    
    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.level = 1
        self.level_pulse = 0.0
        self.level_changed = False
        self.change_timer = 0.0
        
    def set_level(self, level: int):
        """Set the current level."""
        if level > self.level:
            self.level_changed = True
            self.change_timer = 1.0
        self.level = level
        
    def update(self, dt: float):
        """Update level animations."""
        self.level_pulse += 4.0 * dt
        
        if self.level_changed:
            self.change_timer -= dt
            if self.change_timer <= 0:
                self.level_changed = False
                
    def draw(self, screen: pygame.Surface, x: int, y: int):
        """Draw the level."""
        level_text = f"Level: {self.level}"
        
        # Add pulse effect for level changes
        if self.level_changed:
            pulse_scale = 1.0 + 0.2 * math.sin(self.level_pulse * 2)
            scaled_font_size = int(self.font.get_height() * pulse_scale)
            scaled_font = pygame.font.Font(None, scaled_font_size)
            text_surface = scaled_font.render(level_text, True, SCORE_NEW_RECORD_COLOR)
        else:
            text_surface = self.font.render(level_text, True, UI_TEXT_COLOR)
            
        text_rect = text_surface.get_rect()
        text_rect.center = (x, y)
        
        screen.blit(text_surface, text_rect)


class StatusIndicator:
    """Handles game status indicators (paused, etc.)."""
    
    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.large_font = pygame.font.Font(None, 48)
        self.status_text = ""
        self.status_color = UI_TEXT_COLOR
        self.blink_timer = 0.0
        self.show_status = False
        
    def set_status(self, text: str, color: Tuple[int, int, int] = UI_TEXT_COLOR):
        """Set the status text."""
        self.status_text = text
        self.status_color = color
        self.show_status = True
        self.blink_timer = 0.0
        
    def clear_status(self):
        """Clear the status text."""
        self.show_status = False
        self.status_text = ""
        
    def update(self, dt: float):
        """Update status animations."""
        self.blink_timer += dt
        
    def draw(self, screen: pygame.Surface, x: int, y: int):
        """Draw the status indicator."""
        if not self.show_status or not self.status_text:
            return
            
        # Blink effect
        alpha = int(255 * (0.7 + 0.3 * math.sin(self.blink_timer * 4)))
        
        text_surface = self.large_font.render(self.status_text, True, self.status_color)
        text_surface.set_alpha(alpha)
        text_rect = text_surface.get_rect()
        text_rect.center = (x, y)
        
        # Draw background
        bg_rect = text_rect.inflate(20, 10)
        bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surface.fill((*BACKGROUND_OVERLAY_COLOR, 128))
        screen.blit(bg_surface, bg_rect)
        
        screen.blit(text_surface, text_rect)


class HUD:
    """Main HUD class that manages all UI elements during gameplay."""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()
        
        # Initialize fonts
        try:
            self.score_font = pygame.font.Font(FONT_PATH, SCORE_FONT_SIZE)
            self.ui_font = pygame.font.Font(FONT_PATH, UI_FONT_SIZE)
        except:
            self.score_font = pygame.font.Font(None, SCORE_FONT_SIZE)
            self.ui_font = pygame.font.Font(None, UI_FONT_SIZE)
            
        # Initialize UI components
        self.score_display = ScoreDisplay(self.score_font)
        self.high_score_display = HighScoreDisplay(self.ui_font)
        self.game_timer = GameTimer(self.ui_font)
        self.level_display = LevelDisplay(self.ui_font)
        self.status_indicator = StatusIndicator(self.ui_font)
        
        # Layout settings
        self.margin = 20
        self.top_y = self.margin + 20
        self.bottom_y = self.screen_height - self.margin - 20
        
    def update(self, dt: float):
        """Update all HUD components."""
        self.score_display.update(dt)
        self.high_score_display.update(dt)
        self.game_timer.update(dt)
        self.level_display.update(dt)
        self.status_indicator.update(dt)
        
    def set_score(self, score: int):
        """Set the current score."""
        self.score_display.set_score(score)
        
    def set_high_score(self, high_score: int, is_new_record: bool = False):
        """Set the high score."""
        self.high_score_display.set_high_score(high_score, is_new_record)
        
    def set_level(self, level: int):
        """Set the current level."""
        self.level_display.set_level(level)
        
    def start_timer(self):
        """Start the game timer."""
        self.game_timer.start()
        
    def stop_timer(self):
        """Stop the game timer."""
        self.game_timer.stop()
        
    def reset_timer(self):
        """Reset the game timer."""
        self.game_timer.reset()
        
    def show_status(self, text: str, color: Tuple[int, int, int] = UI_TEXT_COLOR):
        """Show a status message."""
        self.status_indicator.set_status(text, color)
        
    def hide_status(self):
        """Hide the status message."""
        self.status_indicator.clear_status()
        
    def draw(self):
        """Draw all HUD elements."""
        # Top row - Score and High Score
        score_x = self.margin + 100
        high_score_x = self.screen_width - self.margin - 100
        
        self.score_display.draw(self.screen, score_x, self.top_y)
        self.high_score_display.draw(self.screen, high_score_x, self.top_y)
        
        # Bottom row - Timer and Level
        timer_x = self.margin + 80
        level_x = self.screen_width - self.margin - 80
        
        self.game_timer.draw(self.screen, timer_x, self.bottom_y)
        self.level_display.draw(self.screen, level_x, self.bottom_y)
        
        # Center status indicator
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2 - 100
        self.status_indicator.draw(self.screen, center_x, center_y)
        
    def draw_game_info(self, snake_length: int, food_eaten: int):
        """Draw additional game information."""
        # Snake length indicator
        length_text = f"Length: {snake_length}"
        length_surface = self.ui_font.render(length_text, True, UI_TEXT_COLOR)
        length_rect = length_surface.get_rect()
        length_rect.topleft = (self.margin, self.top_y + 40)
        self.screen.blit(length_surface, length_rect)
        
        # Food eaten counter
        food_text = f"Food: {food_eaten}"
        food_surface = self.ui_font.render(food_text, True, UI_TEXT_COLOR)
        food_rect = food_surface.get_rect()
        food_rect.topright = (self.screen_width - self.margin, self.top_y + 40)
        self.screen.blit(food_surface, food_rect)
        
    def draw_fps(self, fps: float):
        """Draw FPS counter (for debugging)."""
        if not hasattr(self, 'show_fps') or not self.show_fps:
            return
            
        fps_text = f"FPS: {fps:.1f}"
        fps_surface = self.ui_font.render(fps_text, True, UI_TEXT_COLOR)
        fps_rect = fps_surface.get_rect()
        fps_rect.bottomright = (self.screen_width - self.margin, self.screen_height - self.margin)
        self.screen.blit(fps_surface, fps_rect)
        
    def toggle_fps_display(self):
        """Toggle FPS display."""
        self.show_fps = getattr(self, 'show_fps', False)
        self.show_fps = not self.show_fps