"""
Menu system for the Snake game.
Handles start menu, game over screen, and pause menu.
"""

import pygame
import math
from typing import List, Tuple, Optional, Dict, Any
from ..config.settings import *
from ..config.colors import *


class MenuItem:
    """Represents a single menu item with text and action."""
    
    def __init__(self, text: str, action: str, font_size: int = MENU_FONT_SIZE):
        self.text = text
        self.action = action
        self.font_size = font_size
        self.selected = False
        self.hover_scale = 1.0
        self.target_scale = 1.0
        
    def update(self, dt: float):
        """Update menu item animations."""
        # Smooth scaling animation
        scale_diff = self.target_scale - self.hover_scale
        self.hover_scale += scale_diff * 8.0 * dt
        
    def set_selected(self, selected: bool):
        """Set selection state and update target scale."""
        self.selected = selected
        self.target_scale = 1.2 if selected else 1.0


class Menu:
    """Base menu class with common functionality."""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.items: List[MenuItem] = []
        self.selected_index = 0
        self.title = ""
        self.subtitle = ""
        
        # Animation properties
        self.fade_alpha = 0
        self.target_alpha = 255
        self.title_offset = 0
        self.background_alpha = 0
        
        # Load fonts
        try:
            self.title_font = pygame.font.Font(FONT_PATH, TITLE_FONT_SIZE)
            self.menu_font = pygame.font.Font(FONT_PATH, MENU_FONT_SIZE)
            self.subtitle_font = pygame.font.Font(FONT_PATH, SUBTITLE_FONT_SIZE)
        except:
            # Fallback to system fonts
            self.title_font = pygame.font.Font(None, TITLE_FONT_SIZE)
            self.menu_font = pygame.font.Font(None, MENU_FONT_SIZE)
            self.subtitle_font = pygame.font.Font(None, SUBTITLE_FONT_SIZE)
    
    def add_item(self, text: str, action: str, font_size: int = MENU_FONT_SIZE):
        """Add a menu item."""
        item = MenuItem(text, action, font_size)
        self.items.append(item)
        self._update_selection()
    
    def handle_input(self, events: List[pygame.event.Event]) -> Optional[str]:
        """Handle menu input and return selected action."""
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_w]:
                    self.selected_index = (self.selected_index - 1) % len(self.items)
                    self._update_selection()
                    return "navigate"
                elif event.key in [pygame.K_DOWN, pygame.K_s]:
                    self.selected_index = (self.selected_index + 1) % len(self.items)
                    self._update_selection()
                    return "navigate"
                elif event.key in [pygame.K_RETURN, pygame.K_SPACE]:
                    if self.items:
                        return self.items[self.selected_index].action
        return None
    
    def _update_selection(self):
        """Update selection state of menu items."""
        for i, item in enumerate(self.items):
            item.set_selected(i == self.selected_index)
    
    def update(self, dt: float):
        """Update menu animations."""
        # Update fade animation
        alpha_diff = self.target_alpha - self.fade_alpha
        self.fade_alpha += alpha_diff * 5.0 * dt
        
        # Update background alpha
        self.background_alpha = min(128, self.fade_alpha * 0.5)
        
        # Update title animation
        self.title_offset = math.sin(pygame.time.get_ticks() * 0.002) * 5
        
        # Update menu items
        for item in self.items:
            item.update(dt)
    
    def draw_background(self):
        """Draw menu background with gradient and overlay."""
        # Create gradient background
        for y in range(SCREEN_HEIGHT):
            color_ratio = y / SCREEN_HEIGHT
            r = int(DARK_BLUE[0] * (1 - color_ratio) + BLACK[0] * color_ratio)
            g = int(DARK_BLUE[1] * (1 - color_ratio) + BLACK[1] * color_ratio)
            b = int(DARK_BLUE[2] * (1 - color_ratio) + BLACK[2] * color_ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))
        
        # Add semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(int(self.background_alpha))
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
    
    def draw_title(self, title: str, y_offset: int = 0):
        """Draw the menu title with glow effect."""
        if not title:
            return
            
        # Create title surface with glow
        title_surface = self.title_font.render(title, True, WHITE)
        title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2, 150 + y_offset + self.title_offset))
        
        # Draw glow effect
        glow_surface = self.title_font.render(title, True, LIGHT_GREEN)
        for offset in [(2, 2), (-2, 2), (2, -2), (-2, -2), (0, 3), (0, -3), (3, 0), (-3, 0)]:
            glow_rect = title_rect.copy()
            glow_rect.x += offset[0]
            glow_rect.y += offset[1]
            glow_surface.set_alpha(30)
            self.screen.blit(glow_surface, glow_rect)
        
        # Draw main title
        title_surface.set_alpha(int(self.fade_alpha))
        self.screen.blit(title_surface, title_rect)
    
    def draw_subtitle(self, subtitle: str, y_offset: int = 0):
        """Draw subtitle text."""
        if not subtitle:
            return
            
        subtitle_surface = self.subtitle_font.render(subtitle, True, LIGHT_GRAY)
        subtitle_rect = subtitle_surface.get_rect(center=(SCREEN_WIDTH // 2, 200 + y_offset))
        subtitle_surface.set_alpha(int(self.fade_alpha * 0.8))
        self.screen.blit(subtitle_surface, subtitle_rect)
    
    def draw_items(self, start_y: int = 300):
        """Draw menu items with animations."""
        for i, item in enumerate(self.items):
            y_pos = start_y + i * 60
            
            # Get font for this item
            font = pygame.font.Font(None, item.font_size) if item.font_size != MENU_FONT_SIZE else self.menu_font
            
            # Create text surface
            color = LIGHT_GREEN if item.selected else WHITE
            text_surface = font.render(item.text, True, color)
            
            # Apply scaling
            if item.hover_scale != 1.0:
                scaled_width = int(text_surface.get_width() * item.hover_scale)
                scaled_height = int(text_surface.get_height() * item.hover_scale)
                text_surface = pygame.transform.scale(text_surface, (scaled_width, scaled_height))
            
            # Position and draw
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, y_pos))
            
            # Draw selection indicator
            if item.selected:
                indicator_size = 10
                pygame.draw.circle(self.screen, LIGHT_GREEN, 
                                 (text_rect.left - 30, text_rect.centery), indicator_size)
                pygame.draw.circle(self.screen, LIGHT_GREEN, 
                                 (text_rect.right + 30, text_rect.centery), indicator_size)
            
            # Apply fade alpha
            text_surface.set_alpha(int(self.fade_alpha))
            self.screen.blit(text_surface, text_rect)


class MainMenu(Menu):
    """Main menu screen."""
    
    def __init__(self, screen: pygame.Surface):
        super().__init__(screen)
        self.title = "SNAKE GAME"
        self.subtitle = "贪吃蛇"
        
        # Add menu items
        self.add_item("Start Game", "start_game")
        self.add_item("High Scores", "high_scores")
        self.add_item("Settings", "settings")
        self.add_item("Quit", "quit")
        
        # Animation properties
        self.snake_animation_time = 0
        self.food_animation_time = 0
    
    def update(self, dt: float):
        """Update main menu animations."""
        super().update(dt)
        self.snake_animation_time += dt
        self.food_animation_time += dt * 2
    
    def draw(self):
        """Draw the main menu."""
        self.draw_background()
        self.draw_decorative_elements()
        self.draw_title(self.title)
        self.draw_subtitle(self.subtitle)
        self.draw_items()
        self.draw_instructions()
    
    def draw_decorative_elements(self):
        """Draw decorative snake and food elements."""
        # Animated snake segments
        for i in range(5):
            x = 100 + i * 30 + math.sin(self.snake_animation_time + i * 0.5) * 10
            y = 400 + math.cos(self.snake_animation_time + i * 0.3) * 20
            color = LIGHT_GREEN if i == 0 else GREEN
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 12)
        
        # Animated food
        food_x = SCREEN_WIDTH - 150
        food_y = 400 + math.sin(self.food_animation_time) * 15
        food_size = 10 + math.sin(self.food_animation_time * 2) * 3
        pygame.draw.circle(self.screen, RED, (int(food_x), int(food_y)), int(food_size))
    
    def draw_instructions(self):
        """Draw control instructions."""
        instructions = [
            "Use WASD or Arrow Keys to move",
            "Press SPACE to pause",
            "Press ESC to return to menu"
        ]
        
        font = pygame.font.Font(None, 24)
        y_start = SCREEN_HEIGHT - 120
        
        for i, instruction in enumerate(instructions):
            text_surface = font.render(instruction, True, LIGHT_GRAY)
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, y_start + i * 25))
            text_surface.set_alpha(int(self.fade_alpha * 0.6))
            self.screen.blit(text_surface, text_rect)


class GameOverMenu(Menu):
    """Game over screen."""
    
    def __init__(self, screen: pygame.Surface):
        super().__init__(screen)
        self.title = "GAME OVER"
        self.score = 0
        self.high_score = 0
        self.is_new_high_score = False
        
        # Add menu items
        self.add_item("Play Again", "restart")
        self.add_item("Main Menu", "main_menu")
        self.add_item("Quit", "quit")
    
    def set_score_data(self, score: int, high_score: int, is_new_high_score: bool = False):
        """Set score information for display."""
        self.score = score
        self.high_score = high_score
        self.is_new_high_score = is_new_high_score
    
    def draw(self):
        """Draw the game over menu."""
        self.draw_background()
        self.draw_title(self.title)
        self.draw_score_info()
        self.draw_items(400)
    
    def draw_score_info(self):
        """Draw score information."""
        y_pos = 220
        
        # Current score
        score_text = f"Score: {self.score}"
        score_color = LIGHT_GREEN if self.is_new_high_score else WHITE
        score_surface = self.subtitle_font.render(score_text, True, score_color)
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, y_pos))
        
        if self.is_new_high_score:
            # Add "NEW HIGH SCORE!" text
            new_high_text = "NEW HIGH SCORE!"
            new_high_surface = self.menu_font.render(new_high_text, True, YELLOW)
            new_high_rect = new_high_surface.get_rect(center=(SCREEN_WIDTH // 2, y_pos - 40))
            new_high_surface.set_alpha(int(self.fade_alpha))
            self.screen.blit(new_high_surface, new_high_rect)
        
        score_surface.set_alpha(int(self.fade_alpha))
        self.screen.blit(score_surface, score_rect)
        
        # High score
        high_score_text = f"High Score: {self.high_score}"
        high_score_surface = self.subtitle_font.render(high_score_text, True, LIGHT_GRAY)
        high_score_rect = high_score_surface.get_rect(center=(SCREEN_WIDTH // 2, y_pos + 40))
        high_score_surface.set_alpha(int(self.fade_alpha * 0.8))
        self.screen.blit(high_score_surface, high_score_rect)


class PauseMenu(Menu):
    """Pause menu overlay."""
    
    def __init__(self, screen: pygame.Surface):
        super().__init__(screen)
        self.title = "PAUSED"
        
        # Add menu items
        self.add_item("Resume", "resume")
        self.add_item("Restart", "restart")
        self.add_item("Main Menu", "main_menu")
        
        # Lighter background for overlay
        self.target_alpha = 200
    
    def draw(self):
        """Draw the pause menu as an overlay."""
        # Draw semi-transparent background
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(int(self.background_alpha))
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Draw menu content
        self.draw_title(self.title, -50)
        self.draw_items(350)


class HighScoreMenu(Menu):
    """High score display menu."""
    
    def __init__(self, screen: pygame.Surface):
        super().__init__(screen)
        self.title = "HIGH SCORES"
        self.scores: List[Tuple[int, str]] = []
        
        # Add menu items
        self.add_item("Back", "main_menu")
    
    def set_scores(self, scores: List[Tuple[int, str]]):
        """Set the high scores to display."""
        self.scores = scores[:10]  # Show top 10 scores
    
    def draw(self):
        """Draw the high score menu."""
        self.draw_background()
        self.draw_title(self.title)
        self.draw_scores()
        self.draw_items(500)
    
    def draw_scores(self):
        """Draw the high score list."""
        if not self.scores:
            no_scores_text = "No high scores yet!"
            no_scores_surface = self.subtitle_font.render(no_scores_text, True, LIGHT_GRAY)
            no_scores_rect = no_scores_surface.get_rect(center=(SCREEN_WIDTH // 2, 300))
            no_scores_surface.set_alpha(int(self.fade_alpha))
            self.screen.blit(no_scores_surface, no_scores_rect)
            return
        
        start_y = 250
        for i, (score, date) in enumerate(self.scores):
            y_pos = start_y + i * 30
            
            # Rank
            rank_text = f"{i + 1}."
            rank_surface = self.subtitle_font.render(rank_text, True, LIGHT_GREEN)
            rank_rect = rank_surface.get_rect(right=SCREEN_WIDTH // 2 - 100, centery=y_pos)
            
            # Score
            score_text = f"{score}"
            score_surface = self.subtitle_font.render(score_text, True, WHITE)
            score_rect = score_surface.get_rect(left=SCREEN_WIDTH // 2 - 80, centery=y_pos)
            
            # Date
            date_surface = self.subtitle_font.render(date, True, LIGHT_GRAY)
            date_rect = date_surface.get_rect(left=SCREEN_WIDTH // 2 + 50, centery=y_pos)
            
            # Apply fade alpha and draw
            for surface in [rank_surface, score_surface, date_surface]:
                surface.set_alpha(int(self.fade_alpha))
            
            self.screen.blit(rank_surface, rank_rect)
            self.screen.blit(score_surface, score_rect)
            self.screen.blit(date_surface, date_rect)