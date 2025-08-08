import pygame
import math
import random

# UI相关常量
FONT_PATH = None  # 使用系统默认字体
FONT_SIZE = 24
TITLE_FONT_SIZE = 48
UI_TEXT_COLOR = (255, 255, 255)

# 窗口尺寸 - 修改为与settings.py中的WINDOW_SIZE保持一致
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

# 游戏状态常量 - 修改为字符串常量以与main.py保持一致
GAME_STATE_START = "start"
GAME_STATE_PLAYING = "playing"
GAME_STATE_PAUSED = "paused"
GAME_STATE_OVER = "game_over"

class UI:
    def __init__(self):
        pygame.font.init()
        # Load fonts
        try:
            if FONT_PATH:
                self.game_font = pygame.font.Font(FONT_PATH, FONT_SIZE)
                self.title_font = pygame.font.Font(FONT_PATH, TITLE_FONT_SIZE)
            else:
                raise FileNotFoundError
        except:
            # Fallback to system font if custom font fails to load
            self.game_font = pygame.font.SysFont('arial', FONT_SIZE)
            self.title_font = pygame.font.SysFont('arial', TITLE_FONT_SIZE)
        
        # Initialize UI elements
        self.high_score = 0
        self.transition_alpha = 255  # For fade effects
        self.particles = []  # For particle effects
        
    def update_high_score(self, score):
        self.high_score = max(self.high_score, score)
    
    def create_text(self, text, font, color, pos, centered=True):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if centered:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        return text_surface, text_rect
    
    def draw_score(self, screen, score):
        score_text = f'Score: {score}'
        high_score_text = f'High Score: {self.high_score}'
        
        # Draw current score
        score_surf, score_rect = self.create_text(
            score_text, 
            self.game_font, 
            UI_TEXT_COLOR, 
            (WINDOW_WIDTH - 20, 20),
            centered=False
        )
        score_rect.topright = (WINDOW_WIDTH - 20, 20)
        screen.blit(score_surf, score_rect)
        
        # Draw high score
        high_score_surf, high_score_rect = self.create_text(
            high_score_text,
            self.game_font,
            UI_TEXT_COLOR,
            (20, 20),
            centered=False
        )
        screen.blit(high_score_surf, high_score_rect)
    
    def draw_start_screen(self, screen):
        # Draw title
        title_surf, title_rect = self.create_text(
            'Snake Game',
            self.title_font,
            UI_TEXT_COLOR,
            (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3)
        )
        screen.blit(title_surf, title_rect)
        
        # Draw instructions
        instructions = [
            'Press SPACE to Start',
            'Use Arrow Keys or WASD to move',
            'Press ESC to Quit'
        ]
        
        for i, instruction in enumerate(instructions):
            inst_surf, inst_rect = self.create_text(
                instruction,
                self.game_font,
                UI_TEXT_COLOR,
                (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + i * 40)
            )
            screen.blit(inst_surf, inst_rect)
    
    def draw_game_over(self, screen, score):
        # Update high score
        self.update_high_score(score)
        
        # Draw "Game Over" text
        game_over_surf, game_over_rect = self.create_text(
            'Game Over',
            self.title_font,
            UI_TEXT_COLOR,
            (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3)
        )
        screen.blit(game_over_surf, game_over_rect)
        
        # Draw final score
        final_score_surf, final_score_rect = self.create_text(
            f'Final Score: {score}',
            self.game_font,
            UI_TEXT_COLOR,
            (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        )
        screen.blit(final_score_surf, final_score_rect)
        
        # Draw high score
        high_score_surf, high_score_rect = self.create_text(
            f'High Score: {self.high_score}',
            self.game_font,
            UI_TEXT_COLOR,
            (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 40)
        )
        screen.blit(high_score_surf, high_score_rect)
        
        # Draw restart instruction
        restart_surf, restart_rect = self.create_text(
            'Press SPACE to Restart',
            self.game_font,
            UI_TEXT_COLOR,
            (WINDOW_WIDTH // 2, WINDOW_HEIGHT * 2 // 3)
        )
        screen.blit(restart_surf, restart_rect)
    
    def draw_game_over_screen(self, screen, score):
        """别名方法，用于向后兼容性，调用原始的draw_game_over方法"""
        self.draw_game_over(screen, score)
    
    def create_particles(self, pos, color, num_particles=10):
        """Create particles for visual effects"""
        for _ in range(num_particles):
            angle = math.radians(random.randint(0, 360))
            speed = random.randint(2, 5)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(20, 40)
            self.particles.append({
                'pos': list(pos),
                'velocity': velocity,
                'lifetime': lifetime,
                'color': color,
                'size': random.randint(2, 4)
            })
    
    def update_particles(self):
        """Update particle positions and lifetimes"""
        for particle in self.particles[:]:
            particle['pos'][0] += particle['velocity'][0]
            particle['pos'][1] += particle['velocity'][1]
            particle['lifetime'] -= 1
            
            if particle['lifetime'] <= 0:
                self.particles.remove(particle)
    
    def draw_particles(self, screen):
        """Draw all active particles"""
        for particle in self.particles:
            pygame.draw.circle(
                screen,
                particle['color'],
                (int(particle['pos'][0]), int(particle['pos'][1])),
                particle['size']
            )
    
    def draw_pause_screen(self, screen):
        # Create semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        screen.blit(overlay, (0, 0))
        
        # Draw pause text
        pause_surf, pause_rect = self.create_text(
            'PAUSED',
            self.title_font,
            UI_TEXT_COLOR,
            (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        )
        screen.blit(pause_surf, pause_rect)
        
        # Draw resume instruction
        resume_surf, resume_rect = self.create_text(
            'Press SPACE to Resume',
            self.game_font,
            UI_TEXT_COLOR,
            (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50)
        )
        screen.blit(resume_surf, resume_rect)
    
    def update(self):
        """Update UI elements that need regular updates"""
        self.update_particles()
    
    def draw(self, screen, game_state, score=0):
        """Main draw method that handles all UI states"""
        if game_state == GAME_STATE_START:
            self.draw_start_screen(screen)
        elif game_state == GAME_STATE_PLAYING:
            self.draw_score(screen, score)
        elif game_state == GAME_STATE_PAUSED:
            self.draw_pause_screen(screen)
        elif game_state == GAME_STATE_OVER:
            self.draw_game_over(screen, score)
        
        # Always draw particles if there are any
        self.draw_particles(screen)
