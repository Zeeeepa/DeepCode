import pygame
import random
import math

# Game constants
GRID_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20
FOOD_COLOR = (255, 0, 0)
SPECIAL_FOOD_COLOR = (255, 215, 0)
SPEED_FOOD_COLOR = (0, 255, 255)

class Food:
    def __init__(self):
        # Food types with their properties (color, score, effect)
        self.FOOD_TYPES = {
            'normal': {'color': FOOD_COLOR, 'score': 10, 'probability': 0.7},
            'special': {'color': SPECIAL_FOOD_COLOR, 'score': 30, 'probability': 0.2},
            'speed': {'color': SPEED_FOOD_COLOR, 'score': 20, 'probability': 0.1}
        }
        
        # Initialize food properties
        self.position = (0, 0)
        self.type = 'normal'
        self.particles = []
        self.spawn()
        
    def spawn(self, snake_body=None):
        """Spawn new food at random position with weighted food type, avoiding snake body"""
        # Generate valid position that doesn't overlap with snake
        while True:
            # Get random grid position (corrected calculation)
            grid_x = random.randint(0, GRID_WIDTH - 1)
            grid_y = random.randint(0, GRID_HEIGHT - 1)
            x = grid_x * GRID_SIZE
            y = grid_y * GRID_SIZE
            
            # Check if position overlaps with snake body
            if snake_body is None:
                self.position = (x, y)
                break
            
            position_valid = True
            for segment in snake_body:
                if segment[0] == x and segment[1] == y:
                    position_valid = False
                    break
            
            if position_valid:
                self.position = (x, y)
                break
        
        # Select food type based on probability
        rand = random.random()
        cumulative_prob = 0
        for food_type, props in self.FOOD_TYPES.items():
            cumulative_prob += props['probability']
            if rand <= cumulative_prob:
                self.type = food_type
                break
    
    def respawn(self, snake_body):
        """Respawn food avoiding snake body - alias for spawn method"""
        self.spawn(snake_body)
    
    def check_collision(self, snake_head):
        """Check if snake collided with food"""
        food_rect = pygame.Rect(self.position[0], self.position[1], GRID_SIZE, GRID_SIZE)
        snake_rect = pygame.Rect(snake_head[0], snake_head[1], GRID_SIZE, GRID_SIZE)
        return food_rect.colliderect(snake_rect)
    
    def get_score(self):
        """Return score value for current food type"""
        return self.FOOD_TYPES[self.type]['score']
    
    def create_particles(self):
        """Create particle effect when food is eaten"""
        num_particles = 8
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            dx = speed * math.cos(angle)
            dy = speed * math.sin(angle)
            particle = {
                'pos': [self.position[0] + GRID_SIZE/2, self.position[1] + GRID_SIZE/2],
                'vel': [dx, dy],
                'lifetime': 20,
                'color': self.FOOD_TYPES[self.type]['color']
            }
            self.particles.append(particle)
    
    def update_particles(self):
        """Update particle positions and lifetimes"""
        for particle in self.particles[:]:
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['lifetime'] -= 1
            if particle['lifetime'] <= 0:
                self.particles.remove(particle)
    
    def draw(self, screen):
        """Draw food and particles"""
        # Draw food
        food_rect = pygame.Rect(self.position[0], self.position[1], GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, self.FOOD_TYPES[self.type]['color'], food_rect)
        
        # Draw particles
        for particle in self.particles:
            pos = particle['pos']
            color = particle['color']
            size = int(particle['lifetime'] / 4)
            if size > 0:  # Ensure particle size is positive
                pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), size)
    
    def handle_collision(self, snake):
        """Handle snake collision with food"""
        if self.type == 'speed':
            snake.increase_speed()
        snake.grow()
        self.create_particles()
        self.spawn(snake.body)
        return self.get_score()
